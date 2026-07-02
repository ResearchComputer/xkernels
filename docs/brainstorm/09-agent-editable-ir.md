# 09 — The IR is agent-editable: a schedule IR for autonomous perf-pushing

> The directive: the IR should be **"compilable" by an AI agent** to automatically
> push for performance. This doc takes that literally and designs for it.

## 0. The thesis: design for a different editor

Every existing compiler IR (LLVM, MLIR, SPIR-V, Triton's IR) is optimized for
**algorithmic passes** — transformation functions written by humans, invoked in a
fixed pipeline. They are *terrible* editing surfaces for an LLM: verbose
(dialects, SSA, regions), global (a pass rewrites whole graphs), and opaque (an
edit's effect is only knowable after the pass runs).

An LLM agent is a different editor with different strengths and different failure
modes:

| Editor | Good at | Bad at |
|---|---|---|
| Compiler pass | Global, algorithmic, bit-exact transforms | Anything not pre-encoded |
| **LLM agent** | **Local, named-concept edits; reading intent; ranking candidates from a cost hint** | **Global restructuring; bit-exactness; counting** |

So an *agent-editable IR* is not "MLIR but friendlier." It is a different design
target, with four constraints that follow directly from how LLMs actually work:

1. **Compact + local.** Small token budget; an edit touches a *local* region with
   a *local*, predictable effect. No whole-graph rewrites as the unit of work.
2. **Named concepts over structural soup.** Nodes are the L0–L5 hierarchy objects
   from `08` (tile, wave, matrix-engine op, stage), not SSA values. The agent
   reasons over the *same vocabulary the skills and the contract use*.
3. **Typed holes + declared spaces.** The search space (knobs, legal
   instructions, legal levels) is *declared and bounded*, so an edit is
   checkable, not free-form. The agent can't propose something undecidable.
4. **Closed-form cost signals per node.** Each node carries/derives a roofline,
   bandwidth, and occupancy estimate so the agent can **predict before it
   measures**, ranking edits instead of measuring blind.

**The payoff:** "push for performance" becomes *an accumulation of local,
named, checkable edits* — exactly the regime where LLMs are reliable — rather
than a global rewrite (where they aren't). That is what makes autonomous
perf-pushing plausible instead of wishful.

## 1. Two layers, one of them frozen

The IR is split so that **perf-pushing edits cannot risk correctness by
construction**:

```
   MATH IR      (frozen)        the WHAT — derived from the Layer-2 compute body
   ─────────                   mma / load / store / reduce / pointwise, typed
                               (dtype, shape). This is the correctness oracle;
                               the auto-reference (05 §4) lowers from here.
                               THE AGENT DOES NOT EDIT THIS.

   SCHEDULE IR  (editable)      the HOW — over the L0–L5 hierarchy
   ─────────                   tile shapes, level mappings, pipeline stages,
                               copy atoms, swizzles, knob bindings.
                               THE AGENT EDITS THIS.
```

This separation is the safety property that makes autonomous tuning safe: **a
schedule edit changes *how* the (unchanged) math is realized, never *what* is
computed.** Correctness is preserved by construction; `verify` only needs to
re-confirm the lowering is sound and re-measure perf, not re-derive correctness
from scratch each iteration. (The lowering itself can still have bugs, so `verify`
still runs — but the *agent's edit* is not where correctness risk lives.)

This is also why the auto-reference (`05` §4) is load-bearing here: the math IR
*is* the reference. Edit the schedule all you want; the oracle doesn't move.

## 2. The schedule-IR node vocabulary (L0–L5 made editable)

Every node is a hierarchy object from `08`, now with editable fields and a
cost annotation:

| Node | Editable fields | Cost annotation |
|---|---|---|
| `Tile` | `shape`, `level` (L0/L2) | bytes, occupancy contribution |
| `MapTo` | `op` (from math IR), `level` (L4 FMA / L5 engine), `instruction?` (wgmma/mfma/…) | peak FLOPs at that level/instr |
| `Stage` | `buffer`, `space` (register/scratch/dsmem), `depth` | scratch bytes, pipeline throughput |
| `CopyAtom` | `src→dst`, `width` (vectorize), `swizzle?` | peak bytes/s, bank-conflict risk |
| `Reduce` | `op`, `axis`, `level` (L3 wave / L2 CTA / L0 cross-CTA) | latency, sync cost |
| `Knob` | `name`, `value` (must be in declared space) | — |

A schedule is a graph of these. The agent's edits are *local surgeries* on this
graph. Crucially, the vocabulary is **the same one the skills, the cards, and the
contract speak** (`wave`, `mfma`, `num_stages`, `waves_per_eu`) — there is no
translation step between "what the skill says to do" and "what the edit does."
That is the §8.1 MCP-tools-as-the-procedure principle, now at the IR layer.

## 3. The edit primitives — the agent's "instructions"

Edits are **named, precondition-checked, diff-producing** operations. They are
the MCP tools the skills call (`x-kernel-lib.tools` in the SKILL.md frontmatter,
§7.1), made concrete:

| Edit | What it does | Preconditions checked |
|---|---|---|
| `set_knob(name, value)` | bind a specialization knob | value ∈ declared `specialization_knobs` space |
| `retile(tile_id, shape)` | resize a tile | shape divisible by target's L5 native shape; ≤ scratch budget |
| `map_to(op, level, instruction)` | move a math op to a level/instr (FMA@L4 → wgmma@L5) | instruction legal for (target, dtype, shape); dtype path consistent with `reduce_dtype` |
| `add_stage(buffer, depth)` / `remove_stage` | change pipeline depth | scratch budget; producer/consumer still well-formed |
| `set_copy_atom(copy_id, width, swizzle)` | change vectorize width / swizzle | width legal for arch; swizzle legal for space |
| `reduce_level(reduce_id, level)` | move a reduction across L3/L2/L0 | axis legal at that level |
| `promote_override(target, arch)` | **H2→H1 escape**: spawn a full override body when schedule edits can't reach the ceiling | target exists; math IR intact (override must still satisfy it) |

Each edit emits a **structured diff** (node added/removed/changed, cost-delta
predicted) that the agent reads to understand what it just did — closing the
"LLMs can't reliably introspect their own global changes" gap by making the
change *local and named*.

`promote_override` is the acknowledgement that not every ceiling is reachable by
local edits: when the cost model says "compute-bound at L4, but L5 mapping is
blocked by structure" (e.g. needs L1 clusters or TMA descriptors the schedule
can't express), the agent escalates from H2 (schedule edits) to H1 (freehand
override body). The override body is still type-checked against the **math IR**,
so even the escape hatch preserves the correctness-by-construction property.

## 4. The cost model: predict before you measure

This is what makes the agent's search *informed* rather than random. Every
schedule node carries a **closed-form** cost estimate derived from the node +
the target arch:

- `MapTo` → peak FLOPs at that level/instruction (e.g. wgmma@sm_90 = X TF/s;
  MFMA@cdna3 = Y TF/s; FMA@L4 = the much lower scalar ceiling).
- `CopyAtom` / `Stage` → peak bytes/s for that space+width, with a
  bank-conflict penalty factor for known-bad swizzles.
- aggregate → predicted occupancy from total register + scratch pressure.

The agent reads these to **rank candidate edits before launching any kernel**.
A `map_to(op, L5, wgmma)` edit that the cost model says moves compute from 20%
to 80% of roofline is obviously worth trying; a `retile` that costs scratch
without changing the bottleneck is ranked low.

**The honest caveat, front and center:** cost models are wrong. Roofline
predictions are 60–80% accurate on a good day. **The cost model guides; `verify`
decides.** Same discipline as the substrate's "diagnose from the profile, don't
guess" — but here the profile *feeds back into the model*:

```
profile (real occupancy/stall via use-rocprof-compute / use-nsight-compute)
   → write measured numbers back onto the IR nodes as annotations
   → agent's next prediction is calibrated against measured reality
```

So the cost model is itself a **compounding loop** (`05`): it gets less wrong
every time a profile runs, because the profile's numbers become the next
prediction's priors. This is the §6.2 loop applied to the *predictor*, not just
the result.

## 5. The check gate (edits are validated, not trusted)

Before any edit lowers, the gate validates (these map 1:1 to `08` §7's
auto/checked/rejected):

- **Math IR unchanged** — the edit touched only schedule nodes (correctness
  oracle intact).
- **Tile divisibility** — every `Tile` shape divisible by the target's L5 native
  shape (no `BLOCK_M=96` on a `wgmma m=64` target).
- **dtype path consistency** — every reduction's accumulator dtype matches
  `numerics.reduce_dtype` (Axis F, `03`).
- **Memory-space legality** — no `dsmem`/`cluster` on AMD; no TMA descriptor
  below sm_90.
- **Knob containment** — values inside the declared `specialization_knobs`.
- **Scratch budget** — total `Stage` bytes ≤ arch scratch size.

A failed check **rejects the edit with a reason** — exactly `find_impl`'s
`reject_reasons` pattern (§3.2), now at the edit layer. The agent learns *why*
its edit was illegal, which is training signal for the next proposal. Edits that
pass are guaranteed to lower to *something compilable*; whether that something is
*fast* is what `verify` measures.

## 6. The closed loop — autonomous perf-pushing

```
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. LOAD    ir.load(card_id) → schedule IR + cost annotations  │
  │ 2. READ    agent reads cost model + (if available) profile     │
  │ 3. PROPOSE a named edit (skill-driven — see §7)                │
  │ 4. CHECK   gate validates (§5): reject-with-reason or apply    │
  │ 5. LOWER   schedule+math IR → target source → compile          │
  │ 6. VERIFY  correctness (unchanged by construction) + perf ms   │
  │ 7. DECIDE  improved? → record_measurement + append trace       │
  │ 8. PROFILE (periodic) → re-annotate IR nodes (§4 feedback)     │
  └──────────────────────────────┬───────────────────────────────┘
                                 └──→ loop until perf plateau or budget
```

Step 7's **trace** is the compounding artifact: the *ordered sequence of named
edits* that took this card from its starting perf to its current perf, each with
its measured delta. Stored in `provenance.tuning_trace` (namespaced field). The
next agent facing the same `(op, arch, shape)` reads the trace and **skips the
dead-ends** — it doesn't re-discover that `num_stages=4` overflowed scratch on
sm_90, because the trace records "tried, rejected: scratch budget." This is the
§6.2/§7.3 compounding loop applied to the *tuning process itself*, not just the
result. It is the single biggest agent-loop win in the whole design.

## 7. The skills become edit sequences (programmatic, not prose)

Today the skills are SKILL.md prose the agent reads and improvises around. With
the edit primitives as MCP tools, **each skill is a named sequence of edits** —
the `x-kernel-lib.tools` frontmatter binding (§7.1) becomes a real program:

| Skill | Edit sequence |
|---|---|
| `autotune-knob-sweep` | batch of `set_knob` over the declared space, ranked by cost model, measured; winner → `perf.measured` |
| `map-to-matrix-cores` | `map_to(op, L5, wgmma\|mfma)` after checking dtype/shape prerequisites (fp8→e4m3fnuz on CDNA3, etc.) |
| `tune-for-cdna` | `retile` for 64-wide → `map_to MFMA` → `set_knob(waves_per_eu)` → `add_stage(LDS)` |
| `diagnose-memory-bound` | read cost annotation (stall = "wait for memory") → propose `add_stage` / `set_copy_atom(wider)` / `reduce_level` |
| `diagnose-low-occupancy` | read cost annotation (occupancy low) → propose `retile(smaller)` / `reduce_stage` / fewer registers |
| `port-across-arch` | load the source arch's schedule IR, re-bind `level`/`instruction`/`space` to the target arch's vocabulary, re-check, re-measure |

The prose SKILL.md body remains (for agents that aren't wired to the MCP tools —
§8.4 graceful degradation), but a wired agent *executes* the edit sequence
instead of improvising. The IR is what makes the skills reliable: the procedure
is no longer "the agent re-derives what `waves_per_eu` means each time," it's
"the agent issues `set_knob('waves_per_eu', 2)` and the gate checks it."

## 8. Worked example: agent pushes a portable GEMM to the sm_90 ceiling

Start: `gemm_bf16.triton@1.0.0`, correct, **40% of sm_90 roofline**. The agent's
trace (each line is one edit + its check + measured delta):

```
profile → occupancy OK, stall = "long scoreboard" (wait on global load)
   ⟹ diagnose-memory-bound fires
1. add_stage(scratch, depth 2→3)            check ✓ (scratch budget ok)
   lower → verify: 40% → 52%                trace: [+12%, memory-stall fixed]
2. set_copy_atom(a_tile, width 32→128)      check ✓ (128-bit legal)
   lower → verify: 52% → 58%                trace: [+6%, coalescing]
   cost model now says: compute-bound, L5 engine idle
   ⟹ map-to-matrix-cores fires, BUT portable body is arch:any (no L5 wgmma)
3. promote_override("cuda", nvidia_sm90)    check ✓ (math IR intact)
   → spawns sm_90 override body seeded from current schedule
4. map_to(mma, L5, wgmma, shape=(64,N,16))  check ✓ (64|BLOCK_M; k=16|BLOCK_K)
   lower → verify: 58% → 74%                trace: [+16%, tensor engine on]
5. retile(BLOCK_M, 128→256)                 check ✓ (256%64==0)
   lower → verify: 74% → 79%                trace: [+5%, better L5 utilization]
   plateau — cost model says next levers (cluster, TMA desc) need H1 restructure
6. (optional, H1) agent freehand-edits override body to add ctx.cluster(2)
   check ✓ (math IR still intact) → verify: 79% → 83%
record_measurement(arch=sm90, tflops=..., source=run:...) 
provenance.tuning_trace = [steps 1–6 with deltas]
```

Every step is a named edit, every check is local, every delta is measured, and
the **whole trace is replayable and compoundable**. The next agent facing
`gemm_bf16` on sm_90 reads the trace and starts at step 6's plateau, not at 40%.
That is the compounding loop earning its keep.

## 9. Honest risks (this is a bet on LLM-as-editor)

- **The local-edit reliability bet.** The whole design assumes perf-pushing
  decomposes into local named edits — the regime where LLMs are good. This holds
  for knob sweeps, level-mapping, staging, swizzle (the H2 cases, ~most ops).
  It **does not hold** for H1 restructures (step 6 above), where the agent
  freehand-edits an override body and is back to "improvising CUDA." The IR
  doesn't fix that; `verify` remains the gate, and H1 is where agent-authored
  kernels are least reliable. Honest scope: the IR automates the H2 majority;
  H1 stays a human-in-the-loop or a stronger-model task.
- **Cost-model wrongness.** Predictive only; `verify` decides. Mitigated by the
  profile-feeds-back loop (§4), but on a cold start the model is ~roofline-accurate
  and occupancy-naive. Don't oversell the prediction; sell the *measured-delta
  trace*.
- **The profiling dependency.** The cost annotations are sharpest when fed by
  real `rocprof`/`ncu` numbers (the `use-rocprof-compute` / `use-nsight-compute`
  skills). On a CPU-only box the agent works from the *analytical* roofline
  (weaker). This matches the existing CPU-doable gate (`author-an-op-spec` is the
  only CPU-satisfiable skill today); the IR's knob-edit path is similarly
  CPU-doable, the profile-fed path is GPU-gated.
- **This is a real compiler, not a weekend.** Edit primitives + check gate +
  cost model + lowering is a substantial build. Phase it (§10): `set_knob` +
  `retile` first (cheapest, covers autotune-knob-sweep), then `map_to` (covers
  map-to-matrix-cores), then the full schedule + cost model.

## 10. Phasing (inside the `06` roadmap)

The IR lands inside `06`'s Phase 1–3, in order of leverage-per-build-cost:

- **Phase 1 (with the multi-target GEMM):** `set_knob` + `retile` + the check
  gate's divisibility/budget rules. This alone makes `autotune-knob-sweep`
  programmatic and is the biggest agent-loop win for the least build.
- **Phase 2 (with graph capture):** `map_to` + `add_stage` + `set_copy_atom` —
  the edits that map ops to the matrix engine and restructure staging. Covers
  `map-to-matrix-cores` and the diagnose skills' edit paths.
- **Phase 3 (breadth):** the cost model (§4) + profile-feeds-back loop + the
  `tuning_trace` provenance field. This is where autonomous perf-pushing becomes
  *compoundable* across tasks.
- **Phase 4:** `promote_override` and the H1 freehand path — the escape hatch,
  last, when the H2 majority is proven.

## 11. What this decides in `06` A5

`06` A5 asked "bespoke IR vs MLIR?" This doc reframes it: **the deciding factor
is agent-editability**, and MLIR is the wrong choice *because* it's optimized for
algorithmic passes (verbose dialects, SSA, regions — token-expensive and
global). The bespoke IR is bespoke *specifically* to be token-compact,
named-concept, typed-hole, cost-annotated — the four constraints in §0. That is
not "full control" as a vague benefit; it is a hard requirement derived from the
editor. **Lean (upgrading `06` A5): bespoke agent-editable schedule IR; do not
adopt MLIR unless a future need (a 4th target, a polyhedral pass) forces it, and
even then keep the agent surface as a thin editable view over whatever IR is
underneath.**
