# 06 — Open questions, risks, and a phased roadmap

v0.2 has *decided* multi-target + graphs + perf from day 1. So the open questions
here are *within* that stance, not about it, and the roadmap front-loads the
things v0.1 used to defer.

## A. The remaining hard questions (ranked by how much they gate ship-ability)

### A1. Per-target override granularity: H1 (full-body) vs H2 (primitive-level)? *(the central empirical question)*

The portable Layer-2 body is correct everywhere but won't reach ceilings. How
often can a target reach its ceiling by *swapping primitive bodies* (H2 — the
elegant case, where the kernel structure is shared and the reference shares
structure with every target) vs how often does it need a *whole new body*
(H1 — e.g. the sm_90 GEMM that restructures for TMA+clusters+wgmma)?

**Why this gates everything:** if H2 suffices for most GEMM/attention, the DSL is
small (a primitive library + structure-sharing bodies) and the reference-drift
guarantee (`05` §4) holds tightly. If H1 dominates, the DSL is mostly a
multi-backend template generator and the reference shares only the *contract*
with the native kernels — drift is back to being `verify`-caught, not
structurally impossible.

**This is empirically answerable in Phase 1** by porting ~3 real GEMM/attention
ops and counting how many needed H1. The strawman (`04` Ex.2) shows H1; whether
H2 could have reached the same ceiling is the open probe. Lean: design for H2,
fall back to H1 per-op, *record which dominated* so the answer compounds.

### A2. The lowest-common-denominator landmine (§10) — now the central design problem

v0.1 listed this as a "watch-out." With perf day 1, it's the thing the design
*exists to solve*. A DSL that lowers the same body to CUDA and HIP and produces
two correct, both-slow cards is the failure mode. The mitigations, all required
for v1:

- **Per-target override bodies** (Layer 3) — reaching the ceiling is explicit
  per-target work, not a hoped-for side effect of a shared source.
- **Roofline-grade acceptance** — a native card's `perf.measured` must cite a
  measured roofline fraction; the Phase-1 gate is "≥ a credible fraction of the
  *vendor* ceiling" (AMD graded vs the AMD MFMA ceiling, never vs the NVIDIA
  card — §10). A correct card under ceiling is a failure to fix.
- **Autotune wired from day 1** — the `@targets(..., knobs={...})` space is
  swept at build and the winner written to `perf.measured`. No "ship untuned."

If after Phase 1 the native cards can't reach a credible roofline fraction, the
honest scope reduction is "DSL emits the portable card + the reference; native
ceilings stay hand-written skills" — i.e. the DSL becomes the cold-start
producer, not the perf producer. That's a smaller but still honest product.

### A3. Conditional-node coverage for MoE/sparse (the graph story's hard part)

Graphs win on launch overhead, but MoE routing, sparse attention indexing, and
masked dispatch *branch on data* — and a static graph DAG can't. CUDA's
`cudaGraphAddConditionalNode` (12.2+) covers some of this; HIP's support is
newer and uneven (`07` §4.3). The DSL must:

- Map a guarded block to a conditional node where the target supports it;
- **Reject at build time** (not silently degrade to per-iteration re-launch —
  that throws away the graph benefit *exactly* on the workloads that need it
  most) where it doesn't;
- Offer stream-capture (`07` §4.4) as the explicit escape hatch.

**Resolved (Phase 3, shipped 2026-07-01):** the v1 is **dense-chains-only**,
honestly. `torch.cuda.CUDAGraph` cannot capture host-side `if` on device data —
the GPU→CPU sync invalidates the capture (`cudaErrorStreamCaptureInvalidated`),
and the invalidation is **process-fatal** (no `capture_error_mode` recovers it;
verified across global/thread_local/relaxed). `test_vkl_graph_conditional.py`
probes this in a subprocess (the poisoned state dies with the child) and asserts
the capture is rejected, never silently degraded — the §4.3 honesty rule holds.
Conditional nodes ship when torch/CUDA grows `cudaGraphAddCond` support (the test
is the canary: it flips to "captures"). The dense-chain graph itself is live
(`vkl.graph`), measured at 5–6.5× speedup over sequential on launch-bound chains
(§8 gate, GB10 + H100), bit-exact correctness.

**Still open:** mapping the repo's MoE/sparse workload (`sparse_mla_attention`,
`moe_align_block_size`, `moe_sum_reduce`) to conditional nodes once the upstream
support lands. Lean: re-run the probe per torch release; the canary test drives
it.

### A4. Does the auto-reference scale to the interesting ops? — *RESOLVED 2026-07-02 (the oracle-safety scope line)*

"Same code, two backends" (`05` §4) is obviously true for elementwise/reduce/
GEMM. Less obvious for:
- **Sparse / indexed ops** — gather/scatter semantics may not fall out of a tile
  program cleanly.
- **Online algorithms** (`mha_merge_state`'s online-softmax merge) — the CPU path
  must reproduce the *associativity* the device kernel relies on.

**The scope line (drawn explicitly, not discovered at `verify` time):** split
"data-dependent" by whether the construct preserves the *oracle property* —
one source lowers to a bit-exact torch reference **and** a device kernel, both
deterministic. Three cases:

| Case | Construct | Oracle-safe? | Disposition |
|---|---|---|---|
| (a) data-**addressing** | `Gather` / `Slice` / `Concat` — the index is an *input* tensor; the op is pure, parallel, deterministic (`base[idx]` ≡ `tl.load(base + idx)`) | **yes** | **IN the math IR** (added 2026-07-02) |
| (b) deterministic online **monoid** | flash-attention's `(m,l,o)` online-softmax combiner — associative, so any eval order agrees up to fp rounding | **yes, *if* the reference IS the online algorithm** (not a naive softmax) | IN, future — the reference must reproduce the device associativity |
| (c) data-**selection** | top-k, sort, RNG (sampling) — the *set* of chosen elements depends on values; RNG is non-deterministic by construction | **no** | **stays hand-path** — cannot be an "obviously-correct pure DAG" |

**What shipped for (a):** `Gather(base, index, axis)`, `Slice(base, axis,
start, stop)`, `Concat(a, b, axis)`, `Unsqueeze(base, axis)` are math-IR nodes
(`ir/math.py`), evaluated by the torch oracle (`lower/mathbody._TorchEval`) and
lowered by BOTH the flat-1D (`elementwise`) AND the multi-dim
(`_TritonGenMultiDim`) Triton codegen — the latter decomposes each lane's flat
offset into per-axis coords so every addressing node computes its own address.
The `Gather` now supports an **N-D index** (the index's full shape replaces the
gathered axis, `index_select` placement), so `base[page_table]` with a 2-D index
lowers exactly. Two showcases, both verified bit-exact + deterministic on an
NVIDIA GB10 (sm_121) with reference↔triton parity at `max_rel=0.0`:
  * **`apply_rope`** (#68) — a 1-D index gather (`cache[positions]`) plus
    `Slice`/`Concat`/`Unsqueeze` for the rotate-half products.
  * **`paged_kv_gather`** (#71 building block) — a **2-D index** gather
    (`pool[page_table]`), the unpage primitive behind paged attention: pure copy,
    zero arithmetic, so the gate is `max_abs=0` (any drift is a codegen bug).
The data-addressing family closes the loop on real hardware, not just on the
CPU oracle — and it factors out the one DSL-expressible slice of paged attention
(the ragged `cu_seqlens` reduction that makes full #71 hand-path stays there).

**The hard line, restated:** the math IR's value is that its CPU lowering is
*obviously* the reference because every node is pure pointwise/reduce/mma (+ now,
indexing). Data-*selection* (top-k/sort/RNG) is hand-path **not** because it is
hard to codegen but because it cannot be made into that kind of obviously-correct
oracle — so it would reopen the very drift gap the IR exists to close. Case (b)
is admitted only under the A4 caveat (reference reproduces the online order).

**What shipped for the deterministic softmax prefix:** `temperature_softmax`
(#69/#70 prefix) is now a rowwise math-IR body: per-row temperature broadcast,
`reduce_max`, `exp`, `reduce_sum`, and normalize. It proves multiple reductions
compose in one rowwise body without admitting data-selection semantics.

**Still hand-path (per the line above):** `topk_softmax` (#70, top-k half),
`sampling` (#69, RNG+sort), `dsa_topk` (#54, top-k). `flash_mla` (#53) and
`gqa_attention` (#71) split: their *compute* (MMA + online-softmax) is case (b)
and DSL-eligible once the structured reduce lands; their paged-KV gather is case
(a) and DSL-eligible now; their putative top-k/selection parts stay hand.

### A5. The internal IR (forced by multi-target day 1) — *now answered by [`09`](09-agent-editable-ir.md)*

You cannot fan Python-AST out to Triton + CUDA + HIP + graphs without an internal
IR. v0.1 could defer this; v0.2 cannot. The deciding question is no longer
"bespoke vs MLIR" — it is **"compiler-pass-optimized or agent-editable?"** The
whole point of the IR (per the directive) is that an AI agent *compiles* it to
push performance, so the IR must be designed for an LLM editor: token-compact,
named-concept (the L0–L5 hierarchy from `08`), typed-hole, cost-annotated. MLIR
is the wrong shape for that (verbose dialects, SSA, regions — built for
algorithmic passes). **Lean (decided): a bespoke agent-editable schedule IR
(`09`), split into a frozen math layer (the correctness oracle / auto-reference)
and an editable schedule layer; do not adopt MLIR unless a future need forces it.**
The edit primitives (`set_knob`, `retile`, `map_to`, `add_stage`, …) are the
MCP tools the skills call, making `autotune-knob-sweep` / `map-to-matrix-cores` /
`tune-for-cdna` programmatic instead of prose. Phased inside the roadmap below
(`09` §10): knob/retile edits first (cheapest, biggest win), then `map_to`, then
the cost model + `tuning_trace` compounding loop.

### A6. Trust model for DSL-emitted cards + graphs

§11 already asks how many confirmations before an agent-authored card
auto-publishes. DSL-emitted cards have the same question *plus* a new
amplification: a single DSL bug emits many wrong cards at once (the §4.2
swizzle-helper traceability risk, amplified across backends and graph nodes).
**Lean:** DSL-emitted cards pass the *same* gates; the DSL's own test suite
(round-trip spec equivalence + reference-equivalence + roofline-floor on a frozen
op set) is the additional guard — exactly §7.3's frozen-replay discipline, now
applied to the *producer*, mirroring how it's already applied to skills.

### A7. Graph-card schema (new artifact or namespaced field?)

Is a graph a new artifact type ("Composition Card") or a namespaced
`launch.graph` field on the Impl Card (`07` §7)? Lean: namespaced field until a
second use case forces a split. Keep the standard core thin (§8 discipline).

### A8. Naming

**RESOLVED (2026-06-30):** `vkl` = the **Vibe Kernel Language** — contract-native,
agent-editable kernel authoring DSL. The earlier placeholder `xtl` ("xkernels
tile language") was retired in a full rename (package, modules, tests, docs, and
committed-card source paths all `xtl` → `vkl`). "Vibe" signals the design target:
productive enough to author by feel — human or agent — while the frozen math IR +
decidable schedule gate keep it honest.

## B. The §10 landmines, as a publish checklist

Before any DSL card (single-kernel *or* graph) is allowed to publish:
- [ ] Hand-written cards still work and remain first-class. (Not a gatekeeper.)
- [ ] No `warp=32` / `smem` / `tensor_cores` literal in any emitted multi-arch
      source path. (Wave size is a parameter.)
- [ ] Emitted JSON uses the *exact* arch vocabulary `find_impl` expects; graph
      metadata is in *namespaced* fields only. (No renamed standard fields.)
- [ ] The reference is auto-derived and round-trips (dense families), or the
      hand-written reference is explicitly flagged (irregular families). (No
      silent drift.)
- [ ] DSL cards pass the *same* shape sweep and `verify_parity` gate. (No looser
      tolerance.)
- [ ] **Native DSL cards are graded against the vendor roofline, not "it ran on
      both."** A correct-but-under-ceiling native card is a failure to fix.
- [ ] Graph cards beat their sequential-launch baseline; a slower graph is a bug.

## C. Phased roadmap (multi-target + graphs + perf from day 1)

v0.1 phased "Triton → multi-target." v0.2 collapses that: multi-target is in
Phase 1, because deferring it would mean shipping a DSL that doesn't test its own
thesis.

### Phase 0 — Invariants & scope *(CPU-only, days)*
- Pick the name (A8).
- Resolve A1's *measurement plan* (not the answer): agree the Phase-1 deliverable
  counts H1-vs-H2 per op.
- **Prove the round-trip invariant first** — DSL header ⇄ JSON spec equivalent
  via existing ingest; auto-reference ⇄ hand-written reference bit-equivalent on
  a frozen op. The whole effort rests on this; prove it before any lowering.

### Phase 1 — Multi-target GEMM, one native ceiling, perf-graded *(GPU: NVIDIA + AMD)*
- Build the **internal IR** (A5) — forced, not optional.
- Lowerings: **Triton (portable) + CUDA + HIP**, all from one source.
- Op scope: **one GEMM family** (`gemm_bf16`) with a **per-target sm_90 override
  and a CDNA3 override** (`04` Ex.2).
- Deliverable: `vkl build gemm_bf16` emits three cards; all pass `verify`;
  `verify_parity` passes; **the native cards hit a credible fraction of their
  vendor roofline** (A2). Measure H1-vs-H2 (A1).
- **Gate to Phase 2:** the roofline fraction. If native DSL cards can't reach it,
  execute the A2 scope reduction (DSL = cold-start producer only) *before*
  broadening.

### Phase 2 — Graph capture + the dense composition milestone *(GPU: NVIDIA + AMD)*
- Lowering: Layer 4 — **CUDA graphs + HIP graphs** via explicit construction
  (G1), parameter nodes (§4.2), conditional nodes where supported (§4.3).
- Op scope: a **2–3 kernel dense chain** (`rmsnorm → gemm → activation`, `04`
  Ex.3) captured on both backends; `verify` runs the whole graph; `verify_parity`
  passes.
- Deliverable: the captured graph **beats the sequential-launch baseline** (B
  checklist). Probe conditional-node coverage on the MoE/sparse family (A3) —
  scope graphs to dense chains if coverage is poor.

### Phase 3 — Breadth + composition + numerics-as-data
- Op scope: the rest of the `meta/docs/` families — norms (incl. `dual_rmsnorm`
  as the simplest), MoE combine/reduce, sparse attention (with the A4
  auto-reference scope line drawn).
- Implement Axis E2 (named epilogue hooks with the §10 shape-change rule as a
  static error) and Axis F (numerics checked against the precision path;
  `cross_backend_rtol` *suggested* from the dtype path — a concrete attack on
  §11's open question).
- Expand Axis B toward declarative skeletons for the GEMM/attention families
  *if* Phase 1 showed H2 (primitive-level override) suffices.

### Phase 4 — Agent-authoring loop (the §6 payoff)
- Instrument: does DSL-authoring reduce median-iterations-to-verify-pass vs
  hand-authoring, per §9's metrics? (The agent-native justification, A4 in v0.1.)
- If yes: wire `vkl build` into `tile-a-gemm` / `port-cuda-to-hip` /
  `add-epilogue-fusion` / `fuse-elementwise-chain` as the cold-start step, so
  skills produce a correct multi-target + graph-capable card first and tune second.

### Explicitly *not* in scope for any phase
- A standalone language grammar (Axis A2) — rejected.
- Primitive-authoring (Axis D2) — deferred until a real op needs it.
- Replacing hand-written cards / making the DSL mandatory — never (§8.4, §10).
- Stream-capture as the *default* graph mode (G2) — escape hatch only.

## D. The smallest experiment that could kill the idea

Before committing to Phase 1, run this probe (a few days, not weeks):
> Hand-translate **`gemm_bf16` with a sm_90 override** into the strawman syntax
> (`04` Ex.2) **and a 2-kernel `rmsnorm_gemm` graph** (`04` Ex.3). *Manually*
> emit, for each: the JSON spec, the auto-reference, the per-target cards (incl.
> the graph card), and the host graph-construction code. Confirm:
 (a) `verify` + `verify_parity` pass with **zero** hand-editing of JSON;
 (b) the native sm_90 card hits a credible roofline fraction;
 (c) the captured graph beats the sequential baseline;
 (d) **(the `09` addition)** sketch the edit sequence (`add_stage` →
 `promote_override` → `map_to` → `retile`, per `09` §8) an agent would issue to
 push the portable GEMM to the sm_90 ceiling, and confirm each edit's
 preconditions (§5) are locally decidable from the schedule IR alone.

If (a) fails, the contract-native thesis is weaker than claimed → shrink to
"auto-reference only." If (b) fails, multi-target day 1 is over-ambitious →
execute the A2 scope reduction. If (c) fails, graph capture isn't paying off on
dense chains → defer graphs to the MoE/sparse conditional-node investigation.
If (d) fails — i.e. the perf-pushing edits *aren't* locally decidable and the
agent would need global reasoning to propose them — then the agent-editable-IR
thesis (`09`) is weaker than claimed → shrink to "knob-sweep only" (the edits
that are trivially local) and leave level-mapping as a skill-driven, human-checked
path. **Any of these is an honest outcome; none is a reason to abandon the whole
idea — they're scope-correctors, and the probe is cheap enough to run before
building the IR.**
