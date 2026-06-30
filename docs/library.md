# Agent-Native Cross-Platform GPU Kernel Library — Design Spec

**Status:** Draft v0.2 (adds multi-backend: NVIDIA + AMD)
**Author:** design sketch
**One-line thesis:** Build a kernel library whose primary consumer is an LLM agent, not a human. Every artifact must let an agent decide *does this apply?*, *how do I specialize it?*, and *how do I know it's correct and fast?* — without running anything. Then make the run loop deterministic, structured, and cheap, so that successful work compounds instead of being re-derived per task.

**Cross-platform stance (the central architectural decision):** Portability lives in the **contract, not the source.** A single backend-agnostic **Op Spec** defines the operation's semantics, applicability, reference implementation, and numerical tolerances *once*. Multiple **Implementation Cards** — native CUDA for NVIDIA, native HIP/ROCm for AMD, optionally a portable-DSL build — each *satisfy* that one spec and are validated against the same reference. We treat **functional portability** (same result everywhere) as a guarantee, and **performance portability** (fast everywhere) as explicitly *not* free — perf is measured and tuned per architecture. Backends in scope: NVIDIA (CUDA, sm_80/sm_90, tensor cores, TMA, wgmma) and AMD (HIP/ROCm, CDNA gfx9xx, matrix cores/MFMA, LDS). The model is designed to admit further backends (Triton/SYCL/Metal) without schema changes.

---

## 1. Goals, non-goals, principles

### 1.1 Goals
- A kernel corpus that an agent can **search, select, specialize, compose, and verify** with minimal generation of new CUDA.
- **Compounding value:** each solved task makes future tasks cheaper (measured tunings, new skills, new cards get written back).
- **Machine-checkable correctness** as a first-class contract, not an afterthought.
- **Deterministic, structured feedback** at every step (retrieval, compile, correctness, profiling) so the agent reasons over data, not prose.

### 1.2 Non-goals
- Not a human-facing "nice docs" library. Human readability is welcome but secondary to machine legibility.
- Not a single mega-kernel with thousands of template flags. The validity surface must stay reason-able.
- Not an autotuner replacement. We declare the tuning space; existing tools can search it.
- Not betting on raw model capability to "figure out layouts." Encode the constraints.

### 1.3 Design principles
1. **The contract is the product.** A fast kernel behind an ambiguous interface is agent-hostile.
2. **Reject before you compile.** Applicability must be decidable from metadata alone.
3. **Compose over generate.** Specializing known-good primitives beats emitting fresh PTX.
4. **Determinism in the feedback path.** Stochastic benchmarks poison the learning loop.
5. **Everything that worked becomes reusable.** One-off wins get promoted to cards, tunings, or skills.

---

## 2. Two-level model: Op Spec + Implementation Cards

To be cross-platform, the library splits into two artifact types. This split *is* the portability strategy.

- An **Op Spec** is backend-agnostic. It defines the operation's semantics, input/output contract, applicability constraints, **the single reference implementation**, and **numerical tolerances**. There is exactly one Op Spec per logical operation.
- An **Implementation Card** is backend-specific. It is the source plus the per-architecture details: which backend/arch it targets, its tuning knobs, its measured performance, and the primitives it uses. Many Implementation Cards point at one Op Spec — e.g. a CUDA card and a HIP card for the same GEMM.

The rule that makes portability a guarantee: **every Implementation Card is validated against its Op Spec's one reference and tolerances.** Correctness is defined once, centrally; backends compete only on performance.

### 2.1 Op Spec schema (backend-agnostic)

```yaml
# op_spec.schema
id: string                      # e.g. "fused_rmsnorm_matmul@1.4.0"
name: string
version: semver
op:
  signature: string            # "rmsnorm(x) @ w  (fused epilogue)"
  canonical_op: enum           # gemm | conv2d | rmsnorm | attention | reduce | ...
  fusions: [enum]              # [rmsnorm_prologue, bias, gelu]

inputs:
  <arg_name>: { dtype: [enum], layout: enum, rank: int, shape_symbols: [string] }
outputs:
  <arg_name>: { dtype, layout, rank, shape_symbols }

constraints:                    # decidable from shape/dtype alone; backend-agnostic
  - "K % 8 == 0"
  - "dtype(x) == dtype(w)"
preconditions:                  # runtime invariants the caller guarantees
  - "x is contiguous"
  - "no NaN/inf in inputs"

numerics:                       # defined ONCE; applies to every backend
  reference: string            # id of the reference impl (see §5) — backend-neutral (NumPy/PyTorch)
  rtol: float
  atol: float
  notes: string                # e.g. "accumulate in fp32"
  cross_backend_rtol: float    # how closely two backends must agree with each other (§5.3)

shape_sweep: string             # id of the mandatory correctness sweep manifest
composes_with: [op_id]          # composition is reasoned at the op level
```

### 2.2 Implementation Card schema (backend-specific)

```yaml
# impl_card.schema
id: string                      # e.g. "fused_rmsnorm_matmul.cuda@1.4.0" / ".hip@1.4.0"
implements: op_id               # the Op Spec it must satisfy
backend: enum                   # cuda | hip | triton | sycl | ...
                               # NOTE: `backend` is the vendor-LANGUAGE slot
                               # (cuda = any NVIDIA-native CUDA, hip = any
                               # AMD-native ROCm/HIP). It does NOT distinguish
                               # authoring technology: a hand-written C++
                               # extension (ops/ffn/cuda/) and a CUTE DSL
                               # kernel (ops/*/cute/) BOTH register `backend:
                               # cuda`. The source SUBDIR name is the authoring-
                               # tech hint (`cuda/` = cpp extension, `cute/` =
                               # CUTE DSL, `triton/` = Triton). `Backend` stays
                               # vendor-language because the contract is about
                               # portability targets, not toolchains.
arch:
  family: enum                 # nvidia_sm121 | nvidia_sm100 | nvidia_sm90 | nvidia_sm80 | amd_cdna3 | amd_cdna2 | any
  requires: [enum]             # tensor_cores | tma | cluster   (nvidia)
                               # matrix_cores | mfma            (amd)
  wave_size: int               # 32 (NVIDIA warp) | 64 (AMD wavefront) — affects tiling/reductions
  scratch: { kind: enum, bytes: int }   # kind: smem (nvidia) | lds (amd)

specialization_knobs:           # per-backend; spaces differ across architectures
  BLOCK_M: { type: int, choices: [64, 128, 256] }
  BLOCK_N: { type: int, choices: [64, 128, 256] }
  BLOCK_K: { type: int, choices: [32, 64] }
  waves_per_eu: { type: int, choices: [1, 2] }   # amd-specific example
  num_stages: { type: int, choices: [2, 3, 4] }

perf:
  regime: string               # per-arch; "fast on NVIDIA" does not imply "fast on AMD"
  roofline: enum               # compute_bound | memory_bound | latency_bound
  measured:                    # written back per (arch, shape, dtype) by the compounding loop (§6)
    - { arch: nvidia_sm90, shape: {M: 4096, N: 4096, K: 4096}, dtype: bf16,
        knobs: {BLOCK_M: 128, ...}, tflops: 412, achieved_bw_pct: 0.71, source: "run:..." }
    - { arch: amd_cdna3,   shape: {M: 4096, N: 4096, K: 4096}, dtype: bf16,
        knobs: {BLOCK_M: 256, ...}, tflops: 388, achieved_bw_pct: 0.69, source: "run:..." }

uses_primitives: [primitive_id]
supersedes: [impl_card_id]
provenance:
  authored_by: enum            # human | agent | hybrid
  skill_used: [skill_id]       # which authoring/porting skills produced this (§7)
  derived_from: impl_card_id   # e.g. a HIP card ported from a CUDA card
  created: timestamp
  source_path: string
```

### 2.3 Why the split earns its place
- **Correctness can't drift across backends:** one reference, one tolerance set, applied to every implementation. A HIP card and a CUDA card are right *in the same sense*.
- **Reject-before-compile still works**, now in two stages: filter Op Specs by `constraints`/`preconditions`, then filter Implementation Cards by `arch`.
- **Performance honesty:** `perf.regime` and `perf.measured` are per-arch by construction, so the library never implies NVIDIA tunings transfer to AMD.
- **Provenance.derived_from** captures porting lineage (CUDA → HIP), so a fix to the source card can flag its ports for revalidation.

### 2.4 Invariants the library enforces
- An Op Spec with unsatisfiable/non-decidable `constraints` is rejected at ingest.
- An Implementation Card cannot be published until it passes its Op Spec's `shape_sweep` against the shared reference on its target arch.
- For an op with ≥2 backends, the backends must also agree with *each other* within `cross_backend_rtol` (§5.3).
- `perf.measured` entries must cite a reproducible `source` run id and an `arch`; un-sourced or arch-less numbers are dropped.

---

## 3. Retrieval / index layer

Retrieval is **structured query over contract fields**, not vector similarity over source text. (Embedding `.cu` files fails on exactly the things that matter: a transposed layout, a divisibility rule.)

### 3.1 Query interface
Retrieval is two-stage to mirror the Op Spec / Implementation Card split.
```
find_impl(
  canonical_op,            # required
  input_specs,             # dtypes, layouts, shape bindings (concrete or symbolic ranges)
  target_arch,             # e.g. amd_cdna3 / nvidia_sm90  (REQUIRED — drives backend selection)
  available_features,      # tensor_cores | mfma | tma | ...
  required_fusions = [],
  objective = "throughput" # throughput | latency | memory
) -> ranked [ {impl_card_id, op_id, backend, arch, applicable, reject_reasons:[...],
               score, matched_measurement?} ]
```

### 3.2 Ranking
1. **Op stage:** filter Op Specs by `constraints`/`preconditions` that are statically checkable.
2. **Impl stage:** among matching ops, keep Implementation Cards whose `arch.family` + `arch.requires` fit `target_arch`/`available_features`.
3. Prefer cards with a `perf.measured` entry matching the concrete `(arch, shape, dtype)`.
4. Fall back to per-arch `perf.regime` + `roofline` alignment with the objective.
5. Break ties by provenance trust and recency.

If the op exists but **no implementation matches the target backend** (e.g. a CUDA card exists, AMD does not), retrieval returns the op with an explicit `missing_backend` signal — which is the trigger for a porting skill (§7.2), not a dead end.

Crucially, **every result carries `reject_reasons`** for non-matches — the agent (and humans debugging it) learn *why* a card was excluded, which is itself training signal for better skills.

---

## 4. Two tiers: primitives vs. kernels

Separate **composable primitives** from **complete kernels**. Agents specialize and compose primitives far more reliably than they emit whole kernels.

### 4.1 Primitives (building blocks)
Tiling iterators, scratchpad-staging pipelines, swizzle helpers, lane/wave-level reductions, epilogue hooks, accumulator policies. Each primitive has a mini-contract (inputs, layout assumptions, arch requirements, composition points) — same schema family as Implementation Cards, smaller.

Primitives expose a **common interface, backend-specific implementation.** A `wave_reduce` or `mma_tile` primitive exists on both backends with the same composition contract, but the bodies differ — and the differences are exactly the things that bite if left implicit:

| Concept | NVIDIA (CUDA) | AMD (HIP/ROCm) |
|---|---|---|
| Execution group | warp = **32** lanes | wavefront = **64** lanes |
| On-chip scratch | shared memory (smem) | LDS |
| Matrix engine | tensor cores (wmma/wgmma) | matrix cores (MFMA) |
| Bulk async copy | `cp.async` / TMA | global→LDS DMA path |
| Tuning unit | `num_warps`, stages | `waves_per_eu`, stages |

The **wave-size difference (32 vs 64)** is not cosmetic — it changes tiling math, reduction tree depth, and occupancy arithmetic. Primitives must own this so kernels above them don't hardcode 32.

### 4.2 Kernels
Complete, launchable Implementation Cards assembled from primitives. A card's `uses_primitives` list gives:
- **Traceability:** a bug in a swizzle helper flags every card (on every backend) that uses it.
- **Composition:** `composes_with` (at the op level) + explicit epilogue hooks let an agent attach a `gelu` epilogue to a GEMM without rewriting the main loop.
- **Skill targeting:** authoring and porting skills (§7) operate mostly at the primitive-composition level, which is where backend differences are localized.

---

## 5. Verification harness

Verification **ships with the library** and is **deterministic**. Correctness is something the agent *checks*, never something it re-establishes by intuition.

### 5.1 Defined once on the Op Spec (shared across backends)
- A **reference implementation** that is *backend-neutral* (NumPy/PyTorch on CPU or either GPU) — never written in CUDA or HIP, so it can't favor a backend.
- **Numerical tolerances** (`rtol`, `atol`, accumulation notes) and `cross_backend_rtol`.
  Correctness uses the **standard combined per-element criterion**
  `|a − e| ≤ atol + rtol·|e|` (as NumPy/PyTorch/pytest do): an element passes
  if it is within the *absolute* OR the *relative* slack, whichever is looser.
  This matters for bf16/fp32 mixed precision — a single bf16 ULP at moderate
  magnitude (e.g. 1 ULP ≈ 0.016 at |out|=2) exceeds any sane `atol` but is
  fine relative to `rtol`; the combined form passes it, the naive
  `abs≤atol AND rel≤rtol` form would false-fail it.
- A **shape sweep manifest**: shapes/dtypes that must pass before publish, including edge cases (non-power-of-2, tiny M, huge K, boundary divisibility, and shapes that stress the 32- vs 64-lane boundary).

### 5.2 Harness interface (structured I/O)
The harness takes `arch` so the same call validates any backend.
```
verify(impl_card_id, arch, knobs, shapes, seed) -> {
  compiled: bool,
  correctness: { passed, max_abs_err, max_rel_err, failing_shapes: [...] },   # vs Op Spec reference
  determinism_check: bool,           # same seed → same bits within tolerance
  perf: { tflops, achieved_bw_pct, occupancy, achieved_waves, stall_reasons: {...} },
  artifacts: { asm_path, profile_path, run_id }   # ptx (nvidia) | gcn isa (amd)
}
```
The output is a parseable blob — no raw profiler text (Nsight or rocprof) the model has to squint at. `stall_reasons`/`occupancy` are normalized to a common vocabulary across backends so the *diagnosis* skills (§7) branch on the same fields regardless of vendor.

### 5.3 Cross-backend parity (the portability gate)
For any op with ≥2 implementations, beyond each card matching the shared reference, the harness runs a **parity check**: do the backends agree with *each other* within `cross_backend_rtol`? This catches a whole class of bugs the single reference can miss — e.g. one backend silently accumulating in fp16. A new or changed Implementation Card cannot publish if it breaks parity.
```
verify_parity(op_id, archs, shapes, seed) -> { agree: bool, max_pairwise_rel_err, diverging: [...] }
```

### 5.4 Determinism rules
- Fixed seeds, pinned input generators, pinned clock policy where possible (per backend).
- Benchmarks report median + IQR over N runs; a card with high variance is flagged, not silently averaged.
- No hidden global state on the benchmark path.
- Determinism is enforced *per backend* — AMD and NVIDIA each have their own bitwise-stability baseline; cross-backend agreement uses the looser `cross_backend_rtol`, never bit-equality.

---

## 6. The agent execution loop + compounding

### 6.1 Per-task loop (thin, tool-driven)
```
1. SPEC      normalize the requested op + shapes + dtypes + target sm
2. RETRIEVE  find_kernels(...) -> ranked candidates
3. SELECT    pick top candidate; if a matching perf.measured exists, skip to VERIFY
4. SPECIALIZE choose knobs (from declared space) or run autotune sweep
5. VERIFY    harness: correctness first, then perf
6. DIAGNOSE  if slow/incorrect, branch into an authoring/diagnosis skill (§7)
7. RECORD    write back measurement / new card / skill outcome
```
The agent does not freestyle optimization; the card's `specialization_knobs` define the space and the skills define the *procedure*.

### 6.2 Write-back (this is the whole point of a library)
- A successful `(op, shape, dtype, sm) → knobs → tflops` tuple is appended to `perf.measured`. Next agent skips autotuning.
- A genuinely novel correct+fast kernel becomes a **new staged card** (auto-promoted after it passes the shape sweep and review).
- Every loop emits a **skill outcome record** (§7.3) regardless of success.

The marginal cost of each task should trend down over time. If it doesn't, the compounding mechanism is broken and that's the top-priority bug.

---

## 7. Kernel-authoring skills + how the skills library evolves

A **skill** is reusable procedural knowledge: a playbook the agent follows to *write*, *specialize*, or *fix* a kernel. Cards are nouns (what exists); skills are verbs (how to make/improve them). Skills are versioned, scored, and themselves subject to a compounding loop.

### 7.1 Skill format — adopt the open SKILL.md standard, don't invent one

A bespoke YAML schema would only be executable by our own runtime. Instead, skills are authored as **SKILL.md** — the open Agent Skills format (a folder with a `SKILL.md`: YAML frontmatter + markdown instructions, optionally bundling scripts/templates). It's already consumed by Claude Code, OpenAI Codex, Gemini CLI, GitHub Copilot, Cursor, and Cline, so a skill authored once runs on any skills-compatible agent. We keep our library-specific metadata in a namespaced frontmatter block so standard consumers ignore what they don't understand and our runtime reads the rest.

```markdown
---
name: tune-for-cdna
description: >                 # the standard trigger field every agent reads
  Make a functionally-correct HIP GEMM/attention kernel fast on AMD CDNA:
  re-tile for 64-wide wavefronts, map to MFMA, tune waves_per_eu, restage via LDS.
  Use when an AMD HIP card passes correctness but misses its perf regime.
license: Apache-2.0
# --- namespaced extension; non-standard consumers ignore this block ---
x-kernel-lib:
  id: tune-for-cdna@1.2.0
  backend_scope: [hip]
  when_to_use:
    triggers: ["hip card correct but slow", "perf < amd roofline regime"]
    preconditions: ["arch.family in [amd_cdna2, amd_cdna3]"]
  inputs_required: ["impl_card_id", "target arch", "failing perf regime"]
  tools: [find_impl, verify, autotune, record_measurement]   # MCP tools it calls (§8)
  validation: { must_pass: ["correctness sweep", "parity", "perf >= amd roofline baseline"] }
  references: [impl_card_id, primitive_id]
  metrics: { uses, success_rate, median_iterations, regression_count }  # maintained by §7.3
  provenance: { authored_by, created, supersedes: [skill_id] }
---

## Procedure
1. Pull the card and its failing profile via `verify(...)`; read normalized `stall_reasons`.
2. Re-tile for 64-lane wavefronts (do **not** assume warp=32). ...
3. Map the inner product to MFMA matrix-core intrinsics where dtype/shape allow. ...
4. Sweep `waves_per_eu` and LDS staging depth via `autotune(...)`.
5. Re-run the shape sweep + `verify_parity`; `record_measurement(arch=amd_cdna3, ...)`.

## Pitfalls
- Hipified code left at warp=32 tiling — silently halves occupancy. ...
```

The **markdown body is the universal layer**: any LLM agent can read and follow the prose procedure even if it can't parse our `x-kernel-lib` block or call our tools. The frontmatter `description` is what every agent uses for trigger selection. The `x-kernel-lib.tools` field binds steps to the MCP tools in §8, so a fully-integrated agent executes the same procedure programmatically. One artifact, three levels of fidelity depending on how much of the stack the consuming agent supports.

Every skill declares a `backend_scope` — `agnostic` (operates at the Op Spec / contract level), or a specific list like `[cuda, hip]`. This lets the agent pick a skill that actually applies to the target backend, and lets the evolution loop (§7.3) score, say, a tiling skill *separately* on NVIDIA vs AMD, since a procedure can be reliable on one and weak on the other.

### 7.2 An initial skill set (seed the library with these)
Backend-agnostic / both-backend authoring:
- **`tile-a-gemm`** *(scope: cuda, hip)* — build a tiled GEMM from primitives for a given dtype/arch; the workhorse. Wave-size aware (32 vs 64).
- **`add-epilogue-fusion`** *(agnostic procedure, backend-specific hooks)* — attach bias/activation/norm epilogues via hooks without touching the main loop.
- **`mixed-precision-convert`** — take an fp32 kernel to bf16/fp16 with fp32 accumulation; re-validate tolerances *and* cross-backend parity.
- **`autotune-knob-sweep`** — search the declared knob space for the target arch; record the winner to `perf.measured` tagged with `arch`.
- **`diagnose-low-occupancy`** — branch on normalized `stall_reasons`/`occupancy` to a concrete fix (register/VGPR pressure, scratch usage, block/wave size).
- **`diagnose-memory-bound`** — improve coalescing / async-copy (NVIDIA) or LDS-staging (AMD) / vectorized loads when roofline says memory-bound.
- **`fuse-elementwise-chain`** — collapse a chain of elementwise ops into one kernel to kill launch + bandwidth overhead.

Cross-platform-specific (these are what "support AMD" actually requires):
- **`port-cuda-to-hip`** *(scope: cuda→hip)* — produce a HIP Implementation Card from a CUDA one. Step 1 may use HIPIFY for a *correctness-only* starting point; the skill explicitly treats hipified output as a draft, not a deliverable. Sets `provenance.derived_from`.
- **`tune-for-cdna`** *(scope: hip)* — take a functionally-correct HIP card and make it fast on CDNA: re-tile for 64-wide wavefronts, map to MFMA, tune `waves_per_eu`, restage through LDS. This is the step that turns "it runs on AMD" into "it's good on AMD."
- **`map-to-matrix-cores`** *(scope: hip)* — replace generic FMA inner loops with MFMA matrix-core ops where dtype/shape allow (the AMD analog of tensor-core targeting).
- **`port-across-arch`** *(scope: cuda)* — adapt within a vendor (sm_80 → sm_90: TMA, clusters, wgmma), re-validating contracts.
- **`establish-parity`** — given two backend implementations of one op, run `verify_parity`, and if they diverge, localize which backend's numerics are off and route to the right fix.

Each skill is **narrow on purpose**. Broad skills mis-fire; tight `when_to_use` triggers keep selection reliable (the same lesson as tight Op Spec `constraints`). The porting and tuning skills are split deliberately: **functional port** (`port-cuda-to-hip`) and **performance tuning** (`tune-for-cdna`) are different procedures with different success criteria, and conflating them is the classic way "AMD support" ships as "AMD-compatible but 4x slow."

### 7.3 How the skills library evolves (the governance loop)

This is the part that makes the skills library an asset rather than a static wiki. Every agent run emits a **skill outcome record**:
```
{ skill_id, version, task_signature, result: success|partial|fail,
  iterations, final_tflops_vs_regime, failure_mode?, run_id }
```
These feed a continuous loop:

1. **Score.** Roll outcome records into each skill's `metrics` (success_rate, median_iterations, regression_count). A skill that needs many iterations is "expensive" even if it eventually works.
2. **Promote.** When an agent solves a task *without* an existing skill — via novel steps that succeeded — those steps are mined into a **candidate skill** (staged, `authored_by: agent`). If it later reproduces success on ≥N independent tasks, it's promoted to published.
3. **Revise.** When a skill's success_rate drops or a recurring `failure_mode` appears, open a revision: tighten `when_to_use`, add a `pitfall`, or fix a `procedure` step. Bump the version.
4. **Split / merge.** A skill that fires on too many unrelated triggers gets split (mis-firing). Near-duplicate skills get merged with a `supersedes` link.
5. **Deprecate.** A skill consistently dominated by another (same triggers, worse metrics) is deprecated; `supersedes` preserves lineage so we never lose the reasoning.
6. **Guard against regressions.** Skill changes run against a **frozen replay set** of past tasks before publish — a new skill version may not regress tasks the old one solved. (Same discipline as the kernel shape sweep, applied to procedures.)

### 7.4 Skills ↔ cards ↔ measurements: the three compounding loops
- **Cards** accumulate `perf.measured` tunings → retrieval gets faster and autotuning gets skipped.
- **Skills** accumulate outcome records → authoring gets more reliable and cheaper per task.
- **Provenance** links them (`card.provenance.skill_used` ↔ `skill.metrics`) so you can ask "which skills produce the highest-quality cards?" and invest there.

All three loops share one rule: **a frozen replay/sweep set gates every change**, so the library can only get better, never silently worse.

---

## 8. Interoperability: any coding agent as a consumer

The library must not be usable only by *our* agent. The design rule: **separate the portable, declarative artifacts from the runtime, and expose each layer through an open standard** so a coding agent gets as much value as its capabilities allow — and our own agent is just the most fully-integrated consumer, not a privileged one. Three standards, already widely adopted as of 2026, cover the whole surface:

### 8.1 Capabilities → MCP (the most important interop move)
Retrieval, verification, and write-back are exposed as **Model Context Protocol** tools/resources. MCP is supported across essentially every coding agent (Claude Code, Cursor, Codex, Cline, Continue, Gemini CLI…), so any of them can call:
- `find_impl(...)`, `get_op_spec(...)`, `get_impl_card(...)` — retrieval (§3), cards/specs also served as MCP **resources**.
- `verify(impl_card_id, arch, ...)`, `verify_parity(op_id, archs, ...)` — the correctness + parity harness (§5).
- `autotune(...)`, `record_measurement(...)` — specialization + compounding write-back (§6).

This is the single highest-leverage interop decision: **even an agent that ignores our skills and writes a kernel from scratch can still call `verify`/`verify_parity`** and inherit the correctness-and-parity guarantee. The verification surface is valuable on its own, independent of the rest of the library.

### 8.2 Procedures → SKILL.md
Authoring/porting skills (§7) are packaged in the open **SKILL.md** format, consumable by any skills-compatible agent. Prose body = universal; `description` = trigger selection; the namespaced `x-kernel-lib.tools` binding = programmatic execution for agents wired to our MCP server.

### 8.3 Discovery → AGENTS.md
A coding agent dropped into the repo needs to *know the library exists* and how to use it. A top-level **AGENTS.md** (the vendor-neutral, Linux-Foundation-stewarded standard read by 28+ tools, including Claude Code via CLAUDE.md) states: how to reach the MCP server, that any new/edited kernel **must pass `verify` + `verify_parity` before it's considered done**, and where the skills live. Per recent findings that bloated, auto-generated context files *hurt* agent success, this file is kept deliberately minimal and precise — a pointer and a hard rule, not a manual.

### 8.4 Tiers of consumption (graceful degradation)
| Consumer capability | What it gets |
|---|---|
| Reads files only | JSON Op Specs / Impl Cards (published JSON Schema) + SKILL.md prose — can select and copy a kernel |
| Speaks MCP | Live retrieval + the full verify/parity/autotune loop — correctness guaranteed even for self-written kernels |
| Skills-compatible | Executes authoring/porting procedures directly |
| Fully integrated (our agent + others wired to the MCP server) | All of the above plus write-back into the compounding loops (§6, §7.3) |

The artifacts themselves stay in **plain, schema'd, vendor-neutral formats** (JSON + JSON Schema for specs/cards, Markdown for skills) precisely so the bottom tier is never empty. Open standards are doing the portability work; we're not asking any agent to adopt a proprietary client.

---

## 9. First milestone (prove the substrate before breadth)

Pick **~10 high-traffic ops** (GEMM variants, attention, rmsnorm, a couple of fused epilogues, a reduction). For each:
1. Write an airtight **Op Spec** (full `constraints` + `preconditions` + `numerics` + `cross_backend_rtol`).
2. Write the backend-neutral reference impl + shape-sweep manifest + tolerances.
3. Ship **two Implementation Cards** — one CUDA (NVIDIA), one HIP (AMD/CDNA).
4. Wire the deterministic harness on both backends + the `verify_parity` gate.

Then the acceptance test, stated cross-platform: **an agent can select-and-specialize these correctly across a shape sweep, on at least one NVIDIA and one AMD target, with backends passing parity, without generating any new kernel source** — measuring (a) retrieval precision per backend, (b) correctness pass rate, (c) cross-backend parity pass rate, (d) median iterations to a tuned result, (e) % of tasks served from `perf.measured` cache after warm-up, **per arch**.

A second, portability-specific milestone: **starting from CUDA-only cards, an agent uses `port-cuda-to-hip` + `tune-for-cdna` to stand up the AMD implementations**, and we measure the AMD result both for correctness (must pass) and for performance relative to the AMD roofline (the honest bar — not relative to the NVIDIA card). This validates that the *porting* loop, not just the static library, actually works.

Only once that holds do we invest in breadth and the full skill-promotion machinery. Capability you can buy later from a better model; the contract + verification + cross-backend-parity + compounding layer is the part that's actually yours — and it's what makes "support AMD" a property of the system rather than a one-time porting heroics project.

---

## 10. Anti-goals / things to deliberately resist
- **Human-prose docs as the primary interface.** Ambiguous to machines; encode contracts instead.
- **One mega-kernel with a thousand flags.** Validity surface explodes; the agent can't reason about applicability.
- **Vector search over raw source as the retrieval story.** Misses the constraints that actually decide applicability.
- **Non-determinism anywhere on the feedback path.** It poisons every compounding loop.
- **"The model will infer the layout."** It won't, reliably. Encode it.
- **Skills with loose triggers.** They mis-fire and erode trust in selection. Keep them narrow.
- **Un-sourced perf numbers.** Every measurement cites a reproducible run (and an `arch`) or it's dropped.
- **Conflating functional portability with performance portability.** "It runs on AMD" is not "it's good on AMD." Keep the functional port and the perf-tuning skills separate, and never report AMD perf relative to the NVIDIA card — report it against the AMD roofline.
- **A CUDA-shaped reference implementation.** The shared reference must be backend-neutral, or correctness silently tilts toward whichever backend it resembles.
- **Hardcoding warp=32.** Wave size is 32 on NVIDIA and 64 on AMD; bake it into primitives, never into kernels.
- **One lowest-common-denominator source for all backends.** Tempting for maintenance, but it usually leaves both backends slow. Portability belongs in the contract; let implementations be native. (A portable-DSL build is allowed as *one* implementation among others — never as the only one.)
- **A proprietary client as the only way in.** If consuming the library requires our SDK, we've lost. Lead with open standards (MCP / SKILL.md / AGENTS.md) and plain schema'd files; our agent is the most-integrated consumer, not a gatekeeper.
- **A bespoke skill DSL.** Already rejected (§7.1) — it strands the skills on our runtime. Author on SKILL.md.

---

## 11. Open questions
- Canonical op vocabulary: how granular? (e.g., is "attention" one op or a family with variants as separate cards?)
- Where does generation live — inside skills, or as a separate fallback service the skills call?
- Trust model for agent-authored cards/skills: how many independent confirmations before auto-publish?
- Cross-hardware regime modeling: heuristic `perf.regime` strings vs. a learned cost model fed by `perf.measured`.
- How aggressively to garbage-collect dominated cards vs. keep them for lineage.
- **Portability production path:** native per-backend cards only, or also a portable DSL (Triton, or CUTLASS + AMD Composable Kernel) as a fast way to seed both backends from one source? Current lean: allow DSL builds as *one implementation among others*, gated by the same verification, never as the sole backend.
- **Backend roadmap:** after CUDA + HIP, which next — Triton as a portability backend, SYCL, Metal? The schema is designed to admit them, but each adds a primitive set + harness wiring + arch vocabulary.
- **Setting `cross_backend_rtol`:** how loose is honest? fp16/bf16 accumulation order differs across vendors; too tight and parity false-fails, too loose and it stops catching real numeric bugs.
- **AMD perf baselines:** what's the credible roofline/reference to grade AMD cards against (rocBLAS/hipBLASLt, Composable Kernel) so "fast on AMD" has an objective bar?
- **Standards drift:** SKILL.md / MCP / AGENTS.md are young and evolving. How do we pin versions and absorb breaking changes without re-authoring the corpus? (Lean: keep our content in the namespaced/extension fields, treat the standard's required fields as a thin stable core.)
- **Untrusted external consumers:** when an outside agent calls `record_measurement`, how do we trust the write-back? (Likely: external calls are read + verify only; write-back to the shared corpus requires a reproducible, server-side-rerun `source` run.)
```
