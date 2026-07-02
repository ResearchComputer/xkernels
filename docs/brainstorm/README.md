# Brainstorm: a higher-level, cross-platform, performance-first kernel-authoring DSL

**Status:** exploratory RFC (draft v0.2) — 2026-07-01 — **Phase 0/1/1.5/2.0a/2.0b/2.2a/2.2b/2.3/2.1-GPU/3 SHIPPED** (decidable gate; cost model + 70% roofline gate; override mechanism with enforced oracle property + LIVE native CUDA codegen on GB10; CUDA/HIP graph capture at 5–6.5× over sequential on launch-bound chains; conditional-node boundary honestly documented). Phase 4 (compounding loop / MCP surface) is the remaining track.
(the `vkl/` package compiles, `@kernel` lowers to a generated Triton kernel
that `verify` passes on H100, beating the hand baseline — see
[`11-implementation-plan.md`](11-implementation-plan.md) §8/§9/§10).
**Scope:** a design exploration, *not* a decision to build. But the *stance* is
now decided on three points that v0.1 left open: see "The decided stance" below.
Where this doc set still has open questions, they are open *within* that stance,
not about it.

## The one-paragraph thesis

`xkernels` made **portability live in the contract** (an Op Spec + backend Impl
Cards + a deterministic harness). But authoring a kernel today means handwriting
four disjoint things — a Triton kernel, a CUDA/HIP C++ extension, a CUTE-DSL
kernel, *and* the JSON Op Spec / reference / sweep / input-gen — by hand, in
separate files, with the portability vocabulary (wave 32 vs 64, smem vs LDS,
tensor cores vs MFMA) re-encoded implicitly inside each one, and with no way to
reach a native performance ceiling from one source. This brainstorm proposes a
**contract-native authoring DSL** (`vkl`, the **Vibe Kernel Language**) that sits *above* the
existing substrate: one source that (a) emits the Op Spec mechanically so it
cannot drift, (b) auto-derives the backend-neutral reference by running the same
compute on CPU, (c) **lowers to multiple native backends from day 1** with
per-target override bodies that reach the vendor ceiling (not a portable
lowest-common-denominator), and (d) **captures multi-kernel compositions into
CUDA/HIP graphs from day 1**, collapsing launch overhead on the fused/short
chains that dominate real workloads. The contract stays the product; the DSL is
a far better *producer* of cards than typing them by hand.

## The decided stance (what changed from v0.1)

v0.1 leaned "Triton-first, defer native + graphs." The directive for v0.2
reverses three of those leans into **hard requirements from day 1**:

| v0.1 lean | v0.2 decision | Where it lands |
|---|---|---|
| Triton-only Phase 1; multi-target is a go/no-go later | **Multi-target from day 1** — Triton + CUDA + HIP all lower, all first-class | `02`, `03`, `06` |
| Native ceiling is a skill's job, not the DSL's ("starting cards only") | **Performance-optimized** — the DSL reaches the vendor ceiling via per-target override bodies; a correct-but-slow DSL card is a failure | `02`, `04`, `06` |
| Graphs unmentioned | **CUDA/HIP graph capture from day 1** — compositions lower to instantiated graphs, with parameter + conditional nodes | `07` (new) |

This does **not** touch the hard rules (§10): the contract is still the product,
the reference is still backend-neutral, `warp` is never hardcoded to 32, and the
DSL remains *one* producer among others — never a gatekeeper. "Performance
portability is not free" (§10) is *honored*, not repealed: the DSL pays for perf
by giving each target its own override body, not by pretending one source is
fast everywhere.

## Working name

**`vkl`** — the **Vibe Kernel Language**: a contract-native, agent-editable kernel
authoring DSL. (The earlier placeholder name `xtl` / "xkernels tile language" was
retired; the package, modules, tests, docs, and committed-card source paths all
renamed `xtl` → `vkl` on 2026-06-30. The name "vibe" signals the design target:
a kernel language productive enough to author by feel — human or agent — while
the frozen math IR + decidable schedule gate keep it honest.)

## Doc map

| Doc | What it argues |
|---|---|
| [`01-why-now.md`](01-why-now.md) | The authoring pain today (incl. launch overhead on fused chains, unreachable native ceilings), grounded in real kernels in this repo; success criteria + non-goals. |
| [`02-core-idea.md`](02-core-idea.md) | The thesis in **four layers** (contract / compute / targets-with-overrides / orchestration), the per-target-override model that buys perf, and how it respects §10. |
| [`03-design-space.md`](03-design-space.md) | The axes. Lowering-target axis is now **decided (multi-target)**; new axes for graph capture model and override granularity. |
| [`04-strawman.md`](04-strawman.md) | Worked examples: `dual_rmsnorm`, a **multi-target tiled GEMM with a per-target sm_90 override**, and a **2-kernel graph** showing graph capture + parameter nodes. |
| [`05-substrate-and-loops.md`](05-substrate-and-loops.md) | How the DSL feeds the compounding loops; provenance; what stays **unchanged**. |
| [`06-open-questions-roadmap.md`](06-open-questions-roadmap.md) | The *remaining* open questions (graphs vs capture, conditional coverage, graph-card schema), the §10 landmines as a checklist, and a multi-target-from-day-1 roadmap. |
| [`07-graphs.md`](07-graphs.md) | **(new)** CUDA/HIP graph capture as a first-class capability: the `@graph` model, parameter/conditional nodes, graph × autotune, graph × fusion, substrate fit, perf honesty. |
| [`08-programming-model.md`](08-programming-model.md) | **(new)** The tiling/SIMD programming model: a 6-level hardware hierarchy (device→cluster→CTA→wave→lane→matrix-engine), block-level-by-default / explicit-hierarchy-override, and the auto/checked/rejected contract that makes the §10 anti-goals compiler-enforced. |
| [`09-agent-editable-ir.md`](09-agent-editable-ir.md) | **(new)** The IR designed for an *LLM editor*, not a compiler pass: frozen math layer + editable schedule layer over the L0–L5 hierarchy; named edit primitives (`set_knob`, `retile`, `map_to`, …) as the skills' MCP tools; a closed-form cost model so the agent predicts before it measures; a check gate; and the `tuning_trace` compounding loop. This is what makes autonomous perf-pushing plausible. |
| [`10-ir-data-structures.md`](10-ir-data-structures.md) | **(new)** The IR made concrete: math-IR node schemas (frozen oracle), schedule-IR node schemas (editable), the edit/diff/trace format, cost-model formulas grounded in the existing `registry/cost_model.py` + `archs.py` (per-instr peaks, roofline aggregate, occupancy), the check gate, and the lowering dispatch — all mapped to the fields `ImplCard.from_doc` already ingests. |
| [`11-implementation-plan.md`](11-implementation-plan.md) | **(new)** The engineering plan: package layout (`src/xkernels/vkl/`), the no-touch rule for existing surfaces, module breakdown, a phased effort-tagged roadmap (Phase 0 probe → 1 Triton → 2 multi-target → 3 graphs → 4 compounding) with hard gates and pre-committed fallbacks, the testing strategy, and the concrete first commit. |

## How to read this if you're short on time

Read `01` (the problem), then `08` (the programming model — tiling/SIMD is the
hardest part), then `09` (the agent-editable IR — this is what makes autonomous
perf-pushing the point of the whole effort), then `04` (the strawman incl. the
graph example). That quartet is enough to tell you whether the idea has legs.
The rest is the argument and the plan.

If you're going to *build* it, skip to `10` (concrete data structures) and
`11` (implementation plan) — and start with `11` §7's first commit.

## The decision this brainstorm does NOT make

It does **not** decide to build the DSL. It argues the idea is continuous with
the existing design, sketches a credible shape that is multi-target +
graph-capable + perf-targeting from day 1, with an IR an agent can compile toward
performance, and names the remaining risks. The "go/no-go" is gated on the
kill-experiment in `06` §D: hand-translate a **multi-target GEMM with a
per-target override + a 2-kernel graph**, manually emit all artifacts, and
confirm `verify` + `verify_parity` + a roofline check all pass with zero
hand-editing of JSON — *and* (the `09` addition) sketch the edit sequence an
agent would issue to push that GEMM from its portable starting perf to the sm_90
ceiling, confirming each edit is locally checkable.
