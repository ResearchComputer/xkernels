# 03 — The design space (axes, options, trade-offs)

v0.2 has *decided* the biggest axis (multi-target day 1). This doc now argues the
remaining axes *within* that decision: how the multi-target lowering reaches
ceilings, how graphs are captured, and the usual surface/syntax questions. For
each axis: the real options and what each costs.

## Axis A — Surface syntax

| Option | What it is | Pro | Con |
|---|---|---|---|
| **A1. Embedded Python (decorator + restricted AST)** | `@kernel(...)` over a Python function; the DSL is a subset of Python the toolchain parses/lowers. | Inherits the repo's Python toolchain; trivially `register()`-able; agent-friendly (parseable AST); matches Triton & CUTE which are *already* Python-embedded. | "Restricted subset" needs a clear boundary, or authors reach for disallowed constructs and get confusing errors. |
| **A2. A standalone declarative language** (own grammar/parser) | `.vkl` files, MLIR-ish or custom. | Total freedom; can enforce purity. | New toolchain to build, debug, document, teach agents. Violates the §8 "open standards, don't strand on a proprietary client" spirit unless it lowers to MLIR (a lot of work for MLIR's benefit, which we could get by targeting Triton/CUTLASS instead). |
| **A3. Pure-declarative IR (no human-facing syntax)** | Authors write contract + tile-plan in JSON/YAML; lowering is mechanical. | Maximally machine-legible; trivial drift control. | Barely a "language programmers can write kernels in" — this is back to authoring JSON, which is the problem. |

**Decision:** **A1.** The repo is Python-first. An embedded DSL is the
lowest-friction surface and the most agent-legible. The boundary problem is real
but tractable (strict allow-list, like Triton's own). **Open within A1:**
Python-AST → internal IR → multi-target, vs Python-AST → Triton-AST directly.
Multi-target day 1 *forces* the former (you need an IR to fan out to CUDA + HIP +
Triton + graphs). This is non-negotiable now and is the bulk of the build cost.

## Axis B — Imperative vs. declarative split

How much of a kernel does the author *write* vs. *declare*?

- **B1. Mostly imperative** (Triton-like): author writes the tile loop, loads,
  `mma`, store. DSL wraps with the contract header and named portability
  primitives.
- **B2. Mostly declarative** (schedule-separation): author describes the math
  ("C = A @ B; epilogue = bias+gelu"), a separate *schedule* describes tiling.
  CUTLASS/CK philosophy. Most expressive for GEMM-like families; awkward for
  irregular kernels (MoE align, sparse-MLA indexer).
- **B3. Hybrid**: imperative compute, declarative *composition hooks* (epilogue
  fusion, residual adds) attached by name. Matches the existing fusion skills.

**Lean:** **B3** for v1, expanding toward B2 *per family* where it pays
(essentially: GEMM/attention get a declarative skeleton; everything else stays
imperative). The repo's op mix is too varied for one style. **Open:** whether the
declarative skeleton is worth building in v1 or is added per-family later. (`06`
recommends imperative-first, declarative skeletons as a Phase-3 expansion.)

## Axis C — Lowering targets *(DECIDED: multi-target day 1)*

> **v0.1 left this open and leaned "Triton-first." v0.2 decides: Triton + CUDA +
> HIP all lower from day 1, each a first-class Impl Card.** The remaining question
> is no longer *whether*, but *how each reaches its ceiling* — which is Axis H.

| Target | Role in v0.2 | Lowering path |
|---|---|---|
| **Triton** | Portable baseline + reference path (`arch: any`). Always emitted. | Python-AST → Triton-AST → `@triton.jit`. |
| **CUDA** | Native NVIDIA ceiling (sm_80/sm_90: tensor cores, TMA, clusters, wgmma). | Python-AST → CUTLASS/CUTE templates, or emitted `.cu` + `cppimport`. |
| **HIP** | Native AMD ceiling (CDNA2/3: MFMA, LDS-DMA). | Python-AST → Composable Kernel, or emitted `.hip` via the existing hipify bridge. |

**The build cost is real and front-loaded:** multi-target day 1 means the
internal IR (Axis A's open question) is a Phase-1 deliverable, not a later
refactor. This is the price of the directive; `06` accepts it and time-boxes it.

## Axis D — Granularity: primitives vs. whole kernels

- **D1. Kernel-level only** (v1): the DSL describes whole kernels; primitives are
  library-provided building blocks the author calls.
- **D2. Primitive-authoring too**: define new portability primitives (a novel
  swizzle) with per-backend bodies, reuse across kernels. Makes §4.1's
  "common interface, backend-specific implementation" a first-class authoring unit.

**Lean:** **D1 for v1**, design the surface so D2 is a *possible* later extension
(primitives are just kernels with a composition contract). Don't build the
primitive-authoring machinery until a real op needs a primitive the library lacks.

## Axis E — Fusion / epilogue composition

How does the DSL express fusion?

- **E1. Inline fusion** (B1-ish): write the epilogue in the same compute body.
  Simple; loses reusability.
- **E2. Named composition hooks** (B3): a kernel declares an epilogue hook and a
  separate `@epilogue(gelu, bias)` attaches. Reusable; matches the skills.
  **The skills' shape-change rule must be enforced statically:** fusion that
  *changes output arity/semantics* (residual+rmsnorm emitting both the norm and
  the new residual) is a **new Op Spec**, not a card variant — the DSL routes it
  to the `author-an-op-spec` flow, not a silent card extension.

**Lean:** **E2**, with the shape-change rule a static compile error. This makes a
§10-style rule machine-checkable instead of convention. Note (from `07`): fusion
and graph capture are *orthogonal* — a graph node can be a fused kernel.

## Axis F — Numerics as data, not comments

`numerics` (rtol/atol/reduce_dtype/cross_backend_rtol) is contract data. In the
DSL it should be *enforced*:

- Every reduction feeds an accumulator whose dtype matches `reduce_dtype` —
  statically checkable from the compute body.
- `cross_backend_rtol` should be *derivable* from the precision path (bf16 in /
  fp32 acc → a known ULP budget), with the DSL *suggesting* a value rather than
  demanding the author guess. (A concrete attack on §11's open question about how
  to *set* it.)

**Lean:** numerics as a *checked* first-class layer.

## Axis G — Graph capture model *(NEW; see 07)*

How does a `@graph` lower?

- **G1. Explicit construction** — emitter builds `cudaGraphAddKernelNode` +
  `AddDependencies` from declared dataflow. Cleanest, optimizable,
  parameter/conditional-node friendly. **The v1 default.**
- **G2. Stream capture** — `BeginCapture`/`EndCapture` records an existing launch
  sequence. Opaque to the runtime; harder to wire to parameter/conditional nodes.
  **The escape hatch**, opt-in via `@graph(captures="stream")`.
- **G3. Hybrid** — explicit where expressible, stream-capture for the
  data-dependent tail. Most general; most complex emitter.

**Lean:** **G1 default, G2 escape hatch, G3 not in v1.** The conditional-node
question (how much of MoE/sparse G1 can cover) is the headline open question
(`06` B3) — G3 is the answer only if G1 demonstrably can't cover the workload.

## Axis H — Per-target override granularity *(NEW; this is how perf is bought)*

The portable Layer-2 body is correct everywhere but won't reach ceilings. How
much of a kernel can a per-target override replace?

- **H1. Full-body override** — the target's override replaces the entire compute
  body (e.g. the sm_90 GEMM body uses TMA+clusters+wgmma throughout). Simplest
  contract (override is a whole new body); biggest authoring duplication; the
  override and the reference share only the *contract*, not code.
- **H2. Primitive-level override** — the portable body calls `mma(...)`,
  `stage_async(...)`, `wave_reduce(...)`; a per-target override replaces just the
  *primitive bodies*, not the kernel structure. The kernel body stays shared →
  less duplication; the reference and every target share the same structure.
- **H3. Region override** — override a marked *region* (the inner K-loop), keep
  the rest shared. Middle ground.

**Lean:** **H2 where possible, H1 where necessary.** H2 is the dream — the
portable body *is* the structure, and reaching a ceiling is "swap the primitive
implementations for native ones," which is exactly §4.1's "common interface,
backend-specific implementation." H1 is the fallback for ops whose ceiling needs
a fundamentally different structure (e.g. an online-softmax attention that
restructures the whole loop for sm_90). **Open:** how often H2 suffices. If the
answer is "most GEMM/attention," the DSL is small and elegant; if "rarely," H1
dominates and the DSL is mostly a multi-backend template generator. This is
empirically answerable in Phase 1 and shapes the whole tool.

## Cross-cutting: what the decided axes force first

With Axis C decided (multi-target), the **internal IR (Axis A's open question) is
on the critical path** — you cannot fan out Python-AST to three backends without
one. With perf day 1, **Axis H (override granularity)** is the empirical question
that most shapes the authoring experience. Both are Phase-1 deliverables in `06`.
