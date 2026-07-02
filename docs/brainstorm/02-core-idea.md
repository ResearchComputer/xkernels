# 02 — The core idea: contract-native authoring, multi-target + perf-first

## The thesis in one sentence

**Author the kernel in one place that *is* the contract; emit the Op Spec, the
reference, the sweep, and the Impl Cards from it, so they cannot drift; lower
the compute to multiple native backends from day 1, each with its own path to
the vendor ceiling; and capture multi-kernel compositions into CUDA/HIP graphs.**

Everything else is mechanism. **Four layers** carry it (v0.2 adds the fourth —
orchestration — and makes the third carry per-target overrides for performance).

## The four layers

```
   Layer 4       ORCHESTRATION   @graph compositions → CUDA/HIP graphs
                 (see 07)        (parameter nodes, conditional nodes)
                                 │ composes kernels from ↓
   Layer 3       TARGETS         @targets(triton=…, cuda=…, hip=…)
                                 + per-target override bodies → native ceiling
                                 │ instantiates ↓
   Layer 2       COMPUTE         semi-imperative over tiles
                                 mma / wave_reduce / stage_async are NAMED
                                 │ generates
   Layer 1       CONTRACT        declarative — literally the Op Spec
                                 id, inputs, outputs, constraints, numerics
```

### Layer 1 — Contract (becomes the Op Spec)

The declarative header. Its fields are **exactly** the Op Spec fields in
[`library.md`](../../meta/docs/library.md) §2.1, so the spec is *generated*, not
typed:

```python
@kernel(
    id="gemm_bf16@1.0.0",
    kernel="gemm",
    inputs={
        "a": Tensor(rank=2, dtype=["bf16","fp16"], symbols=["M","K"]),
        "b": Tensor(rank=2, dtype=["bf16","fp16"], symbols=["K","N"]),
    },
    outputs={"c": Tensor(rank=2, dtype=["bf16","fp16"], symbols=["M","N"])},
    constraints=["dtype(a)==dtype(b)", "K % 8 == 0"],
    numerics=Numerics(rtol=1.6e-2, atol=1e-2, reduce_dtype="fp32",
                      cross_backend_rtol=2e-2),
    sweep="gemm_bf16",
)
```

The constraint mini-language (comparisons, arithmetic, `and/or/not`,
`dtype(arg)`) is *the same* one the registry already evaluates, so the header is
just a different spelling of the same JSON. `vkl emit-spec` writes the JSON;
ingest re-validates it. **Round-tripping proves equivalence** — no second notion
of "what the op is."

### Layer 2 — Compute (becomes device code *and* the reference)

The genuinely new idea of the lower three layers. A *portable tile program*:
tiles, loads/stores, reductions, matrix multiplies — **not** `tl.load` or
`__ldg` or `cute::make_tensor`. Primitives take the portability vocabulary as
named arguments:

```python
acc   = mma(a, b, acc, accum=fp32)             # tensor cores on NV, MFMA on AMD
wave_reduce(x, op=sum, axis=0)                 # warp-shuffle on NV, DPP on AMD
a = stage_async(load(...), into=scratch)       # cp.async/TMA on NV, global→LDS DMA on AMD
```

`scratch` is not `smem`; it's a handle whose *kind* (smem/LDS) is bound per
target. `wave_reduce` doesn't know if a wave is 32 or 64 lanes. **The §4.1
table is the API** — exactly the "common interface, backend-specific
implementation" the design already calls for, now expressed instead of re-derived.

**The reference falls out for free.** The same compute layer, run on CPU tiles
with pure `torch`, *is* the backend-neutral oracle §5.1 demands. `reduce_dtype:
fp32` is honored because the CPU path takes the same `.to(fp32)`. **The reference
cannot drift from the kernel because they are the same code.** This closes the
most expensive drift gap in the substrate today.

### Layer 3 — Targets (becomes Impl Cards) — *with per-target overrides*

A `@targets` block declares, per lowering target, the arch requirements and the
declared tuning space — **exactly the Impl Card fields** in §2.2:

```python
@targets(
    triton=dict(arch="any",
                knobs={"BLOCK_M":[64,128,256], "BLOCK_N":[64,128,256], ...}),
    cuda  =dict(arch="nvidia_sm90", requires=["tensor_cores","tma","clusters"],
                knobs={"BLOCK_M":[128,256], ...}),
    hip   =dict(arch="amd_cdna3",   requires=["matrix_cores","mfma"],
                knobs={"BLOCK_M":[128,256], "waves_per_eu":[1,2], ...}),
)
```

Each entry emits one `registry/impls/<op>.<backend>.card.json`. **This is the
multi-target-from-day-1 requirement.** The runtime callable is registered
exactly as today (`register(kernel, Backend.X)(...)`) — the lowering produces it
instead of a human typing it.

**The per-target override (this is how perf is bought, not punted).** The Layer-2
body above is the *portable* path — correct everywhere, and the reference. But a
portable body that reaches the native ceiling is exactly the §10
lowest-common-denominator trap. So each target can declare an **override body**
that replaces the inner loop with native-ceiling mechanisms while keeping the
same contract:

```python
@gemm.target("cuda", arch="nvidia_sm90")      # overrides the portable body
def gemm_sm90(ctx, a, b, *, BLOCK_M, BLOCK_N):
    # native ceiling: TMA descriptors, thread-block clusters, wgmma
    a_desc = ctx.tma_descriptor(a, tile=(BLOCK_M, BLOCK_K))
    acc = zeros((BLOCK_M, BLOCK_N), fp32)
    for k in range(0, a.shape[1], BLOCK_K):
        acc = wgmma(a_desc[k], b_tile, acc, cluster=2)   # native instr
    store(acc.to(a.dtype), ctx.out.c[...])
```

The override is the Impl Card for the sm_90 target; the portable body is *still*
the reference and *still* ships as the `triton` (arch:`any`) card. Both pass
`verify` against the same reference. **Performance portability is paid for by
giving each target its own override, not by wishing one source were fast
everywhere** — which is the §10 anti-goal, honored by construction.

A target with no override lowers the portable body (correct, may miss the
ceiling); a target with an override lowers *that*. The `perf.measured` entry
honestly records which it is and what roofline fraction it hit. A correct CUDA
card that misses the sm_90 roofline is, per the directive, a **failure to fix**,
not a shippable starting point — that's the bar.

### Layer 4 — Orchestration (becomes graphs) — *new in v0.2*

Single kernels compose into launch sequences, and launch sequences have overhead.
Layer 4 captures a composition of Layer-3 kernels into a CUDA/HIP graph so the
whole sequence launches once. This is substantial enough to have its own doc —
see [`07-graphs.md`](07-graphs.md). The short version: a `@graph` declares
dataflow between kernel calls; the emitter instantiates a graph with parameter
nodes (for runtime-varying shapes/args) and conditional nodes (for MoE/sparse
data-dependent control). One DSL source → a CUDA graph *and* a HIP graph, both
validated against the composed reference.

## How this respects every hard rule in §10

| §10 anti-goal | How the design avoids it |
|---|---|
| *One lowest-common-denominator source for all backends* | The portable body is the **reference + fallback**, not the ceiling. Each target reaches its ceiling via a per-target override (Layer 3). A correct-but-slow card is a failure (per the directive), not a shippable default. |
| *Hardcoding warp=32* | `wave` is a named primitive/parameter; no `32` literal means lanes. `@targets(..., wave_size=…)` binds it, never the compute body. |
| *The model will infer the layout* | Layout is declared in Layer 1 (inputs/outputs) and Layer 3 (target arch), not inferred. |
| *Non-determinism on the feedback path* | Lowering is deterministic; graph instantiate/launch is on the deterministic path (`07` §9). The harness is untouched. |
| *A CUDA-shaped reference* | The reference is auto-derived from the *portable* Layer-2 body, structurally backend-neutral (§5.1). |
| *Conflating functional with performance portability* | Functional portability = the portable body passes `verify` everywhere. Performance portability = the per-target override hits the roofline. Two distinct claims, two distinct gates. |
| *A proprietary client as the only way in* | The DSL emits standard JSON specs/cards (incl. graph cards via namespaced fields). An agent reading JSON loses nothing (§8.4). |
| *A bespoke skill DSL* | That anti-goal is about *skills* (SKILL.md), not kernel source. Different layer. |

## What stays *completely unchanged*

The DSL is a **producer**; the substrate is the **product**. Nothing the agent or
harness already does is altered:

- `find_impl(...)` reads the emitted JSON — unchanged
  ([`src/xkernels/retrieval.py`](../../src/xkernels/retrieval.py)).
- `verify(card, arch)` and `verify_parity(op)` run on the emitted cards (a graph
  card is one launchable unit) — unchanged
  ([`src/xkernels/verify.py`](../../src/xkernels/verify.py)).
- `register(kernel, Backend.X)` dispatch — unchanged
  ([`src/xkernels/_dispatch.py`](../../src/xkernels/_dispatch.py)).
- The JSON Schemas in `registry/schema/` — extended with *namespaced* fields only
  (graph metadata); the standard core is untouched.

Delete the DSL tomorrow and the corpus keeps working, because the corpus is the
JSON, and the DSL only writes JSON. **This is what makes a multi-target,
graph-capable, perf-first design safe to experiment with**: the ambition lives in
the producer, not in load-bearing infrastructure.

## Relationship to existing higher-level efforts

The repo already touches Triton, CUTE DSL (`cutlass.cute`), and the CUDA↔HIP
hipify bridge; outside it are CUTLASS and AMD's Composable Kernel. The DSL is
**not a competitor** — it is a meta-layer above them:

- It lowers the portable body to **Triton** (the always-available `arch: any` card).
- It lowers per-target overrides to **CUTE/CUTLASS** (NVIDIA matrix core) or
  **Composable Kernel** (AMD matrix core) — *driving*, not replacing, those tools.
- It lowers orchestration to **CUDA/HIP graph APIs** directly.

The DSL's job is "be the one place where contract + reference + multi-target
ceiling + graph composition are declared together, emitting the substrate
artifacts the rest of the library already consumes." `04` makes this concrete;
`06` asks whether that's worth the build cost.
