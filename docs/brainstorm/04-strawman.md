# 04 — Strawman: what the DSL would actually look like

A sketch to make `01`–`03` + `07` concrete, **not** a syntax proposal. The point
is to show (a) the authoring experience, (b) what gets emitted, (c) that the
portability vocabulary is named, and (d) **multi-target lowering + per-target
override + graph capture all in one source.** Three examples, increasing
difficulty.

The syntax assumes **A1 / B3 / D1 / G1 / H2-or-H1** leans from `03`. The
workhorse targets are Triton (portable), CUDA (NVIDIA ceiling), HIP (AMD
ceiling).

---

## Example 1 — `dual_rmsnorm` (simplest real kernel; portable target only)

> **Status: CLOSED (Phase 1.5, shipped 2026-06-30).** One `@kernel` source
> ([`src/xkernels/vkl/examples/dual_rmsnorm.py`](../../src/xkernels/vkl/examples/dual_rmsnorm.py))
> builds a row-reduce IR that lowers to BOTH torch (the auto-reference, bit-exact
> with the hand `dual_rmsnorm_ref`) AND Triton (a generated `@triton.jit`).
> Registering the lowered kernel + `verify("dual_rmsnorm.triton@1.0.0")` PASSES
> on H100 with zero JSON hand-editing. The generated kernel runs at 0.099 ms vs
> ~0.18 ms for the hand baseline (see `11-implementation-plan.md` §10).

Today: eight hand-written artifacts in four paradigms
([`src/xkernels/ops/norm/triton/dual_rmsnorm_kernel.py`](../../src/xkernels/ops/norm/triton/dual_rmsnorm_kernel.py)
+ spec + reference + sweep + input-gen + card). In the DSL, one source:

```python
from vkl import kernel, targets, launch, Tensor, Numerics, Derived
from vkl.tiles import arange, load, store, rsqrt, sum, next_pow2, fp32

@kernel(
    id="dual_rmsnorm@1.0.0", kernel="dual_rmsnorm",
    inputs={
        "x1": Tensor(rank=2, dtype=["fp32","bf16"], symbols=["T","d1"]),
        "w1": Tensor(rank=1, dtype=["fp32","bf16"], symbols=["d1"]),
        "x2": Tensor(rank=2, dtype=["fp32","bf16"], symbols=["T","d2"]),
        "w2": Tensor(rank=1, dtype=["fp32","bf16"], symbols=["d2"]),
    },
    outputs={"o1": Derived("x1"), "o2": Derived("x2")},
    constraints=["dtype(x1)==dtype(w1)", "dtype(x2)==dtype(w2)"],
    numerics=Numerics(rtol=1.6e-2, atol=1e-2, reduce_dtype="fp32",
                      cross_backend_rtol=1.6e-2),
    sweep="dual_rmsnorm",
)
@targets(triton=dict(arch="any", knobs={}))      # → dual_rmsnorm.triton.card.json
@launch(grid=lambda x1: (x1.shape[0],))          # one program per token row T
def dual_rmsnorm(ctx, x1, w1, x2, w2, eps=1e-6):
    row = ctx.program_id(0)
    def rmsnorm(x, w, d, out):
        cols = arange(0, next_pow2(d)); m = cols < d
        v = load(x[row, cols], mask=m).to(fp32)        # reduce_dtype honored
        inv = rsqrt(sum(v*v) / d + eps)                 # fp32 reduction
        wv = load(w[cols], mask=m).to(fp32)
        store(v * inv * wv, out[row, cols], mask=m)
    rmsnorm(x1, w1, x1.shape[1], ctx.out.o1)
    rmsnorm(x2, w2, x2.shape[1], ctx.out.o2)
```

### What `vkl build dual_rmsnorm` emits

| Emitted artifact | How it's derived | Compared to today |
|---|---|---|
| `registry/ops/dual_rmsnorm.spec.json` | Directly from `@kernel` — fields 1:1 with §2.1. | Was hand-written; now generated + round-trip-validated by existing ingest. |
| `reference.py::dual_rmsnorm_ref` | Auto-derived: the same body on CPU `torch` tiles. | Was a separate hand-written function; now *cannot* drift. |
| `registry/impls/dual_rmsnorm.triton.card.json` | From `@targets(triton=…)`: `arch.family=any`, empty knobs, `authored_by="dsl"`. | Was hand-written; now generated. |
| Registered callable + input-gen case | Compute body → `@triton.jit`; default seeded generator from `inputs`+`symbols`. | Was the whole `.py`; now generated. |

`verify("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3")` runs unchanged — the
emitted card looks *identical* to a hand-written one. **The DSL is invisible to
the harness by design.** Note: `wave_size` never appears; the card honestly
claims `"wave_size": 0` (wave-agnostic row reduce) because *no wave primitive is
used* — inferred from absence, not remembered by a human.

---

## Example 2 — multi-target tiled bf16 GEMM *with a per-target sm_90 override*

> **Status: PARTIAL — the portable body (Triton) is CLOSED (Phase 2.0a, shipped
> 2026-06-30); the override MECHANISM is CLOSED (Phase 2.1, shipped 2026-06-30).**
> One `@kernel` source
> ([`src/xkernels/vkl/examples/gemm_bf16.py`](../../src/xkernels/vkl/examples/gemm_bf16.py))
> builds the math IR (`MMA`/`Pointwise`) that lowers to BOTH torch `matmul` (the
> bit-exact auto-reference) AND a generated tiled `tl.dot` K-loop. `verify(
> "gemm_bf16.triton@1.0.0")` passes on H100 with zero JSON hand-editing. The
> schedule-IR autotune (Phase 2.2a) took it to 460 TFLOPS = 97% of cuBLAS / 47% of
> the wgmma ceiling (the §2 gate's BELOW_BAR, honest). The per-target override
> *mechanism* (decorator + oracle-invariant check + card emission) shipped
> (Phase 2.1 CPU half); the native CUDA/CUTE + HIP/CK codegen is the remaining
> GPU-gated work, environment-blocked on this node (no cu12 nvcc). See
> `11-implementation-plan.md` §14–§15.

This is where the §4.1 vocabulary is load-bearing, where multi-target day 1
matters, and where perf day 1 forces an override body. The portable body (also
the reference) is shared; the sm_90 override reaches the NVIDIA ceiling.

```python
from vkl import kernel, targets, launch, Tensor, Numerics
from vkl.tiles import (load, store, zeros, mma, stage_async, scratch,
                       range, fp32, bf16)

@kernel(
    id="gemm_bf16@1.0.0", kernel="gemm",
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
@targets(
    triton=dict(arch="any",
                knobs={"BLOCK_M":[64,128,256], "BLOCK_N":[64,128,256],
                       "BLOCK_K":[32,64], "num_warps":[4,8], "num_stages":[2,3,4]}),
    cuda  =dict(arch="nvidia_sm90", requires=["tensor_cores","tma","clusters"],
                knobs={"BLOCK_M":[128,256], "BLOCK_N":[128,256], "stages":[3,4]}),
    hip   =dict(arch="amd_cdna3",   requires=["matrix_cores","mfma"],
                knobs={"BLOCK_M":[128,256], "waves_per_eu":[1,2], "stages":[3,4]}),
)
@launch(grid=lambda a, c, *, BLOCK_M, BLOCK_N:
        (c.shape[0]//BLOCK_M, c.shape[1]//BLOCK_N))
def gemm(ctx, a, b, *, BLOCK_M, BLOCK_N, BLOCK_K):
    """Portable body — correct on every target, AND the reference."""
    mi, ni = ctx.program_id(0), ctx.program_id(1)
    acc = zeros((BLOCK_M, BLOCK_N), fp32)                 # fp32 accumulation enforced
    for k in range(0, a.shape[1], BLOCK_K):
        a_tile = stage_async(load(a[mi:mi+BLOCK_M, k:k+BLOCK_K]), into=scratch)
        b_tile = stage_async(load(b[k:k+BLOCK_K, ni:ni+BLOCK_N]), into=scratch)
        acc = mma(a_tile, b_tile, acc, accum=fp32)        # tensor cores / MFMA by target
    store(acc.to(a.dtype), ctx.out.c[mi:mi+BLOCK_M, ni:ni+BLOCK_N])


# ---- per-target override: reach the sm_90 ceiling (TMA + clusters + wgmma) ----
@gemm.target("cuda", arch="nvidia_sm90")            # Axis H1 (full-body override)
def gemm_sm90(ctx, a, b, *, BLOCK_M, BLOCK_N):
    mi, ni = ctx.program_id(0), ctx.program_id(1)
    a_desc = ctx.tma_descriptor(a, tile=(BLOCK_M, BLOCK_K))   # TMA, not plain load
    acc = zeros((BLOCK_M, BLOCK_N), fp32)
    with ctx.cluster(2):                                       # thread-block cluster
        for k in range(0, a.shape[1], BLOCK_K):
            acc = wgmma(a_desc[k], b_tile, acc)               # native wgmma instr
    store(acc.to(a.dtype), ctx.out.c[mi:mi+BLOCK_M, ni:ni+BLOCK_N])
```

### What this example establishes

- **No `32`, no `smem`, no `tensor_cores` literal in either body.** `scratch` is
  a handle (kind bound per target); `mma`/`wgmma` pick their instruction by
  target+dtype+shape; `stage_async` picks its copy engine by target. The §4.1
  table is the API.
- **`@targets` literally is the Impl Card set** — three entries → three cards
  (`gemm.triton`, `gemm.cuda`, `gemm.hip`).
- **The override reaches the ceiling without breaking the contract.** `gemm_sm90`
  passes `verify` against the *same* auto-reference (the portable body on CPU),
  because both compute `C = A @ B` in fp32 accumulation. The override is *more*
  native code, not a *different* op.
- **`verify_parity("gemm_bf16@1.0.0")` is the gate** — all three targets share
  one reference, so §5.3 parity is meaningful the moment the cards are emitted.
- **Perf day 1 is a deliverable, not aspirational:** the `cuda` card's
  `perf.measured` must record a roofline fraction *for the override*, and the
  autotune sweep searches the override's knob space, not the portable body's.

### What this example deliberately does NOT claim

- That `mma`/`wgmma`/`stage_async` as sketched are rich enough for every sm_90 /
  CDNA3 feature. Whether H2 (primitive-level override, the elegant case) suffices
  or H1 (full-body, shown above) dominates is the empirical Axis-H question
  `06` gates Phase 1 on.
- That the override is *done* when it compiles. Reaching the roofline is still
  finished by the autotune sweep + the tuning skills; the DSL provides the
  *correct native starting point* so that loop starts hot.

---

## Example 3 — a 2-kernel graph (rmsnorm → gemm), captured

Single-kernel perf is Example 2; launch-overhead perf needs composition capture
(`07`). A chain where fusion is illegal (different grids, reduction boundary)
but launch overhead still bites:

```python
from vkl import graph

@graph(
    id="rmsnorm_gemm@1.0.0",
    captures=True,                         # → CUDA graph + HIP graph, not 2 launches
    params=["x", "w", "w2"],               # runtime-varying → parameter nodes
)
def rmsnorm_gemm(x, w, w2):
    xn = rmsnorm(x, w)                     # node 1 (a real @kernel from Ex.1-style)
    y  = gemm(xn, w2)                      # node 2; depends on node 1 — graph edge
    return y
```

### What `vkl build rmsnorm_gemm` emits (per target)

- A **composed Op Spec** (`registry/ops/rmsnorm_gemm.spec.json`) whose
  `inputs`/`outputs` are the graph boundary (`x, w, w2 → y`) and whose
  `numerics.reference` is the *composed* auto-reference (CPU eval of the chain).
- One **graph Impl Card per target** (`rmsnorm_gemm.cuda.card.json`,
  `rmsnorm_gemm.hip.card.json`, `rmsnorm_gemm.triton.card.json`), each with a
  namespaced `launch: { graph: true, nodes: [...], params: [...] }` field.
- The target's emitted host code:

```c
// CUDA target emission (sketch)
cudaGraph_t g; cudaGraphCreate(&g, 0);
cudaGraphNode_t n1 = add_kernel_node(g, rmsnorm_kernel, {x,w});      // params: x,w
cudaGraphNode_t n2 = add_kernel_node(g, gemm_kernel,    {xn,w2}, deps={n1}); // w2
cudaGraphExec_t exec; cudaGraphInstantiate(&exec, g, ...);
// runtime: setParams(x,w,w2) + cudaGraphLaunch(exec, stream)  — ONE launch
```

HIP emission is the same shape (`hipGraph_t`, `hipGraphInstantiate`,
`hipGraphLaunch`, `hipGraphExecKernelNodeSetParams`). **One DSL source → two
graph backends**, both validated against the composed reference by `verify`,
which runs the *whole graph* as one launchable unit. `verify_parity` then checks
the CUDA graph and HIP graph agree.

### What this example establishes

- **Graph capture is ordinary-looking code.** The author writes kernel calls; the
  dataflow (`xn → y`) is the edge; the emitter builds the DAG. No manual
  `cudaGraphAddDependencies` in user code.
- **Parameter nodes make it reusable** — one captured graph serves every
  `(x, w, w2)` shape/arg, paying instantiate once and `setParams`+launch thereafter.
- **The substrate sees a normal card.** An old consumer that ignores
  `launch.graph` runs the kernels sequentially (functionally identical); a
  graph-aware runtime captures. Functional portability holds either way; the
  graph is the performance-portability path. (`07` §7.)
- **Graph × fusion composes** — if `gemm` had a `bias+gelu` epilogue (Example 2 +
  an `@epilogue` hook), node 2 of the graph is the *fused* kernel. Fuse what you
  can, graph-capture the rest. (`07` §6.)

---

## What the strawman is meant to establish

1. **The authoring surface is plausible** across the full v0.2 ambition —
   portable, multi-target-with-override, and graph-captured, all readable.
2. **Emission is mechanical and round-trippable** — every artifact is consumed
   by *unchanged* substrate code (`verify`, `find_impl`, the schemas), via
   namespaced extensions where needed (graph metadata).
3. **The reference-derivation is the killer feature** of the lower layers — "same
   code, two backends" structurally guarantees §5.1's backend-neutral reference,
   including for composed graphs.
4. **Per-target override is how perf is bought honestly** — not by pretending one
   source is fast everywhere (§10), but by giving each target its own ceiling
   body against the same contract.
5. **The open questions survive the sketch** — the H1-vs-H2 override question and
   the conditional-node coverage question (`07` §4.3) are not resolved by the
   strawman; only shown *expressible*. `06` keeps them open and gates Phase 1.
