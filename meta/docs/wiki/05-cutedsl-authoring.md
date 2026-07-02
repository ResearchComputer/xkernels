# Authoring CUTE DSL (`cutlass.cute`) kernels — what works, what bites

Recorded from the 2026-06-29 work that added the **`cuda` backend** to xkernels
via NVIDIA's CUTE Python DSL and landed **5 verified cards** on a DGX Spark
(Grace+Blackwell **GB10 = sm_121**, CTK 13.0). Every API call, signature, and
gotcha below was discovered by reading the installed package source + probing on
the GPU, **not from memory** — the DSL has essentially no standalone tutorial
documentation, so this page is the map an agent authoring the next card needs.

This is a *portability layer for one vendor* (NVIDIA), **not** a cross-vendor
backend. It slots in as `backend: "cuda"` Impl Cards under the existing Op
Specs, one implementation among others; the Triton + reference backends stay the
portable path (per `meta/docs/library.md`).

> **TL;DR.** A CUTE DSL kernel is three functions: `@cute.kernel` (device),
> `@cute.jit` (host tiler), and a torch-facing launch fn. The two things that
> will eat your time are (1) the **per-call launch overhead** — `@cute.jit`
> `__call__` rebuilds the MLIR engine every call (~9 ms); the fix is a cached
> `cute.compile()` handle launched **tensors-only** (~40 µs, 119×) — and (2)
> that the API surface is found by **grep, not memory** (math is lowercase
> `math.rsqrt(x)` from `_mlir.dialects.math`; reductions are
> `warp_reduction_sum` from `cutlass.cute.arch`). The 5 cards in
> `src/xkernels/ops/{gemm,norm,moe,attention,mhc}/cute/` are the working
> templates — copy the closest one.

## 1. The package (the first trap)

PyPI naming is hostile. Three packages, one real:

| package | installs | use? |
|---|---|---|
| `cutlass` (0.5.0) | an **unrelated squatter** (`linear_model`/`metrics`) | ✗ |
| `nvidia-cutlass` (4.2.0.0) | `pycute` (the pure-Python CUTE *layout* algebra) + codegen — **not** the compiler | ✗ |
| **`nvidia-cutlass-dsl`** (4.5.2) | the top-level `cutlass` module → `cutlass.cute` (the DSL compiler) | ✓ |

The CUDA-toolkit must match: install **`nvidia-cutlass-dsl[cu13]`** on a CTK-13
box (the `[cu13]` extra). On ds5 the xkernels `cute` extra pins exactly this
(`pyproject.toml`). The DSL invokes **nvcc** under the hood, so
`export CUDA_HOME=/usr/local/cuda-13.0` (+ `$CUDA_HOME/bin` on PATH) is mandatory
before any JIT.

## 2. The canonical import block

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, math          # thread regs + math intrinsics
from cutlass.cute.runtime import from_dlpack           # torch <-> CUTE tensor bridge
from cutlass.cute.typing import Tensor                 # the Tensor type for signatures
from cutlass.cutlass_dsl import T                      # T.i32() for the read_ptx_sreg calls
# reductions / smem / barriers (only when you need them — see §7):
from cutlass.cute.arch import alloc_smem, sync_threads, warp_reduction_sum
```

`from_dlpack(torch_tensor)` gives a `Tensor` the kernel indexes; the same tensor
is the output sink (zero-copy). The `Tensor` type is for **annotations only**
(it is what `@cute.kernel` parses to build the MLIR signature).

## 3. The three-function structure

Every card is `@cute.kernel` (device) + `@cute.jit` (host) + a plain Python
launch function. The minimal skeleton (this is `_vecadd` in
`src/xkernels/ops/_cute_backend/smoke_vecadd.py` — the working "hello world"
that proves JIT + run on a new arch):

```python
_BLOCK_THREADS = 128

@cute.kernel
def _kernel(gIn: Tensor, gOut: Tensor, n: cutlass.Constexpr) -> None:
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())     # thread id
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())   # block  id
    i = bidx * _BLOCK_THREADS + tidx
    if i < n:                                    # bounds predication is MANDATORY (tail CTA)
        gOut[(i,)] = gIn[(i,)] * cutlass.Float32(2.0)

@cute.jit
def _host(In: Tensor, Out: Tensor, n: cutlass.Constexpr) -> None:
    _kernel(In, Out, n).launch(
        grid=[(n + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )

def my_op_cute(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    gIn, gOut = from_dlpack(x), from_dlpack(out)
    _host(gIn, gOut, x.numel())          # NOTE: slow path — see §5
    return out
```

Key shape facts the skeleton encodes:

- **`cutlass.Constexpr`** is how you pass a baked-in value. It specializes the
  compiled kernel — two different `n` produce two compiled kernels. (This is
  load-bearing for §5: the handle caches *per Constexpr tuple*.)
- **Integer tensor indexing** (`gIn[(i,)]`, `gOut[(m, n)]`) is the simplest
  access pattern and is what every one of the 5 cards uses. `None` broadcasts:
  `gA[(m, None)]` returns the K-row `A[m, :]`; `gB[(None, n)]` the K-column.
  Multi-dim indexes map to the tensor's logical shape (row-major).
- **`cutlass.Float32(0.0)` / `cutlass.Int32(0)`** are the typed literals — use
  them for accumulators/loop counters so the DSL infers the right MLIR type.
  Plain `0.0` works in many spots but typed literals are unambiguous.
- **`@cute.jit` is the host entry**; it builds the launch and (optionally) the
  CTA tiler. For simple one-CTA-per-row / one-CTA-per-tile kernels it is just the
  `.launch(grid=, block=)` wrapper around the kernel call.

The fancier static-layout path (`composition`, `zipped_divide`,
`make_copy_atom`, `make_rmem_tensor`) — the real CUTE tiling algebra — is used in
`smoke_vecadd.py`'s `_vecadd_kernel` (copied from
`cutlass.cute.testing._convert` in the installed package). The 5 production cards
do **not** need it: integer indexing + per-thread tiles is enough for
correctness at the sweep sizes, and the matrix-engine path (which would need
static layouts) is gated (§10). Reach for the layout algebra only when tiling
through SMEM with `cp.async`/TMA — a documented high-effort follow-up, not the
default.

## 4. Thread / CTA indexing

```python
tidx = nvvm.read_ptx_sreg_tid_x(T.i32())     # lane in block
bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())   # block x
bidy = nvvm.read_ptx_sreg_ctaid_y(T.i32())   # block y
```

`T` is `cutlass.cutlass_dsl.T` — the type helper. `read_ptx_sreg_*` returns the
raw PTX special register; `T.i32()` gives it a type. **Do not guess alternatives**
(`cute.thread_idx()`, `cute.block_idx()` etc. don't exist in this build) —
`nvvm.read_ptx_sreg_*` is the canonical way and is what every vendored torch
CUTE kernel uses. Warp/lane decomposition when you need it:

```python
warp_id = tidx // 32
lane    = tidx % 32
```

## 5. The compile-cache pattern — the one big perf lever

**This is the single most important fact on the page.** Left as-is, the
`@cute.jit.__call__` path (the skeleton in §3) **rebuilds the MLIR execution
engine on every call** (~9.3 ms steady, 117 ms cold), even with identical tensors
and identical Constexpr values. ncu shows the actual GPU dispatch is ~90 µs, so
the GPU sits idle ~99% of the time. This makes every card look 100× slower than
it is and will mask every kernel-level optimization.

The fix: compile once into a reusable handle, cache it per Constexpr tuple,
launch **tensors-only**.

```python
_COMPILED_HANDLE_CACHE: dict[tuple, object] = {}

def my_op_cute(x, ...):
    ...
    gIn, gOut = from_dlpack(x), from_dlpack(out)
    key = (M, N, K)                              # the Constexpr tuple
    handle = _COMPILED_HANDLE_CACHE.get(key)
    if handle is None:
        _host(gIn, gOut, M, N, K)                # warmup via the slow path (preprocesses AST)
        torch.cuda.synchronize()
        handle = cute.compile(_host, gIn, gOut, M, N, K)   # compile-once
        _COMPILED_HANDLE_CACHE[key] = handle
    handle(gIn, gOut)                            # fast launch — TENSORS ONLY
    return out
```

Measured on ds5: **9307 µs → 41.6 µs/call (223× at the GEMM shape, 119×
end-to-end)**, identical numerics. Source: `scripts/archive/ds5-probes/ds5_c_solved_test.py`,
`scripts/archive/ds5-probes/ds5_jitcache_probe.py`. This pattern is duplicated verbatim in all 5
cards' launch functions (`_COMPILED_HANDLE_CACHE` + `cute.compile` + tensors-only
launch).

**The load-bearing gotcha:** the handle specializes on the Constexpr at compile
time, so it MUST be launched with ONLY the tensor args. Re-passing the Constexpr
corrupts the TVM-FFI execution-args ABI and **SEGFAULTS** in the native launch
(an uncatchable SIGSEGV, not a Python exception). Proven in
`scripts/archive/ds5-probes/ds5_c_final_probe.py`: `handle(gA, gB, gOut)` OK;
`handle(gA, gB, gOut, M, N, K)` segfaults. Hence the handle is cached **per
Constexpr tuple** and always called with the bare tensor list.

> If you take one thing from this page: never ship a CUTE card on the bare
> `@cute.jit` `__call__` path. Always `cute.compile` + cache + tensors-only.

## 6. Math intrinsics — lowercase `math.opname(x)`

Available: `math.rsqrt`, `math.sqrt`, `math.exp`, `math.log`, `math.absf`,
`math.sin`/`cos`/`tanh`, … from `cutlass._mlir.dialects.math`. **The calling
convention is lowercase function call**, not the MLIR Op class:

```python
from cutlass._mlir.dialects import math
scale = math.rsqrt(mean + eps)      # ✓ lowercase fn
exp_a = math.exp(lse_a - m)
lse   = m + math.log(denom)
# NOT: math.RsqrtOp(x).results[0]   ✗ that's the raw MLIR builder, also works but ugly
```

Confirmed by `scripts/archive/ds5-probes/ds5_dsl_math_probe2.py`, which grepped the installed
package for the canonical usage and tried all three conventions (`RsqrtOp(x).results[0]`,
`math.rsqrt(x)`, `RsqrtOp(x).result`) — `math.rsqrt(x)` is the DSL's own
convention (it's what `arith.py` in the package emits). `exp`/`log` are the
**natural** (base-*e*) versions, matching torch — the triton card's log2+`/LOG2E`
is an equivalent optimization the DSL does not require you to do by hand.

`max`/`min`/conditional are **plain Python** (`if lb > m: m = lb`), not intrinsics
— the DSL lowers them. See `_merge_state_kernel` for `max` via conditional.

## 7. Reductions, SMEM, and barriers

The block-wide reduction primitive set (from `cutlass.cute.arch`):

```python
from cutlass.cute.arch import alloc_smem, sync_threads, warp_reduction_sum

smem = alloc_smem(cutlass.Float32, n_slots)     # SMEM buffer of n_slots fp32
smem[i] = val                                    # write
v = smem[i]                                      # read
sync_threads()                                   # __syncthreads
partial = warp_reduction_sum(acc, threads_in_group=32)   # per-warp reduce
```

The two-pass reduction pattern (one CTA per row, 128 threads = 4 warps) is in
`src/xkernels/ops/norm/cute/rmsnorm_kernel.py` and is the template for any
row-reduction:

```python
# Pass 1: thread-stride Kahan sum-of-squares
acc = cutlass.Float32(0.0); c = cutlass.Float32(0.0)
col = tidx
while col < D:
    x = gX[(row, col)]
    sq = x * x
    ...                                          # Kahan fold
    col = col + _BLOCK_THREADS

acc = warp_reduction_sum(acc, threads_in_group=32)   # each lane now has warp's sum
if lane == 0:
    smem[warp_id] = acc
sync_threads()

if tidx == 0:                                    # thread 0 folds the 4 warp partials
    total = smem[0] + smem[1] + smem[2] + smem[3]
    smem[NUM_WARPS] = math.rsqrt(total / D + eps)    # broadcast scale
sync_threads()
scale = smem[NUM_WARPS]                          # every thread reads it
```

Confirmed working on sm_121 by `scripts/archive/ds5-probes/ds5_dsl_rowsum_probe.py`. The primitives
were found by reading `cutlass/cutlass_dsl/cuda_jit_executor.py` +
`cutlass/cute/arch/__init__.py` in the installed package — **not** by guessing.
A whole-device reduction (grid-wide) is not needed for any current card; it
would need atomic adds or a second kernel.

**When you DON'T need a block reduce:** if the reduction axis is small
(`moe_sum_reduce`'s `top_k=8`), a **per-thread serial Kahan** is simpler and
faster — one CTA per token row, each thread does its own 8-element sum. Don't
reach for `warp_reduction_sum` + SMEM unless the reduction axis is larger than a
warp can hold per thread. (Both patterns are in the 5 cards.)

## 8. Kahan compensated summation (use it by default)

Every reduction in the 5 cards is **Kahan-compensated**. fp8-block-scaled and
bf16-cast operands have per-element/per-block magnitude variation, so a naive
sequential sum loses low-order bits vs torch's tree reduction and can exceed the
op's fp32 `rtol` (1e-3) on ill-conditioned elements. The compensation term
recovers it, bringing agreement to ~1e-6 — *marginally more accurate* than
torch's blocked reduction. Template (from `_fp32_matmul_kernel`):

```python
acc = cutlass.Float32(0.0); c = cutlass.Float32(0.0)
k = cutlass.Int32(0)
while k < K:
    term = a_row[(k,)] * b_col[(k,)]
    y = term - c
    t = acc + y
    c = (t - acc) - y
    acc = t
    k = k + 1
```

Cheap (one extra FMA), unconditionally improves numerics, costs nothing here.
Use it for any K-reduce / sum-of-squares / weighted-sum.

## 9. The `T` shadowing gotcha (and the dtype-plumbing convention)

**Never name a `cutlass.Constexpr` `T`** — it shadows `cutlass.cutlass_dsl.T`,
the type used in `T.i32()`, turning `nvvm.read_ptx_sreg_tid_x(T.i32())` into
`AttributeError: 'int' object has no attribute 'i32'`. `hc_prenorm_gemm` hit
this (constexpr was `T` for "tokens"); renamed to `ROWS`. The other cards dodged
it by using `M`/`N`/`K`/`H`/`D`/`top_k`. Use any of those, not `T`.

**Host dtype plumbing convention** (matches the reference + triton cards): upcast
bf16/fp8 inputs to fp32 on the host (bit-identical to the reference's
`x.float()`), run the kernel in pure fp32, cast the result back to `x.dtype` in
the caller. This isolates the GPU work to a correct fp32 op. **Exception** — see
§11 for the bf16-native-read perf optimization that *doesn't* upcast.

## 10. What is NOT available on sm_121 / CTK-13.0 (be honest about it)

The matrix engine is gated for these ops on this stack. Do not assume an MMA path
exists — verify, and if not, ship the honest fp32-FMA path:

| op | path | status on sm_121/CTK-13.0 |
|---|---|---|
| fp8 block-scale MMA | `MmaFP8Op` (m16n8k32→fp32) | **gated on CTK ≥ 13.1** |
| sm_121 native blockscale | `MmaSM120BlockScaledOp` | exists but is **MX microscaling** (e8m0/e4m3), not DeepSeek's fp32 block=128 scales → wrong contract |
| bf16 MMA | tensor cores | would fail the op's fp32 sweep point (rtol 1e-3) |

So `mm_fp8_blockscale.cuda` dequants to fp32 on the host and runs a fp32 FMA
GEMM. This is honestly non-peak (the card says so); the matrix-engine path is a
documented follow-up gated on CTK 13.1 landing on ds5. Find the MMA ops by
grepping `cutlass/cute/atom/__init__.py` + the vendored
`torch/_inductor/kernel/vendored_templates/cutedsl/` templates for the
`Mma*Op` names and their `cc`/`gpu_arch` guards — the guards are how we learned
fp8 MMA needs 13.1.

## 11. bf16-native-read (promote-on-load) — the real perf lever for memory-bound kernels

For a **memory-bound** kernel, host-upcasting a bf16 input to fp32 *both* adds a
separate upcast launch *and* **doubles the kernel's read traffic** (29 MB vs
14.7 MB). The DSL promotes bf16→fp32 on load **losslessly** (the cast is exact),
so you can read bf16 natively and accumulate in fp32 — bit-identical to the
reference's `x.float()`. Proven by `scripts/archive/ds5-probes/ds5_bf16_load_probe.py`:

```python
@cute.kernel
def _k(gBf: Tensor, gFp: Tensor, n: cutlass.Constexpr):
    ...
    b = gBf[(i,)]            # bf16 load
    acc = b * cutlass.Float32(2.0)   # promotes to fp32 in the multiply
    gFp[(i,)] = acc
```

Applied where traffic actually matters:
- **`mha_merge_state`**: reads two large bf16 tensors → e2e **0.084→0.042 ms
  (2.0×)**, achieved BW 28%→**56%** of GB10 peak.
- **`moe_sum_reduce`**: kernel dispatch 134→108 µs (20%).

**NOT applied** to `dual_rmsnorm`/`hc_prenorm_gemm` (launch-bound at sweep sizes,
<0.5 MB — bf16-read won't move them) or the GEMM (its fp32 inputs come from host
fp8→fp32 dequant, the op's design). Don't apply it blindly — roofline first
(`scripts/archive/ds5-probes/ds5_roofline_survey.py`), apply only where the op is memory-bound AND
the bf16 input is large enough to matter.

## 12. Negative results (recorded so you skip them)

Two optimizations that *looked* right and **regressed or did nothing** — both
diagnosed with ncu (read `.agents/skills/use-nsight-compute/SKILL.md` first):

- **2-way Kahan ILP** (two even/odd Kahan chains merged at the end, to attack the
  residual scoreboard stall on the GEMM): ncu showed scoreboard stall
  **11.6→15.4 cyc**, duration 90→102 µs. Occupancy rose 53→68% but didn't
  translate — the single chain is already at its ILP ceiling for the 8×16
  one-output-per-thread tile; the 2-way merge-dependency + register pressure
  outweighs the ILP gain. **Reverted.** The real lever is the matrix engine
  (gated), not scalar unrolling.
- **H-wave occupancy retile** on `moe_sum_reduce`: ncu flagged "grid too small"
  (0.22 waves/SM, grid=128 CTAs). Lifted the grid to [M,NUM_H_WAVES]=512 CTAs
  (0.89 waves/SM). Kernel time **unchanged (134→133 µs)** — occupancy was NOT the
  bottleneck; the ncu OPT message was a red herring. **Reverted** to the simple
  one-CTA-per-row grid; bf16-read (§11) was the real win.

**Lesson:** on these scalar memory-bound kernels, the ncu "grid too small /
optimize occupancy" OPTs are noise. The lever is memory traffic (§11) or the
matrix engine (§10), not occupancy or ILP.

## 13. Wiring a card into xkernels

Three files per op (the `gemm` card is the reference; `__init__.py` wraps the
import in the guard):

1. **`src/xkernels/ops/<type>/cute/{__init__.py, entry.py, <op>_kernel.py}`** —
   `entry.py` does host dtype plumbing + dequant + calls the kernel launch fn;
   `<op>_kernel.py` has the three-function structure (§3). `entry.py` ends:
   ```python
   from ...._backends import Backend, detect_vendor
   from ...._dispatch import register
   if detect_vendor() == "nvidia":
       register("<op>", Backend.CUDA)(my_op_cute)
   ```
   (Only register on NVIDIA — the CUTE DSL is NVIDIA-only.)
2. **`src/xkernels/ops/<type>/__init__.py`** — wrap the `from .cute import entry`
   in `backend_registration_guard("<op>", Backend.CUDA, source=...)`, same
   pattern as triton. The guard suppresses import errors when the DSL isn't
   installed (AMD box, no `cutlass.cute`).
3. **`registry/impls/<op>.cuda.card.json`** — `backend: "cuda"`,
   `arch.family: "nvidia_sm121"`, `perf.measured: [...]`. The schema
   (`registry/schema/impl_card.schema.json`) now admits `nvidia_sm100` +
   `nvidia_sm121` in the `arch.family` enum (added in this work).

**Device guard in every entry** (mandatory for `verify_parity` to not segfault):
`verify_parity()` hardcodes `device='cpu'`, so it reaches the entry with CPU
tensors. The `cute.compile` handle's native launch segfaults on a host pointer
(SIGSEGV, uncatchable). Raise a clean `RuntimeError` first:
```python
if not getattr(x, "is_cuda", False):
    raise RuntimeError("CUTE DSL kernel requires CUDA tensors; got device='cpu'. ...")
```
This lets the harness record `cuda` as a *caught backend error* in parity rather
than crashing the process. Genuine cross-backend agreement is measured on the GPU
(`verify(arch="nvidia_sm121")` against the shared reference).

## 14. Discovery methodology (how the API was found — reusable)

The DSL has no standalone tutorial. The API surface was reverse-engineered by
**grepping the installed package**, then probing on the GPU. Reuse this for any
unknown API:

```python
import os, glob, re, cutlass
ROOT = os.path.dirname(cutlass.__file__)
pat = re.compile(r"(warp_reduction_sum|alloc_smem|math\.\w+)\s*\(")
for fp in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
    txt = open(fp, errors="ignore").read()
    for m in pat.finditer(txt):
        print(os.path.relpath(fp, ROOT), txt[:m.start()].count("\n")+1, m.group(1))
```

Then **try the candidate conventions on a tiny kernel** (the probe pattern in
`scripts/archive/ds5-probes/ds5_dsl_math_probe2.py` / `scripts/archive/ds5-probes/ds5_dsl_reduction_probe.py`): define a 4-element
`@cute.kernel` that calls the op each way, compare vs torch on known values. The
two richest sources of canonical usage in the package:

- `cutlass/cute/testing/_convert.py` — the canonical minimal vecadd (the
  `smoke_vecadd.py` template).
- `torch/_inductor/kernel/vendored_templates/cutedsl/` — **authoritative,
  tested** production DSL kernels (blockscaled GEMM, grouped GEMM) that torch
  ships. These are the best models for a real card, though the blockscaled one is
  hard-gated to `cc ∈ {100,101,103}` (read the guard before copying).

Every probe script under `scripts/archive/ds5-probes/ds5_*.py` is kept as the reproducible trail —
if the DSL API moves in a future `nvidia-cutlass-dsl` release, re-run the
matching probe to re-derive the convention rather than trusting this page.

## The 5 working cards (templates by pattern)

| card | file | pattern it demonstrates |
|---|---|---|
| `mm_fp8_blockscale.cuda@1.0.0` | `ops/gemm/cute/mm_fp8_blockscale_kernel.py` | **GEMM** (tiled, Kahan K-reduce, B-transpose coalescing, host dequant) |
| `dual_rmsnorm.cuda@1.0.0` | `ops/norm/cute/rmsnorm_kernel.py` | **Block-wide reduction** (warp_reduce + SMEM + sync + rsqrt) |
| `moe_sum_reduce.cuda@1.0.0` | `ops/moe/cute/sum_reduce_kernel.py` | **Per-thread small reduction** + **bf16-native-read** |
| `mha_merge_state.cuda@1.0.0` | `ops/attention/cute/merge_state_kernel.py` | **Online-softmax** (exp/log/max) + bf16-native-read |
| `hc_prenorm_gemm.cuda@1.0.0` | `ops/mhc/cute/prenorm_gemm_kernel.py` | **Fused epilogue** (GEMM + squared-sum share the K-axis; add-epilogue-fusion case a) |

All 5 pass `verify` + `verify_parity` on ds5. Copy the closest one for the next
card; the three-function structure + the compile-cache + the device guard are
identical across all of them.

## Reproduce on ds5

```bash
rcc --profile ds5 push
rcc --profile ds5 run -s '\
  export PATH="$HOME/.local/bin:$PATH"; \
  export CUDA_HOME=/usr/local/cuda-13.0; export PATH=$CUDA_HOME/bin:$PATH; \
  cd /local/home/xiayao/xkernels && . .venv/bin/activate && \
  python -m xkernels.ops._cute_backend.smoke_vecadd'    # the hello-world self-check
# per-card verify + perf:
python scripts/archive/ds5-probes/ds5_verify_card.py mm_fp8_blockscale.cuda@1.0.0 mm_fp8_blockscale@1.0.0
python scripts/archive/ds5-probes/ds5_cute_perf.py
# the API probes (re-derive any convention that moved):
python scripts/archive/ds5-probes/ds5_dsl_math_probe2.py        # math intrinsics
python scripts/archive/ds5-probes/ds5_dsl_rowsum_probe.py       # reduction primitives
python scripts/archive/ds5-probes/ds5_bf16_load_probe.py        # bf16-native-read
python scripts/archive/ds5-probes/ds5_roofline_survey.py        # which regime each card is in
```

Full testbed runbook (environment setup, the roofline survey, all the perf-pass
narratives): [`meta/docs/usage/ds5-testbed.md`](../docs/usage/ds5-testbed.md).
