# Issue #41 — fp8 block-scale dense GEMM perf: native fp8 MFMA fast path on gfx942

**Date:** 2026-06-12
**Status:** Approved
**Issue:** ResearchComputer/xkernels#41 (follow-up to #38 / PR #40)
**Target hardware:** AMD Instinct MI300A (gfx942, CDNA3), ROCm 7.2 / torch 2.11.
**Branch:** `feat/issue-41-fp8-mfma-blockscale-gemm`, stacked on
`feat/issue-38-fp8-blockscale-gemm` (PR #40, open/unmerged — `main` has no
`mm_fp8_blockscale` yet). Rebase onto `main` after #40 merges.
**Precedent for shipping correct-but-unrewarded perf work:** #17, #20.

## Purpose

PR #40 (issue #38) shipped a **portable, correctness-first** Triton fp8
block-scale dense GEMM (`mm_fp8_blockscale`) for gfx942 — the V4 MLA
(`q_a`/`kv_a`/`q_b`/`kv_b`), `gate`, and shared-expert projections. It is correct
on-device (rel 7.1e-4 – 8.7e-4) but **not a speedup**: it ships *off* the
latency-critical path and loses to the `torch_mm_fp8_blockscale` reference on
every V4 shape (bf16/ref 0.42×–0.77×), sustaining only ~21–37 TFLOP/s vs the
~400 TFLOP/s gfx942 ceiling (#17).

Two root causes, both addressed here:

1. **No native fp8 MFMA.** #40 dequantizes fp8→fp32/bf16 in registers then runs
   `tl.dot` on the *widened* operands, so it never uses the CDNA3 fp8 matrix
   path. The torch reference, despite materializing fp32, routes its matmul
   through a tuned BLAS that wins.
2. **Naive tiling.** Fixed 64³ tiles (forced down from 64×128 after an LDS
   `OutOfResources`), no autotune, no software pipelining.

This issue is the **fast path**: `tl.dot` directly on fp8 e4m3 operands with the
block scales applied as a post-accumulation correction, plus an autotuned tile
space.

## The math — two-level (block-promoted) accumulation

Standard DeepSeek block-scale layout (`block = 128`):

- `A [M, K]` fp8 e4m3, **per-token-group** scale `A_scales [M, kt]`
  (`kt = ceil(K/128)`); `a_s[m, kb]` scales `A[m, kb*128 : (kb+1)*128]`.
- `B [N, K]` fp8 e4m3 (Linear orientation), **per-block** scale
  `B_scales [nt, kt]`; `b_s[nb, kb]` scales the `[nb*128:(nb+1)*128,
  kb*128:(kb+1)*128]` tile.

The scales are **constant within a 128-K quant block**, so they factor out of the
inner contraction:

```
out[m,n] = Σ_k A_deq[m,k]·B_deq[n,k]
         = Σ_kb  a_s[m,kb]·b_s[n//128,kb] · ( Σ_{k ∈ block kb} A_fp8[m,k]·B_fp8[n,k] )
```

The inner sum is a **raw fp8·fp8 partial GEMM** — exactly the native fp8 MFMA.
Per quant K-block `kb`: accumulate the raw partial in an fp32 block-accumulator,
then **promote** it into the main fp32 accumulator scaled by the (per-row × scalar
per N-block) correction.

**Numerical claim (parity is tight, not bf16-loose):** fp8·fp8 products are exact
in fp32 (3-bit mantissa operands), and the intra-block partial sum and the
promotion are fp32. This is *the same real arithmetic* as #40's exact fp32-dequant
path — only the order (scale-after-block-sum vs scale-per-element) differs — so
parity vs the fp32 dequant oracle should land near #40's **~1e-3**, not the bf16
path's ~2e-2. We assert a tight tolerance and use it as the **format-correctness
detector** (see Risk).

## Design

Stacks on the #38 gemm package. The #40 portable kernel is **kept** as the
correctness fallback; the fast path is **added** alongside it.

```
src/xkernels/ops/gemm/triton/
  mm_fp8_blockscale_kernel.py        # KEEP — #40 portable dequant-then-dot path
  mm_fp8_blockscale_mfma_kernel.py   # NEW  — native fp8 MFMA, block-promoted acc
  configs.py                         # NEW  — CDNA3 autotune space + tuned table
```

### Kernel (`mm_fp8_blockscale_mfma_kernel.py`)

One Triton program per `(pid_m, pid_n)` output tile:

```python
acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
for kb in range(kt):                          # quant K-blocks of size BLOCK=128
    pacc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for ki in range(0, BLOCK, BLOCK_K):       # BLOCK_K | 128; one step if BLOCK_K==128
        a = load A_fp8[rows, kb*128+ki : +BLOCK_K]     # [BLOCK_M, BLOCK_K] fp8, masked
        b = load B_fp8[cols, kb*128+ki : +BLOCK_K]ᵀ    # [BLOCK_K, BLOCK_N] fp8, masked
        pacc += tl.dot(a, b)                  # ← NATIVE fp8 MFMA, fp32 accumulate
    a_sc = load a_s[rows, kb]                 # [BLOCK_M]   per-row group scale
    b_sc = load b_s[cols // 128, kb]          # [BLOCK_N]   per-column block scale
    acc += pacc * a_sc[:, None] * b_sc[None, :]
store acc.to(out_dtype) → C[rows, cols]       # masked
```

- The operands stay in their **fp8 dtype** into `tl.dot` (no pre-dequant) — that
  is what routes to the fp8 matrix path.
- `b_s[cols // 128, kb]` is loaded **per output column**, which generalizes #40's
  "`BLOCK_N` must divide 128 → single scalar" constraint. `BLOCK_N` becomes a free
  tuning knob (incl. 256, spanning >1 N-block); cost is `BLOCK_N` scalar loads per
  K-block — negligible against the dot. `BLOCK_K` must still divide 128 so each
  `tl.dot` slice carries one `(a_s, b_s)`.
- Masking: `rows<M`, `cols<N`, and `ks<K` for the trailing partial K-block. A
  partial block uses its (correct) per-block scale like a full one. `M==0`/`N==0`
  short-circuit in the wrapper.

### Autotune (`configs.py`, mirrors `ops/moe/triton/configs.py`)

CDNA3-reasoned `triton.Config` space, the AMD lowering knobs passed through the
kwargs dict (read by the `tokenspeed_triton` AMD backend, ignored elsewhere):

- `BLOCK_M ∈ {16, 32, 64, 128, 256}` — decode (M=1..8) wants tiny M; prefill
  (M=2048/4096) wants large.
- `BLOCK_N ∈ {64, 128, 256}`, `BLOCK_K ∈ {64, 128}` (divides the quant block).
- `num_warps ∈ {4, 8}`, `num_stages ∈ {1, 2}`.
- `matrix_instr_nonkdim ∈ {16, 32}` (fp8 MFMA `16x16x32` / `32x32x16`),
  `waves_per_eu`, `kpack`.

fp8 operands are **half the LDS bytes** of #40's fp32 tiles, so `BLOCK_K=128`
plus pipelining should fit the 64 KB CDNA3 LDS — directly resolving the
`OutOfResources` that forced 64³. A shape-keyed tuned table + a `prune_configs`
(LDS / occupancy) gate, like `get_moe_int4_config`. `@triton.autotune` keyed on
`(M-bucket, N, K)`; a `GROUP_SIZE_M`-style L2 swizzle is optional (defer unless it
shows up in the bench).

### Wiring (`mm_fp8_blockscale_triton` entry)

Choose mfma ↔ portable. Per the **default-only-if-wins** policy:

- Add a `path: {"auto","mfma","portable"}` (or reuse/extend the existing
  `dot_bf16` knob path-style) argument to the Triton wrapper.
- `"auto"` selects mfma on shapes a tuned table marks faster than `torch_ref`,
  else the portable path. Until on-device numbers exist, `"auto"` = mfma with the
  portable path as the safe fallback on any compile/shape failure.
- Public `mm_fp8_blockscale(...)` signature is **unchanged** for existing callers;
  the new knob is keyword-only with a back-compatible default. The reference
  backend ignores it (signature parity, like `dot_bf16`).

### Format risk — e4m3**fn** vs e4m3**fnuz** (the #1 on-device unknown)

Operands are `torch.float8_e4m3fn` (OCP, bias 7, max 448) from the #38 quant
helpers. CDNA3's native fp8 MFMA historically consumes the **fnuz** encoding
(bias 8, max 240). Three outcomes on rocm7.2 + `tokenspeed_triton`, resolved
on-device **before** tuning:

1. `tl.dot(e4m3fn)` lowers directly to fp8 MFMA (Triton inserts a value-preserving
   HW path) → **tight parity + high TFLOP/s**. Best case; nothing more to do.
2. Silent upcast → bf16 MFMA (≈#40, no fp8 win). Detect: TFLOP/s stuck near the
   bf16 path **and** no `v_mfma_*_fp8*` in the AMDGCN dump.
3. Lossy fn→fnuz (clips |v|>240, ≈half the quantized range) → **loose parity** —
   the automatic diagnostic.

**Fallback for (2)/(3):** feed the MFMA the fnuz format it wants. Two options,
chosen by what the on-device probe shows:
- fnuz-targeted quant helpers (`per_*_quant_fp8(..., fp8_dtype=float8_e4m3fnuz)`
  with matched `FP8_MAX`) so operands arrive fnuz on AMD; or
- a cheap in-op fn→fnuz requant (elementwise, amortized against the GEMM).

The kernel accepts either fp8 dtype (reads `a_fp8.dtype.element_ty`); the tight
GPU parity assertion is the gate that confirms whichever format actually reached
the matrix unit preserved the operand values.

## Testing & validation

- `tests/test_mm_fp8_blockscale_mfma.py`:
  - **Interpreter** (`TRITON_INTERPRET=1`, CPU fp32): tiling, masking, the
    block-promotion index math vs the fp32 dequant oracle. (fp8 `tl.dot` under the
    CPU interpreter falls back to fp32 mul — exercises the math, not the MFMA.)
  - **GPU** (gfx942): native-fp8-MFMA path vs oracle at a **tight** norm-relative
    tolerance (`< 5e-3`, the format detector); block-aligned and odd M/N/K (incl.
    K not a multiple of 128), decode `M=1`, bf16 + fp32 out, empty `M`. V4 MLA
    shape (M=8, N=512, K=7168).
  - Cross-check that the mfma path agrees with the #40 portable path within fp8
    tolerance (two independent implementations of the same op).
- `slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch` (mirrors
  `slurm/test_mm_fp8_blockscale_beverin.sbatch`): on-device gfx942 —
  `pytest` the new file (real compile, `TRITON_INTERPRET` unset) + a standalone
  V4-shape parity max|err| + **perf bench** (mfma vs `torch_ref` vs #40 portable
  fp32/bf16, TFLOP/s) + an **AMDGCN fp8-MFMA assertion** (dump the compiled kernel,
  grep for `v_mfma_*_fp8*`) to confirm root cause #1 is actually fixed.
- `benchmarks/bench_fp8_blockscale_gemm.py`: the V4 MLA shapes
  (1/8/2048×512×7168, 4096×7168×2048), mfma vs `torch_ref` vs #40 portable, TFLOP/s
  + ×ref. Wire into `benchmarks/bench_all.py` / README Performance row **only if it
  wins** (else it stays a standalone honest-result bench, like #17's probe).
- `docs/issue-41-fp8-mfma-blockscale-gemm.md`: the shipped kernel doc — the
  block-promotion math, the fp8-format resolution, the autotune table, and the
  honest on-device perf numbers (#38-style table).

## Ship policy (locked)

Default the mfma path **only on shapes measured faster than `torch_ref`** on
beverin; elsewhere keep it opt-in and document the honest result (the #17/#20
precedent). The portable #40 kernel remains the always-correct fallback.

## Out of scope

- Replacing or deleting the #40 portable kernel (kept as fallback).
- tokenspeed-side serving binding / backend selection for the MLA projections.
- A Gluon rewrite (explicit `amd_mfma` + LDS double-buffer) — the follow-up once
  the autotuned Triton config space is validated on real gfx942, as noted for the
  MoE INT4 kernel.
- Tuning `block` ≠ 128 (DeepSeek convention is fixed at 128).
- fp8 e5m2 operands (V4 block-scale is e4m3).
