# Issue #41 — native fp8 MFMA fast path for the block-scale GEMM (gfx942)

The performance follow-up to [#38](issue-38-fp8-blockscale-gemm.md) (PR #40). #40
shipped a portable, correctness-first Triton fp8 block-scale dense GEMM
(`mm_fp8_blockscale`) for gfx942 that **dequantizes fp8→fp32/bf16 then runs
`tl.dot`** on the widened operands — never touching the CDNA3 native fp8 matrix
path. It is correct (rel 7e-4) but loses to the `torch_mm_fp8_blockscale`
reference on every V4 shape (~21–37 TFLOP/s vs the ~400 TFLOP/s ceiling, #17).

This issue adds the **fast path**: `tl.dot` directly on fp8 operands, with the
block scales applied as a post-accumulation correction, plus an autotuned tile
space. On the V4 MLA / gate / shared-expert shapes it runs **3.4–9.1× faster than
`torch_mm_fp8_blockscale`** and **6.8–10.3× faster than #40's portable kernel**,
reaching **359 TFLOP/s** at prefill (near the gfx942 ceiling). The #40 portable
kernel is kept as the always-correct fallback.

## The math — two-level (block-promoted) accumulation

DeepSeek block-scale layout (`block = 128`): `A [M,K]` fp8 e4m3 with per-token-group
scale `A_scales [M, ceil(K/128)]`, `B [N,K]` fp8 e4m3 with per-block scale
`B_scales [ceil(N/128), ceil(K/128)]`. The scales are **constant within a 128-K
quant block**, so they factor out of the inner contraction:

```
out[m,n] = Σ_k A_deq[m,k]·B_deq[n,k]
         = Σ_kb  a_s[m,kb]·b_s[n//128,kb] · ( Σ_{k ∈ block kb} A_fp8[m,k]·B_fp8[n,k] )
```

The inner sum is a **raw fp8·fp8 partial GEMM** — the native fp8 MFMA. Per quant
K-block `kb`: accumulate the raw partial in an fp32 block-accumulator via a fp8
`tl.dot`, then promote into the main fp32 accumulator scaled by
`a_s[:,None]·b_s[None,:]` (the A scale per row, the B scale loaded per output
column so `BLOCK_N` is a free tuning knob — generalizing #40's "`BLOCK_N` divides
128 → scalar" constraint). This is *the same real arithmetic* as #40's exact
fp32-dequant path — fp8·fp8 is exact in fp32, the intra-block sum and the
promotion are fp32 — so parity is tight (**rel 2–4e-6** on-device), and format
determines **speed**, not correctness.

The kernel lives in `triton/mm_fp8_blockscale_mfma_kernel.py`; the
`Backend.TRITON` entry (`triton/entry.py`) routes `path ∈ {auto, mfma, portable}`.

## The fp8 format that reaches native fp8 MFMA — fnuz, not fn

The operands the quant helpers produce are `torch.float8_e4m3fn` (OCP, bias 7,
max 448). CDNA3's native fp8 MFMA decodes the **fnuz** encoding (bias 8, max 240).
A standalone probe (`benchmarks/probe_fp8_mfma.py`, beverin) settled which format
gets there, dumping the AMDGCN and timing a 2048³-ish fp8 dot:

| operands | parity | MFMA instruction | throughput |
|---|---|---|---|
| `float8_e4m3fn` | exact (2e-7) | `v_mfma_f32_32x32x8_**f16**` (upcast) | 29.7 TFLOP/s |
| `float8_e4m3fnuz` | exact (2e-7) | `v_mfma_f32_32x32x16_**fp8_fp8**` ✅ | **373.9 TFLOP/s** |

So `e4m3fn` silently upcasts to an f16 MFMA (no fp8 win); **`e4m3fnuz` lowers to
the native fp8 MFMA at ~374 TFLOP/s** — a 12.6× gap. Both are numerically exact
(torch's fnuz decode matches the hardware), so the discriminator is the MFMA
mnemonic + throughput, not parity.

The kernel is therefore **fp8-format-agnostic** (it dots whatever dtype it is
given); the fast path is unlocked by feeding it fnuz operands. The quant helpers
gained `fp8_dtype=` (default `e4m3fn`; pass `torch.float8_e4m3fnuz` on AMD). The
on-device validation confirms the *real* kernel emits both
`v_mfma_f32_16x16x32_fp8_fp8` (small-N / decode tiles) and
`v_mfma_f32_32x32x16_fp8_fp8` (large-N prefill tiles).

## Autotune — N is the dominant axis

`get_fp8_gemm_config` is a baked direct-launch table (no per-call runtime
autotune), tuned on beverin via `benchmarks/tune_fp8_blockscale_gemm.py` over the
full CDNA3 config space (`BLOCK_M/N/K`, `num_warps/stages`, `matrix_instr_nonkdim`,
`waves_per_eu`, `kpack`). The key finding: **N**, not M, drives the choice. The
N=512 MLA projections starve a 304-CU GPU with big 128×256 tiles (only ~32
workgroups → 78 TFLOP/s); tiny 64×64 / 16×16-MFMA tiles give many more workgroups
→ **250 TFLOP/s (3.2×)**. The large-N (N=7168) prefill wants big 128×128 /
32×32-MFMA tiles to approach the ceiling.

| regime | tile | MFMA |
|---|---|---|
| `N ≤ 1024` (e.g. N=512) | `BM64 BN64 BK128`, warps 4, stages 2 | 16×16×32 fp8 |
| `N > 1024`, decode `M ≤ 16` | `BM16 BN128 BK128`, warps 4, stages 2 | 16×16×32 fp8 |
| `N > 1024`, prefill | `BM128 BN128 BK128`, warps 8, stages 2 | 32×32×16 fp8 |

`BLOCK_K = 128` (one full quant block) plus 2 stages fits the 64 KB CDNA3 LDS
because fp8 operands are half the bytes of #40's fp32 tiles — directly resolving
the `OutOfResources` that forced #40 down to 64³.

## Validation (beverin, AMD Instinct MI300A / gfx942, torch 2.11.0+rocm7.2)

`tests/test_mm_fp8_blockscale_mfma.py`: interpreter (CPU fp32, exact
block-promotion math vs the dequant oracle — fp8 `tl.dot` is exact under the
interpreter; fnuz dot is GPU-only as the CPU interpreter cannot lower it) and GPU
(tight parity, cross-check vs the #40 portable path, fnuz operands, bf16 out).
On-device `pytest`: **16 passed**. V4-shape parity (fnuz): **rel 2.0e-6 – 3.7e-6**.

## Performance — an honest positive result

`triton.testing.do_bench`, bf16 out, native fp8 MFMA (fnuz operands) vs the
`torch_mm_fp8_blockscale` reference and the #40 portable kernel:

| M | N | K | mfma (fnuz) | TFLOP/s | #40 portable | torch_ref | mfma/ref | mfma/portable |
|--:|--:|--:|------------:|--------:|-------------:|----------:|---------:|--------------:|
| 1    | 512  | 7168 | **0.053 ms** | — | 0.363 ms | 0.185 ms | **3.48×** | 6.8× |
| 8    | 512  | 7168 | **0.052 ms** | — | 0.361 ms | 0.179 ms | **3.43×** | 6.9× |
| 2048 | 512  | 7168 | **0.060 ms** | 250 | 0.619 ms | 0.550 ms | **9.13×** | 10.3× |
| 4096 | 7168 | 2048 | **0.335 ms** | 359 | 2.884 ms | 1.913 ms | **5.72×** | 8.6× |

The decode shapes (M=1/8) are latency-bound (the GEMM is tiny) yet still 3.4×
faster than `torch_ref`; the compute-bound shapes hit 250–359 TFLOP/s. The
`e4m3fn` operands fall back to an f16 MFMA (~0.45–0.59 ms), *slower* than the
portable path — so `path="auto"` routes fn operands to the portable kernel and
only fnuz operands to the mfma fast path ("fastest available").

## What ships

**Ships:** a native-fp8-MFMA `mm_fp8_blockscale` fast path that is, for the first
time on gfx942, **faster than the tuned-BLAS `torch_mm_fp8_blockscale`** on every
V4 shape (3.4–9.1×) — finally putting the MLA / gate / shared-expert projections
on a winning kernel for prefill **and** decode. `path="auto"` (default) selects it
for fnuz operands and keeps fn operands on the portable fallback; `path="mfma"` /
`"portable"` force either. The #40 portable kernel remains the always-correct,
format-agnostic fallback.

**Requires:** fnuz operands for the speedup (the AMD-native fp8 MFMA encoding). The
V4 AMD serving path should quantize the activations/weights to
`float8_e4m3fnuz` — the standard practice for native fp8 MFMA on MI300 — which the
quant helpers now support via `fp8_dtype`. Wiring the tokenspeed serving path to
the xkernels op (and to fnuz quantization) is a tokenspeed change, out of scope
here as in #33/#38.

**Headroom:** the N=512 shapes sustain 250 TFLOP/s vs 359 at large N; further
small-N tiling (or split-K) and a Gluon rewrite (explicit `amd_mfma` + LDS
double-buffer) are the natural next steps once this config space is in use.
