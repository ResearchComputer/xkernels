# Benchmark results — all kernels, both arches

Fresh run **2026-06-26**. Each cell = median of Triton `do_bench`, bf16 unless
noted, optimized backend vs the naive PyTorch a practitioner would write. Shapes
mirror the README "Performance" table (Kimi-K2.6 / V4 serving regime). See
[`01-methodology.md`](01-methodology.md) for harness detail and
[`04-gotchas.md`](04-gotchas.md) for the bristen ✗ rows.

## MI300A (beverin, gfx942) — `bench_all.py`

| Kernel | Shape | Naive PyTorch | Optimized | Speedup |
|---|---|---:|---:|---:|
| `mha_merge_state` | T=8192, H=128, D=128 | 2.447 ms | 0.784 ms | **3.12×** |
| `sparse_mla_attention` | T=8, H=128, D=512, topk=512 | 3.002 ms | 0.111 ms | **27.01×** |
| `mhc_prenorm_gemm` | T=8, K=16384, N=24, splits=16 | 2.647 ms | 0.013 ms | **204.55×** ⚠ |
| `mhc_pre` (+post) | T=8, hc_mult=4, hidden=4096 | 2.762 ms | 0.080 ms | **34.54×** |
| `dual_rmsnorm` | T=8192, d=(1536,512) | 0.238 ms | 0.054 ms | **4.40×** |
| `moe_sum_reduce` | M=8192, top_k=8, H=7168 | 3.126 ms | 0.373 ms | **8.37×** |
| `moe_align_block_size` | M=16384, top_k=8, E=48, block=16 | 55.454 ms | 1.644 ms | **33.73×** |
| `fused_ffn` | M=4096, 4096→11008 (fp16) | 5.425 ms | 5.285 ms | **1.03×** |
| `moe_int4_w4a16` | M=64, E=48, N=4096, K=7168, top_k=8 | 31.970 ms | 1.364 ms | **23.44×** |

⚠ `mhc_prenorm_gemm` at T=8 is launch-overhead-dominated (0.013 ms opt); treat
its speedup as "≫100×", not a precise figure (swings 123×–205× run to run).

Reproduces the checked-in README table within ~3% on every row.

## MI300A — `mm_fp8_blockscale` (gfx942-only, `bench_fp8_blockscale_gemm.py`)

Native fp8 MFMA (`e4m3fnuz`) vs the portable dequant path vs the torch
dequant-then-matmul reference, across V4 MLA shapes.

| M | N | K | mfma (ms) | TFLOP/s | portable (ms) | torch_ref (ms) | mfma/ref | mfma/portable |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 7168 | 0.050 | 0.1 | 0.102 | 0.177 | 3.56× | 2.06× |
| 8 | 512 | 7168 | 0.052 | 1.1 | 0.102 | 0.173 | 3.34× | 1.96× |
| 2048 | 512 | 7168 | 0.061 | 246.1 | 0.207 | 0.502 | 8.22× | 3.38× |
| 4096 | 7168 | 2048 | 0.331 | **363.6** | 0.966 | 1.957 | 5.92× | 2.92× |

The large-shape row hits **363.6 TFLOP/s** — deep in the fp8 MFMA regime
(MI300A fp8 peak ≈ 2× the ~130.7 TF/s BF16 peak). The tiny-M rows are
launch-overhead-bound (single-digit % of peak), as expected for decode.

## A100 (bristen, sm_80) — `bench_all.py`, per-kernel process isolation

| Kernel | Shape | Naive PyTorch | Optimized | Speedup | Note |
|---|---|---:|---:|---:|---|
| `mha_merge_state` | T=8192, H=128, D=128 | 5.171 ms | 1.046 ms | **4.94×** | ✓ |
| `sparse_mla_attention` | T=8, H=128, D=512, topk=512 | — | — | ✗ | `waves_per_eu` KeyError |
| `mhc_prenorm_gemm` | T=8, K=16384, N=24, splits=16 | — | — | ✗ | `waves_per_eu` KeyError |
| `mhc_pre` (+post) | T=8, hc_mult=4, hidden=4096 | — | — | ✗ | Triton SIGSEGV (rc=139) |
| `dual_rmsnorm` | T=8192, d=(1536,512) | 0.517 ms | 0.053 ms | **9.81×** | ✓ |
| `moe_sum_reduce` | M=8192, top_k=8, H=7168 | 6.674 ms | 0.651 ms | **10.26×** | ✓ |
| `moe_align_block_size` | M=16384, top_k=8, E=48, block=16 | 66.549 ms | 0.883 ms | **75.38×** | ✓ |
| `fused_ffn` | M=4096, 4096→11008 (fp16) | 4.744 ms | 4.288 ms | **1.11×** | ✓ |
| `moe_int4_w4a16` | M=64, E=48, N=4096, K=7168, top_k=8 | 57.481 ms | 2.225 ms | **25.83×** | ✓ |

`mm_fp8_blockscale` is **N/A on bristen** (sm_80 has no hardware fp8). 6/9 ops
run; the 3 failures are Triton-version/portability issues, **not** kernel
correctness bugs — see [`04-gotchas.md`](04-gotchas.md) §1 / §1b.

## Cross-arch read

For the 6 ops that run on both, the **A100 relative speedup is often higher**
than MI300A despite the A100 being the slower GPU in absolute ms — because the
naive-PyTorch baseline is also slower on A100, and several of these ops are
memory/launch-bound where the kernel's win over torch scales with how badly torch
does it. Absolute `optimized` ms is what to compare for "how good is the kernel
on this box":

| Kernel | opt ms MI300A | opt ms A100 | A100/MI300A (slower on A100) |
|---|---:|---:|---:|
| `mha_merge_state` | 0.784 | 1.046 | 1.33× |
| `dual_rmsnorm` | 0.054 | 0.053 | 1.0× (launch-bound, both ≈ same) |
| `moe_sum_reduce` | 0.373 | 0.651 | 1.75× |
| `moe_align_block_size` | 1.644 | 0.883 | **0.54× (A100 faster!)** |
| `fused_ffn` | 5.285 | 4.288 | 0.81× (A100 faster) |
| `moe_int4_w4a16` | 1.364 | 2.225 | 1.63× |

Notable: `moe_align_block_size` is **faster on A100** (0.88 vs 1.64 ms) — its
torch baseline is also far slower on A100 (66.5 vs 55.5 ms), so the kernel's
relative win is 75× on A100 vs 34× on MI300A. `fused_ffn` likewise leans A100
(cuDNN/cuBLAS fp16 matmul is very strong on Ampere). These are exactly the
"grade AMD perf against the AMD roofline, never against the NVIDIA card" cases
the AGENTS.md portability stance warns about.
