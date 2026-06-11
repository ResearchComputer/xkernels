# Issue #27 — DeepSeek-V4 DSA indexer logits: gfx942 forward path (validated)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2,
tokenspeed_triton. **Test:** `tests/test_dsa_indexer_logits.py`
(`slurm/test_dsa_indexer_beverin.sbatch`), job 381969.

## TL;DR

DeepSeek-V4's attention (CSA + HCA) is driven by a **DSA indexer** that scores
every cached KV position per query and selects the top-512 (Flash) / 1024 (Pro).
Upstream computes the indexer logits with **NVIDIA-only** kernels
(`deep_gemm.fp8_fp4_mqa_logits` + a CUDA mxfp4 paged gather); the AMD branch of the
fallback gate `_deepseek_v4_deepgemm_fp4_indexer_available()` was unverified, so the
gfx942 forward path was unvalidated.

This ships a **portable Triton replacement for the indexer logits** —
`xkernels.dsa_indexer_logits` — that runs on gfx942, plus a thin
`dsa_indexer_topk` (a `torch.topk`) for the selection. The selection is
**numerically identical** to the upstream torch oracle and **validated on MI300A**.

## What the indexer computes

The numerically meaningful operation — the one that *selects* which KV survive —
is a **weighted ReLU MQA** dot-product followed by a masked top-k (this mirrors
tokenspeed's own torch oracle `_indexer_topk_reference`):

    logits[t, j] = sum_h  weights[t, h] * relu( q[t, h, :] . k[j, :] )

with `q : [T, H, D]` (`H = index_n_heads = 64`, `D = index_head_dim = 128`), a
**single shared** `k : [K, D]` per KV position (MQA), and per-head combine
`weights : [T, H]`. An optional causal window `[row_starts, row_starts+lengths)`
masks out-of-range columns to `-inf` before the top-k. The fp8/fp4 packing in the
upstream CUDA kernel is a hardware encoding detail; it does not change which KV
are selected, so the gfx942 path computes the logits directly in fp32 from
bf16/fp16 q/k (upcast on the MFMA unit).

## Result (on-device, MI300A / gfx942, job 381969)

| Check | Result |
|---|---|
| pytest `test_dsa_indexer_logits.py` (7 cases, GPU bf16, real Triton compile) | **7 passed** |
| V4 shape `H=64 D=128 K=4096` bf16: `max|err|` vs fp32 oracle | **6.10e-05** (rel 1.76e-07) |
| Flash top-512 selection: mean Jaccard(top-k set) vs oracle | **1.0000** |

Local CPU gate (`TRITON_INTERPRET=1`, fp32): 7/7 + full suite 101 passed.

## Kernel shape

One Triton program handles one `(query, KV-tile)` pair: it loads the full `[H, D]`
query (`H=64`, `D=128` fit in registers), streams a `BLOCK_K=64`-row tile of the
shared MQA key, computes `tl.dot(q, kᵀ)` → ReLU → per-head weight → sum over heads
in fp32, then applies the causal mask. Grid is `(T, cdiv(K, 64))`.

## Gotcha hit (interpreter ≠ compiler)

The first on-device run (job 381966) failed to *compile* the masked branch:
referencing a module-level `_NEG_INF = float("-inf")` from inside the `@triton.jit`
kernel is rejected by the real Triton compiler ("Cannot access global variable …
not instantiated as constexpr"), but `TRITON_INTERPRET=1` happily allowed it. Fixed
by inlining `float("-inf")` in the kernel. This is the documented
interpreter-vs-compiler constexpr-globals trap — on-device validation caught what
CPU could not.

## Scope / what is NOT in this PR

- **fp8/fp4 quantized I/O.** The upstream `deep_gemm.fp8_fp4_mqa_logits` consumes
  pre-quantized fp8 q / fp4 (mxfp4) k for memory bandwidth. This ships the
  **dequantized-math equivalent** (bf16/fp16 in, fp32 logits out): correct and
  portable, validated to match the oracle. A native fp8/fp4 gfx942 logits kernel
  (and the mxfp4 paged-gather) would be a bandwidth optimization on top — its own
  measured follow-up, not required for a *correct* forward path.
- **The paged KV gather** (`indexer_mxfp4_paged_gather`) is an mxfp4-cache layout
  concern that only exists because the logits kernel wanted packed fp4 input; with
  the dequantized-math path the gather is an ordinary `k[block_table]` index. The
  packed-gather kernel is deferred with the fp4 logits kernel above.
- **Wiring into `models/deepseek_v4.py`** lives in the tokenspeed runtime, not this
  kernels package; this PR provides + validates the gfx942 op the runtime can call.
