# Issue #28 / #27 — mxfp4 paged KV gather for the DeepSeek-V4 DSA indexer (gfx942)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** ROCm 7.2 / torch 2.11
(host venv: torch 2.12 / triton 3.7 for the interpreter gate).
**On-device job:** `slurm/probe_mxfp4_gather_beverin.sbatch`, beverin job 381968.

## What this ships

The gfx942 **Triton** replacement for the CUDA-only `indexer_mxfp4_paged_gather`
that has **no Triton variant** (issue #27, one of the two blockers under the
DeepSeek-V4 tracking issue #28). DeepSeek-V4's DSA indexer selects the top-512
(Flash) / top-1024 (Pro) KV positions per query; this op gathers those positions
out of a **paged** (block-table indexed) **mxfp4** KV cache and dequantizes them
to bf16 for the attention compute.

Public op: `xkernels.mxfp4_paged_gather(kv_packed, kv_scale, block_table, sel_pos,
*, block_size, group_size=32, out_dtype=torch.bfloat16, backend="auto")` →
`[num_seqs, topk, head_dim]`. Padded selection slots (`sel_pos < 0`) yield a zero
row, matching the CUDA kernel.

## mxfp4 format (OCP MX)

* **E2M1** FP4 element, two per `uint8` (low nibble = even index). The 8 magnitudes
  `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` are decoded **arithmetically** in-kernel (no LUT
  load): with code `c = nib & 7`, exp `e = (c>>1)&3`, mantissa `m = c&1`,
  `|x| = m*0.5` for `c<2` else `(1 + m*0.5) * 2**(e-1)`.
* **E8M0** block scale, one `uint8` per `group_size=32` elements along head_dim:
  `2**(byte - 127)`; the reserved `0xFF` NaN code maps to `0`.

Because every FP4 code is represented exactly in fp32/bf16, the dequant is **exact**
— the only error is the final bf16 round, which here is `0.0000` against the oracle.

## Result (beverin job 381968)

GPU bf16 correctness: **6/6 tests pass** (Triton vs the torch paged-gather oracle).

Decode-shape timing (head_dim=128, block_size=64), one program per `(seq, slot)`:

| num_seqs | topk | triton (ms) | max\|err\| |
|---------:|-----:|------------:|-----------:|
| 16 | 512 | 0.0291 | 0.0000 |
| 32 | 512 | 0.0281 | 0.0000 |
| 64 | 512 | 0.0275 | 0.0000 |
| 64 | 1024 | 0.0423 | 0.0000 |

## Scope / what is NOT solved

This unblocks the **gather half** of issue #27. It does **not** provide the
`deep_gemm.fp8_fp4_mqa_logits` indexer-logits kernel (the other CUDA-only piece),
and it does not touch issue #26 (mxfp4 fused-MoE expert parallelism), the PRIMARY
blocker for fitting V4 on MI300A. A full V4 forward still requires #26 plus the
indexer-logits path; this kernel is a self-contained, validated building block
toward that bring-up.
