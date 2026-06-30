#!/usr/bin/env python
"""Measure CUTE card latency (ms) for record_measurement, at the largest sweep point.
Also re-states the verify verdict for the card notes."""
from __future__ import annotations

import torch

from xkernels.ops.gemm.cute.entry import mm_fp8_blockscale_cute
from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(N, K, device="cuda", dtype=torch.float32)
a8, a_s = per_token_group_quant_fp8(a, block=128)
b8, b_s = per_block_quant_fp8(b, block=128)


def call():
    return mm_fp8_blockscale_cute(a8, a_s, b8, b_s, block=128, out_dtype=torch.bfloat16)


# warmup (JIT) then time
for _ in range(3):
    call()
ms = benchmark(call)
tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
print(f"M={M} N={N} K={K} bf16: ms={ms:.4f}  (tflops={tflops:.2f}, naive fp32-FMA, no matrix engine)")
