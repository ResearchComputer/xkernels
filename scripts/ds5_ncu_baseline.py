#!/usr/bin/env python
"""Profile the naive CUTE fp32 GEMM with ncu on ds5 GB10, and print the roofline
+ dominant-stall + achieved-BW numbers in the diagnose-skills' vocabulary.

This is the BASELINE (pre-fix) profile. ncu 2025.3.1 on ds5 recognizes GB10/sm_121.
We profile the post-warmup steady-state dispatch (--launch-skip 3 --launch-count 1).
"""
from __future__ import annotations
import torch
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute

torch.manual_seed(0)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(N, K, device="cuda", dtype=torch.float32)
for _ in range(3):               # warmup (JIT + caches)
    fp32_matmul_cute(a, b)
torch.cuda.synchronize()
o = fp32_matmul_cute(a, b)        # the profiled dispatch
torch.cuda.synchronize()
