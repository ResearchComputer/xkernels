#!/usr/bin/env python
"""ncu probe harness: run each target kernel several times so ncu can capture a
clean steady-state dispatch (warm-up fills the compile cache + L2)."""
import torch
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute
from xkernels.ops.moe.cute.entry import moe_sum_reduce_cute_entry

torch.manual_seed(0)

# GEMM
af = torch.randn(128, 512, device="cuda", dtype=torch.float32)
bf = torch.randn(512, 512, device="cuda", dtype=torch.float32)
for _ in range(50):
    fp32_matmul_cute(af, bf)
torch.cuda.synchronize()

# moe_sum_reduce
y = torch.randn(128, 8, 7168, device="cuda", dtype=torch.bfloat16)
w = torch.randn(128, 8, device="cuda", dtype=torch.float32)
for _ in range(50):
    moe_sum_reduce_cute_entry(y, w, 1.0)
torch.cuda.synchronize()
print("warmup done")
