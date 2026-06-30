#!/usr/bin/env python
"""Measure CUTE moe_sum_reduce latency (ms) at the largest sweep point."""
from __future__ import annotations
import torch
from xkernels.ops.moe.cute.entry import moe_sum_reduce_cute_entry
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
M, top_k, H = 128, 8, 7168
y = torch.randn(M, top_k, H, device="cuda", dtype=torch.bfloat16)
w = torch.randn(M, top_k, device="cuda", dtype=torch.float32)

def call():
    return moe_sum_reduce_cute_entry(y, w, 1.0)

for _ in range(5):
    call()
ms = benchmark(call)
# bytes (bf16 y=2B, fp32 w=4B): read y[M,top_k,H] + w[M,top_k]; write out[M,H]
bytes_in = 2 * M * top_k * H + 4 * M * top_k
bytes_out = 2 * M * H
total = bytes_in + bytes_out
gbps = total / (ms * 1e-3) / 1e9
print(f"moe_sum_reduce M={M} top_k={top_k} H={H} bf16: ms={ms:.4f}")
print(f"  bytes = {total/1e6:.2f} MB, achieved BW = {gbps:.1f} GB/s ({gbps/273*100:.1f}% of GB10 peak)")
