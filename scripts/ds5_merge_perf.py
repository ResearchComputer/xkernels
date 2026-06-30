#!/usr/bin/env python
"""Measure CUTE mha_merge_state latency (ms) at the largest sweep point."""
from __future__ import annotations
import torch
from xkernels.ops.attention.cute.entry import mha_merge_state_cute
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
T, H, D = 64, 128, 128
out_a = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
out_b = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
lse_a = torch.randn(T, H, device="cuda", dtype=torch.float32).abs()
lse_b = torch.randn(T, H, device="cuda", dtype=torch.float32).abs()

def call():
    return mha_merge_state_cute(out_a, lse_a, out_b, lse_b)

for _ in range(5):
    call()
ms = benchmark(call)
# bytes: read out_a,out_b [T,H,D] bf16(2B) + lse_a,lse_b [T,H] fp32(4B); write out + lse
bytes_in = 2*(T*H*D)*2 + 2*(T*H)*4
bytes_out = (T*H*D)*2 + (T*H)*4
total = bytes_in + bytes_out
gbps = total / (ms * 1e-3) / 1e9
print(f"mha_merge_state T={T} H={H} D={D} bf16: ms={ms:.4f}")
print(f"  bytes = {total/1e6:.2f} MB, achieved BW = {gbps:.1f} GB/s ({gbps/273*100:.1f}% of GB10 peak)")
