#!/usr/bin/env python
"""Measure CUTE dual_rmsnorm latency (ms) at the largest sweep point, for
record_measurement. Also prints the verify verdict context. dual_rmsnorm is a
memory-bound normalize: theoretical BW = bytes read (2*x + 2*w) + bytes written
(2*out), over T rows.
"""
from __future__ import annotations

import torch

from xkernels.ops.norm.cute.entry import dual_rmsnorm_cute
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
# largest sweep point: T=64, d1=1536, d2=512, bf16
T, d1, d2 = 64, 1536, 512
x1 = torch.randn(T, d1, device="cuda", dtype=torch.bfloat16)
w1 = torch.randn(d1, device="cuda", dtype=torch.bfloat16)
x2 = torch.randn(T, d2, device="cuda", dtype=torch.bfloat16)
w2 = torch.randn(d2, device="cuda", dtype=torch.bfloat16)


def call():
    return dual_rmsnorm_cute(x1, w1, x2, w2, eps=1e-6)


for _ in range(5):  # warmup (JIT + compile-cache handles)
    call()
ms = benchmark(call)

# bytes moved (bf16 = 2 bytes): read x1,x2,w1,w2; write out1,out2
bytes_in = 2 * (T * d1 + d1) + 2 * (T * d2 + d2)
bytes_out = 2 * (T * d1 + T * d2)
total_bytes = bytes_in + bytes_out + bytes_in  # x re-read in pass 2 (2-pass design)
gbps = total_bytes / (ms * 1e-3) / 1e9
# GB10 unified-memory peak BW ~273 GB/s; achieved pct vs that roofline
achieved_bw_pct = gbps / 273.0 * 100.0
print(f"dual_rmsnorm T={T} d1={d1} d2={d2} bf16: ms={ms:.4f}")
print(f"  bytes moved (incl 2-pass x re-read) = {total_bytes/1e6:.2f} MB")
print(f"  achieved BW = {gbps:.1f} GB/s  ({achieved_bw_pct:.1f}% of GB10 ~273 GB/s peak)")
