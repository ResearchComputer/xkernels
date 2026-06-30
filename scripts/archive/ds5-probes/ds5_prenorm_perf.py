#!/usr/bin/env python
"""Measure CUTE hc_prenorm_gemm latency (ms) at the largest sweep point."""
from __future__ import annotations
import torch
from xkernels.ops.mhc.cute.entry import hc_prenorm_gemm_cute
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
# Largest sweep point: bf16, T=37, K=128, N=16. Also probe a bigger one.
for T, K, N, dt in [(37, 128, 16, "bf16"), (64, 256, 24, "bf16")]:
    a = torch.randn(T, K, device="cuda", dtype=torch.bfloat16 if dt == "bf16" else torch.float32)
    fn = torch.randn(N, K, device="cuda", dtype=torch.float32)
    def call():
        return hc_prenorm_gemm_cute(a, fn, n_splits=1)
    for _ in range(5):
        call()
    ms = benchmark(call)
    # bytes: read a[T,K] (bf16=2B/fp32=4B) + fn[N,K] fp32(4B); write mul[1,T,N]+sqr[1,T] fp32
    ab = 2 if dt == "bf16" else 4
    bytes_in = ab * T * K + 4 * N * K
    bytes_out = 4 * T * N + 4 * T
    total = bytes_in + bytes_out
    gbps = total / (ms * 1e-3) / 1e9
    print(f"hc_prenorm_gemm T={T} K={K} N={N} {dt}: ms={ms:.4f} BW={gbps:.1f} GB/s ({gbps/273*100:.1f}% peak)")
