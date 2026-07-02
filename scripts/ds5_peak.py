#!/usr/bin/env python
"""Measure honest GB10 (sm_121) peaks: bf16 cuBLAS + fp32 cuBLAS, across sizes.

This calibrates the archdb entry: the matrix-engine ceiling is the peak cuBLAS
reaches at the largest sizes (the achievable roofline, not the marketing number).
"""
import time

import torch

print(f"GB10 / sm_{torch.cuda.get_device_capability()}")
print(f"torch {torch.__version__} | cuBLAS bf16/fp32 peak calibration\n")


def bench_mm(m, n, k, dtype):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    # warmup
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = a @ b
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    flops = 2 * m * n * k
    return ms, flops / (ms * 1e-3) / 1e12


print(f"{'size':<14} {'bf16 ms':>9} {'bf16 TF':>9}   {'fp32 ms':>9} {'fp32 TF':>9}")
bf16_peak = 0
fp32_peak = 0
for sz in [1024, 2048, 4096, 8192]:
    bms, bf16 = bench_mm(sz, sz, sz, torch.bfloat16)
    fms, fp32 = bench_mm(sz, sz, sz, torch.float32)
    bf16_peak = max(bf16_peak, bf16)
    fp32_peak = max(fp32_peak, fp32)
    print(f"{sz}x{sz}x{sz:<6} {bms:>9.3f} {bf16:>9.1f}   {fms:>9.3f} {fp32:>9.1f}")

print(
    f"\n=> honest achievable ceiling:  bf16 ~ {bf16_peak:.0f} TFLOPS  |  "
    f"fp32 ~ {fp32_peak:.0f} TFLOPS"
)
print("   (substrate fp32 scalar peak = 29.5 TFLOPS ; these are tensor-core numbers)")
