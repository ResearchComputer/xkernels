#!/usr/bin/env python
"""Isolate the per-call host overhead in the CUTE DSL path. ncu says the GPU
dispatch is 87us, but do_bench says 9.5ms/call. Find which host step costs ~9ms:
  (a) from_dlpack (tensor wrap),
  (b) _fp32_matmul JIT cache miss (recompile each call?),
  (c) .launch() host machinery.
Uses raw CUDA events (NOT triton do_bench) to rule that out."""
from __future__ import annotations
import torch
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul, fp32_matmul_cute

torch.manual_seed(0)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(N, K, device="cuda", dtype=torch.float32)
bT = b.t().contiguous()
out = torch.empty((M, N), device="cuda", dtype=torch.float32)

def ev(): return torch.cuda.Event(enable_timing=True)
def time(fn, n=30, warm=5, label=""):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = ev(), ev()
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    print(f"  {label:42s} {s.elapsed_time(e)/n:.4f} ms/call")

print("CUDA-event timing (forced, no do_bench):")
# (1) full helper (from_dlpack inside, each call)
time(lambda: fp32_matmul_cute(a, b), label="(1) fp32_matmul_cute [from_dlpack x3 + launch]")
# (2) from_dlpack only
time(lambda: (from_dlpack(a), from_dlpack(bT), from_dlpack(out)), label="(2) from_dlpack x3 (tensor wrap only)")
# (3) pre-made g tensors, call _fp32_matmul repeatedly (cache hit?)
gA, gB, gOut = from_dlpack(a), from_dlpack(bT), from_dlpack(out)
time(lambda: _fp32_matmul(gA, gB, gOut, M, N, K), label="(3) _fp32_matmul SAME g-tensors (cache hit?)")
# (4) torch reference for scale
time(lambda: a @ b.t(), label="(4) torch a@b.t() (reference)")
