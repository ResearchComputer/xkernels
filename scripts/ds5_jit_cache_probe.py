#!/usr/bin/env python
"""Is the 9.3ms per-call overhead a per-call RECOMPILE (DSL cache miss) or a
first-call-only JIT? And does the @cute.jit object expose a compile/launch API
to amortize? Time call #1 vs call #N, and introspect the jit object."""
from __future__ import annotations
import time, torch
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

torch.manual_seed(0)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
bT = b.t().contiguous(); out = torch.empty((M, N), device="cuda")
gA, gB, gOut = from_dlpack(a), from_dlpack(bT), from_dlpack(out)

def host_ms(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

print("call #1 (cold JIT):    %.2f ms" % host_ms(lambda: _fp32_matmul(gA, gB, gOut, M, N, K)))
print("call #2 (warm?):       %.2f ms" % host_ms(lambda: _fp32_matmul(gA, gB, gOut, M, N, K)))
print("call #3 (warm?):       %.2f ms" % host_ms(lambda: _fp32_matmul(gA, gB, gOut, M, N, K)))

# fresh tensors, SAME shape — does a shape-keyed cache hit?
a2 = torch.randn(M, K, device="cuda"); b2 = torch.randn(N, K, device="cuda")
bT2 = b2.t().contiguous(); out2 = torch.empty((M, N), device="cuda")
gA2, gB2, gOut2 = from_dlpack(a2), from_dlpack(bT2), from_dlpack(out2)
print("call #4 new tensors:   %.2f ms  (same shape; cache hit?)" % host_ms(lambda: _fp32_matmul(gA2, gB2, gOut2, M, N, K)))

print("\n@cute.jit object API:", [x for x in dir(_fp32_matmul) if not x.startswith("_") or x in ("__call__",)])
print("has compile/warmup/cache?:", [x for x in dir(_fp32_matmul) if any(k in x.lower() for k in ("compil","warmup","cache","launch","build"))])
