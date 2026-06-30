#!/usr/bin/env python
"""Empirical: can cute.compile() give a reusable compiled handle that launches
fast (~us), eliminating the ~9.3ms/call @cute.jit overhead?

Tries (a) cute.compile(fn, *args) -> handle, call with NEW tensors, time it.
Also (b) inspects the returned executor, (c) checks for env-var cache dirs."""
from __future__ import annotations
import time, os, torch
import cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

torch.manual_seed(0)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
bT = b.t().contiguous(); out = torch.empty((M, N), device="cuda")
gA, gB, gOut = from_dlpack(a), from_dlpack(bT), from_dlpack(out)

def ev(): return torch.cuda.Event(enable_timing=True)
def bench(fn, n=50, warm=10, label=""):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = ev(), ev(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    print(f"  {label:46s} {s.elapsed_time(e)/n:.4f} ms/call")

# baseline: @cute.jit __call__
bench(lambda: _fp32_matmul(gA, gB, gOut, M, N, K), label="(0) baseline @cute.jit call")

print("\n--- (a) cute.compile(fn, *args) -> reusable handle? ---")
try:
    handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)
    print("  type(handle):", type(handle).__name__)
    print("  handle attrs:", [x for x in dir(handle) if not x.startswith("__") or x=="__call__"][:25])
    # call with the SAME tensors first (sanity)
    try:
        r = handle(gA, gB, gOut, M, N, K); print("  handle(same args) ok")
    except Exception as ex:
        print("  handle(same args) failed:", repr(ex)[:120])
        # maybe it's called differently
        for m in ("run","launch","__call__"):
            try:
                getattr(handle, m)(gA, gB, gOut, M, N, K); print(f"  handle.{m}(...) ok"); break
            except Exception as e2: print(f"  handle.{m} failed: {repr(e2)[:90]}")
    # NEW tensors, same shape — the real test
    a2 = torch.randn(M,K,device="cuda"); b2=torch.randn(N,K,device="cuda")
    bT2=b2.t().contiguous(); out2=torch.empty((M,N),device="cuda")
    gA2,gB2,gOut2 = from_dlpack(a2),from_dlpack(bT2),from_dlpack(out2)
    try:
        bench(lambda: handle(gA2, gB2, gOut2, M, N, K), label="(a) handle() NEW tensors same shape")
    except Exception as ex:
        print("  handle(new tensors) failed:", repr(ex)[:140])
except Exception as ex:
    print("  cute.compile FAILED:", repr(ex)[:200])

print("\n--- (c) env-var / file cache knobs ---")
envkeys = [k for k in os.environ if "CUTL" in k.upper() or "CUTE" in k.upper()]
print("  cutlass env vars set:", envkeys or "(none)")
dsl = _fp32_matmul._dsl_object
print("  envar attrs:", [x for x in dir(dsl.envar) if not x.startswith("_")])
for k in ("cache_dir","jit_cache","enable_cache","cache","jit_time_profiling","num_kernels"):
    if hasattr(dsl, k) or hasattr(dsl.envar, k):
        v = getattr(dsl, k, getattr(dsl.envar, k, None))
        print(f"    .{k} = {v!r}")
print("  cache_hits/misses:", getattr(dsl,"cache_hits",None), getattr(dsl,"cache_misses",None))
