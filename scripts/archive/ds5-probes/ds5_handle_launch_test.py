#!/usr/bin/env python
"""Definitive test: is cute.compile() handle launch FAST (~us) vs @cute.jit
(~9.3ms)? Step-localized, flush-per-step, to pin the segfault location.
Tests: same-tensors (proves path works), new-tensors (real use case), no-constexpr."""
from __future__ import annotations
import sys, torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

def log(*a): print(*a); sys.stdout.flush()

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
    log(f"  {label:44s} {s.elapsed_time(e)/n:.5f} ms/call")

log("warmup @cute.jit"); _fp32_matmul(gA, gB, gOut, M, N, K); torch.cuda.synchronize()
log("compiling handle..."); handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)
log("  ->", type(handle).__name__)

log("\n[1] baseline @cute.jit call:")
bench(lambda: _fp32_matmul(gA, gB, gOut, M, N, K), label="@cute.jit (SAME tensors)")

log("\n[2] handle SAME tensors (the key test):")
try:
    handle(gA, gB, gOut, M, N, K); torch.cuda.synchronize(); log("  sanity call OK")
    # correctness check vs torch
    ref = (a @ b.t())
    bench(lambda: handle(gA, gB, gOut, M, N, K), label="handle() SAME tensors")
    log("  max_abs vs torch ref:", (out-ref).abs().max().item())
except Exception as e:
    log("  handle(same) FAILED:", repr(e)[:160])

log("\n[3] handle NEW tensors same shape (real use case):")
a2 = torch.randn(M,K,device="cuda"); b2=torch.randn(N,K,device="cuda")
bT2=b2.t().contiguous(); out2=torch.empty((M,N),device="cuda")
gA2,gB2,gOut2 = from_dlpack(a2),from_dlpack(bT2),from_dlpack(out2)
try:
    handle(gA2, gB2, gOut2, M, N, K); torch.cuda.synchronize(); log("  sanity call OK")
    bench(lambda: handle(gA2, gB2, gOut2, M, N, K), label="handle() NEW tensors")
except Exception as e:
    log("  handle(new) FAILED:", repr(e)[:160])
log("DONE")
