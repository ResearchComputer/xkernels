#!/usr/bin/env python
"""CONFIRM (c) solved: cute.compile handle launches FAST + CORRECT with NEW tensors
(no constexpr re-passed). Compare to 9.3ms baseline."""
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
def bench(fn, n=100, warm=20, label=""):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = ev(), ev(); s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    log(f"  {label:48s} {s.elapsed_time(e)/n*1000:.4f} us/call")

log("warmup + compile handle (once)")
_fp32_matmul(gA, gB, gOut, M, N, K); torch.cuda.synchronize()
handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)

# correctness with SAME tensors (no constexpr)
out.zero_(); handle(gA, gB, gOut); torch.cuda.synchronize()
ref = (a @ b.t())
log("[correctness SAME] max_abs vs torch:", (out-ref).abs().max().item())

# NEW tensors, same shape — the real test
a2 = torch.randn(M,K,device="cuda"); b2=torch.randn(N,K,device="cuda")
bT2=b2.t().contiguous(); out2=torch.empty((M,N),device="cuda")
gA2,gB2,gOut2 = from_dlpack(a2),from_dlpack(bT2),from_dlpack(out2)
handle(gA2, gB2, gOut2); torch.cuda.synchronize()
ref2 = (a2 @ b2.t())
log("[correctness NEW] max_abs vs torch:", (out2-ref2).abs().max().item())

log("\n[perf] (handle launches = no constexpr):")
bench(lambda: _fp32_matmul(gA, gB, gOut, M, N, K), label="(0) baseline @cute.jit")
bench(lambda: handle(gA2, gB2, gOut2), label="(1) handle() NEW tensors, no constexpr")
log("DONE")
