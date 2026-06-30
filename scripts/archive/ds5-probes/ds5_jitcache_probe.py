#!/usr/bin/env python
"""Probe the in-memory jit_cache: after one @cute.jit call, what JitCompiledFunction
is cached, and does its engine launch fast (bypassing the per-call rebuild that
costs ~9ms)? Also localize the cute.compile segfault."""
from __future__ import annotations
import sys, torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

def log(*a):
    print(*a); sys.stdout.flush()

M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
bT = b.t().contiguous(); out = torch.empty((M, N), device="cuda")
gA, gB, gOut = from_dlpack(a), from_dlpack(bT), from_dlpack(out)

log("step1: warmup @cute.jit call")
_fp32_matmul(gA, gB, gOut, M, N, K)
torch.cuda.synchronize()

dsl = _fp32_matmul._dsl_cls   # CuTeDSL singleton (DSLSingletonMeta holds state on the class)
jc = getattr(dsl, "jit_cache", None)
log("step2: jit_cache =", type(jc).__name__, "| keys attr?", hasattr(jc,"__len__"))
try:
    n = len(jc); log(f"  jit_cache len = {n}")
    # what's inside?
    for k, v in (jc.items() if hasattr(jc,"items") else []):
        log(f"  key={k!r:.60}  val={type(v).__name__}  attrs={[x for x in dir(v) if not x.startswith('_')][:20]}")
        # is there an engine / run / launch on the cached fn?
        log(f"    engine/run/launch: ", [x for x in dir(v) if any(t in x.lower() for t in ("engine","run","launch","jit","cuda","kernel","call"))])
        break
except Exception as e:
    log("  jit_cache iterate failed:", repr(e)[:120])

log("step3: cute.compile — localizing segfault")
try:
    log("  calling cute.compile(...)")
    handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)
    log("  cute.compile RETURNED:", type(handle).__name__)
    log("  handle attrs:", [x for x in dir(handle) if not x.startswith("_")][:30])
except Exception as e:
    log("  cute.compile raised:", repr(e)[:200])
log("DONE")
