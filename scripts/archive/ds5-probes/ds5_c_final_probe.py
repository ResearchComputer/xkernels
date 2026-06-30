#!/usr/bin/env python
"""Final cheap checks on (c):
  [1] handle() WITHOUT re-passing constexpr M,N,K (handle already specialized).
  [2] inspect generate_execution_args output (are pointers sane?).
  [3] what does the working @cute.jit __call__ (_func) RETURN — a reusable handle?
Each step flushed + isolated so a segfault is localized."""
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

log("warmup @cute.jit"); _fp32_matmul(gA, gB, gOut, M, N, K); torch.cuda.synchronize()
handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)
log("handle:", type(handle).__name__)

log("\n[2] generate_execution_args inspection:")
try:
    ea, adapted = handle.generate_execution_args(gA, gB, gOut, M, N, K)
    log(f"  exe_args: len={len(ea)} types={[type(x).__name__ for x in ea][:8]}")
    log(f"  exe_args[:6] = {[hex(x) if isinstance(x,int) else x for x in ea][:6]}")
    log(f"  adapted: len={len(adapted)} types={[type(x).__name__ for x in adapted][:6]}")
except Exception as e:
    log("  generate_execution_args FAILED:", repr(e)[:160])

log("\n[1a] handle(gA,gB,gOut) no constexpr:")
try:
    handle(gA, gB, gOut); torch.cuda.synchronize(); log("  OK")
except Exception as e:
    log("  FAILED:", repr(e)[:140])

log("\n[1b] handle.run_compiled_program(exe_args) direct:")
try:
    ea, _ = handle.generate_execution_args(gA, gB, gOut, M, N, K)
    r = handle.run_compiled_program(ea); torch.cuda.synchronize(); log("  OK ret=", r)
except Exception as e:
    log("  FAILED:", repr(e)[:140])

log("\n[3] what does _func (@cute.jit call) RETURN?")
try:
    ret = _fp32_matmul(gA, gB, gOut, M, N, K)
    log(f"  _func returned: {type(ret).__name__} = {ret!r}"[:120])
    if ret is not None:
        log("  attrs:", [x for x in dir(ret) if not x.startswith("_")][:20])
except Exception as e:
    log("  _func FAILED:", repr(e)[:140])
log("DONE")
