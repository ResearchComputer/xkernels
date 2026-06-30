#!/usr/bin/env python
"""Read CudaDialectJitCompiledFunction.run_compiled_program +
generate_execution_args to learn how to launch a compiled handle with NEW
tensors (the compile-once / launch-many path that kills the ~9.3ms overhead)."""
from __future__ import annotations
import sys, inspect, torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

def log(*a): print(*a); sys.stdout.flush()

M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
bT = b.t().contiguous(); out = torch.empty((M, N), device="cuda")
gA, gB, gOut = from_dlpack(a), from_dlpack(bT), from_dlpack(out)
_fp32_matmul(gA, gB, gOut, M, N, K); torch.cuda.synchronize()

handle = cute.compile(_fp32_matmul, gA, gB, gOut, M, N, K)
log("handle type:", type(handle).__name__)
log("module:", type(handle).__module__)
Cls = type(handle)
log("file:", inspect.getsourcefile(Cls))

for meth in ("run_compiled_program","generate_execution_args","__call__","execution_args"):
    fn = getattr(Cls, meth, None)
    log(f"\n=== {meth} ===")
    if fn is None: log("  (absent)"); continue
    try: log("  sig:", inspect.signature(fn))
    except Exception as e: log("  (no sig:", e, ")")
    try:
        src = inspect.getsource(fn)
        log(src[:1100])
        if len(src)>1100: log(f"  ...[{len(src)-1100} more]")
    except Exception as e: log("  (no source:", e, ")")
