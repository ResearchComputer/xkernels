#!/usr/bin/env python
"""Probe device-side transcendental intrinsics (rsqrt/sqrt/exp/max/log) the
normalize/attention cards need. Confirms the exact callable, not guessed.

Checks both the nvvm special-register/intrinsic dialect and the MLIR math
dialect; prints the first working rsqrt + exp + max on a tiny CUTE kernel.
"""
from __future__ import annotations
import torch
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

# What's available on the dialects?
print("=== nvvm attrs mentioning sqrt/rsqrt/exp/log ===")
print([n for n in dir(nvvm) if any(k in n.lower() for k in ("sqrt","rsqrt","exp","log","max"))])
try:
    from cutlass._mlir.dialects import math as mlir_math
    print("=== mlir math attrs ===")
    print([n for n in dir(mlir_math) if not n.startswith("_")][:40])
except Exception as e:
    print("mlir math dialect:", e)


@cute.kernel
def _trans_kernel(gX: Tensor, gOut: Tensor, op: cutlass.Constexpr, n: cutlass.Constexpr):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    i = bidx * 128 + tidx
    if i < n:
        x = gX[(i,)]
        if op == 0:      # rsqrt
            from cutlass._mlir.dialects import math as m
            r = m.rsqrt(x)
        elif op == 1:    # sqrt
            from cutlass._mlir.dialects import math as m
            r = m.sqrt(x)
        elif op == 2:    # exp
            from cutlass._mlir.dialects import math as m
            r = m.exp(x)
        elif op == 3:    # max(x, 0) via fmax intrinsic probe
            from cutlass._mlir.dialects import math as m
            r = m.maxf(x, cutlass.Float32(0.0))
        gOut[(i,)] = r


@cute.jit
def _trans(X, Out, op: cutlass.Constexpr, n: cutlass.Constexpr):
    _trans_kernel(X, Out, op, n).launch(
        grid=[(n + 127)//128, 1, 1], block=[128,1,1])


def run_op(x, op):
    out = torch.empty_like(x)
    gX, gO = from_dlpack(x), from_dlpack(out)
    _trans(gX, gO, op, x.numel()); torch.cuda.synchronize()
    h = cute.compile(_trans, gX, gO, op, x.numel()); h(gX, gO)
    return out


x = torch.tensor([1.0, 4.0, 9.0, 16.0, 0.5, 2.0], device="cuda", dtype=torch.float32)
for name, op, ref in [("rsqrt",0, torch.rsqrt(x)), ("sqrt",1, x.sqrt()),
                      ("exp",2, x.exp()), ("maxf(x,0)",3, x.clamp(min=0))]:
    try:
        got = run_op(x, op)
        err = (got-ref).abs().max().item()
        print(f"{name:10s}: got={got.tolist()[:3]}... err={err:.2e} {'OK' if err<1e-4 else 'MISMATCH'}")
    except Exception as e:
        print(f"{name:10s}: FAILED {type(e).__name__}: {str(e)[:120]}")
