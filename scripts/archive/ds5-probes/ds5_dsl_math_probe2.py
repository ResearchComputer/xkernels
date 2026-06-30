#!/usr/bin/env python
"""Find the correct calling convention for MLIR math ops (rsqrt/sqrt/exp/abs)
from CUTE DSL Numeric values. Greps the installed package source for the
canonical usage, then tries the candidates on a tiny kernel.
"""
from __future__ import annotations
import os, re, glob
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm, math as mlir_math
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T
import torch

ROOT = os.path.dirname(cutlass.__file__)
print(f"grepping {ROOT} for math-op usage ...")
# find files that import/call these ops
pat = re.compile(r"(RsqrtOp|SqrtOp|ExpOp|AbsFOp|math\.(rsqrt|sqrt|exp|absf))\s*\(")
hits = {}
for fp in glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True):
    if "__pycache__" in fp:
        continue
    try:
        txt = open(fp, errors="ignore").read()
    except Exception:
        continue
    for mobj in pat.finditer(txt):
        ln = txt[:mobj.start()].count("\n") + 1
        line = txt.splitlines()[ln-1].strip()
        hits.setdefault(mobj.group(1), []).append(f"{os.path.relpath(fp, ROOT)}:{ln}: {line[:110]}")
for op, lst in hits.items():
    print(f"\n### {op} ({len(lst)} hits)")
    for h in lst[:4]:
        print(f"  {h}")

# Try the likely calling conventions on a real value.
@cute.kernel
def _k(gX: Tensor, gO: Tensor, mode: cutlass.Constexpr, n: cutlass.Constexpr):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32()); bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    i = bidx*128 + tidx
    if i < n:
        x = gX[(i,)]
        r = x
        if mode == 0:
            r = mlir_math.RsqrtOp(x).results[0]
        elif mode == 1:
            r = mlir_math.rsqrt(x)
        elif mode == 2:
            r = mlir_math.RsqrtOp(x).result
        gO[(i,)] = r

@cute.jit
def _h(X, O, mode: cutlass.Constexpr, n: cutlass.Constexpr):
    _k(X, O, mode, n).launch(grid=[(n+127)//128,1,1], block=[128,1,1])

x = torch.tensor([1.0,4.0,9.0,16.0], device="cuda", dtype=torch.float32)
ref = torch.rsqrt(x)
for name, mode in [("RsqrtOp(x).results[0]",0),("math.rsqrt(x)",1),("RsqrtOp(x).result",2)]:
    try:
        o = torch.empty_like(x); gX,gO = from_dlpack(x), from_dlpack(o)
        _h(gX,gO,mode,x.numel()); torch.cuda.synchronize()
        h=cute.compile(_h,gX,gO,mode,x.numel()); h(gX,gO)
        err=(o-ref).abs().max().item()
        print(f"  {name:28s}: err={err:.2e} {'OK' if err<1e-5 else 'MISMATCH'} val={o.tolist()}")
    except Exception as e:
        print(f"  {name:28s}: {type(e).__name__}: {str(e)[:100]}")
