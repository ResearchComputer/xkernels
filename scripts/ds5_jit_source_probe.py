#!/usr/bin/env python
"""Read the @cute.jit implementation + find the compile cache. The jit object is
a plain `function` with _dsl_cls; calling it costs ~9.3ms/call steady-state (GPU
dispatch is 90us). Goal: find compile-once-then-launch-fast."""
from __future__ import annotations
import inspect, re, cutlass.cute as cute
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul

print("=== cute.jit / cute.compile (filtered) ===")
for a in ("jit", "compile", "program", "kernel"):
    v = getattr(cute, a, None)
    print(f"  cute.{a} = {v!r}  [{type(v).__name__}]")
    if v is not None and type(v).__name__ in ("function","_jit","_compile"):
        try:
            print("     src file:", inspect.getsourcefile(v))
            print("     --- source ---")
            print(inspect.getsource(v)[:1400])
        except Exception as e:
            print("     (no source:", e, ")")

print("\n=== _dsl_cls on the jit object (the real DSL class) ===")
cls = _fp32_matmul._dsl_cls
print("  _dsl_cls =", cls, "| MRO:", [c.__name__ for c in type(cls).__mro__][:4] if not isinstance(cls,type) else [c.__name__ for c in cls.__mro__][:4])
print("  _dsl_cls type:", type(cls).__name__)
# is it a class instance or a class? show relevant attrs
import inspect as I
members = [m for m in dir(cls) if not m.startswith("__") or m in ("__call__","__init__")]
print("  attrs:", [m for m in members if any(k in m.lower() for k in ("compil","run","launch","cache","call","build","jit"))])

print("\n=== where is jit defined? grep the package ===")
import cutlass, os
root = os.path.dirname(cutlass.__file__)
print("  package root:", root)
