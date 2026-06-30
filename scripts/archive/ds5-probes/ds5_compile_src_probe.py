#!/usr/bin/env python
"""Read the source of cute.compile (CompileCallable) and CuTeDSL's
compile_and_cache / compile_and_jit / jit / jit_runner, to find the
compile-once -> reusable handle -> fast-launch API."""
from __future__ import annotations
import inspect, cutlass.cute as cute
from cutlass.base_dsl.compiler import CompileCallable
from cutlass.cutlass_dsl.cutlass import CuTeDSL

def show(obj, name, n=1600):
    fn = getattr(obj, name, None)
    print(f"\n=== {type(obj).__name__}.{name} ===")
    if fn is None:
        print("  (none)"); return
    try:
        print("  file:", inspect.getsourcefile(fn))
        src = inspect.getsource(fn)
        print(src[:n])
        if len(src) > n: print(f"  ... [{len(src)-n} more chars]")
    except Exception as e:
        print("  (no source:", e, ")")

# cute.compile is a CompileCallable INSTANCE — how do you use it?
print("=== CompileCallable usage ===")
print("  cute.compile:", cute.compile, "|", type(cute.compile))
print("  class attrs:", [m for m in dir(CompileCallable) if not m.startswith("__") or m in ("__call__","__init__")])
show(CompileCallable, "__call__", 2000)
show(CompileCallable, "__init__", 1200)

# CuTeDSL compile/launch methods
show(CuTeDSL, "compile_and_cache", 2200)
show(CuTeDSL, "compile_and_jit", 1200)
show(CuTeDSL, "jit", 1400)
show(CuTeDSL, "jit_runner", 1800)
