#!/usr/bin/env python
"""Find the public compile-once/run-many API in cutlass.cute, to eliminate the
~9.3ms per-call launch overhead of @cute.jit.__call__.

The torch-vendored wrapper (cutedsl/wrappers/dense_blockscaled_gemm_kernel.py)
uses a two-phase pattern: CuteDslKernel.compile() -> CompiledArtifact (one-time
JIT via self.cute_compile), then _run(compiled_artifact) per call (self.cute_run).
We need the equivalent PUBLIC entry point for a @cute.jit function.

Probes:
  1. cutlass.cute.compile / .jit attributes
  2. what @cute.jit __call__ does (compile each time? cache?)
  3. CuteDslKernel.cute_compile / cute_run source location
  4. the compiled-handle object's API
"""
from __future__ import annotations
import inspect, cutlass.cute as cute

print("=== cutlass.cute top-level callables (compile/run/jit) ===")
for a in sorted(dir(cute)):
    if a.startswith("_"): continue
    v = getattr(cute, a, None)
    t = type(v).__name__
    if t in ("function","builtin_function_or_method") or a in ("jit","compile","runtime"):
        sig = ""
        try: sig = str(inspect.signature(v))
        except Exception: pass
        print(f"  {a}{sig}  [{t}]")

print("\n=== cute.runtime (streams, compile?) ===")
import cutlass.cute.runtime as rt
for a in sorted(dir(rt)):
    if a.startswith("_"): continue
    v = getattr(rt, a, None)
    t = type(v).__name__
    if t in ("function","builtin_function_or_method","module"):
        print(f"  {a}  [{t}]")

print("\n=== @cute.jit object: what is __call__ / is there compile()? ===")
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import _fp32_matmul
print("  type:", type(_fp32_matmul).__name__, type(_fp32_matmul).__mro__[:3])
print("  attrs:", [x for x in dir(_fp32_matmul) if not x.startswith("__") or x=="__call__"])
# does the class (not instance) have compile?
for cls in type(_fp32_matmul).__mro__:
    if "compile" in dir(cls) or any("compil" in s.lower() for s in dir(cls)):
        print(f"  class {cls.__name__} has: {[s for s in dir(cls) if 'compil' in s.lower() or 'run' in s.lower() or 'launch' in s.lower()]}")
        break

print("\n=== CuteDslKernel.cute_compile / cute_run (where defined) ===")
try:
    import cutlass_api.providers.cutedsl.kernel as k
    src = inspect.getsource(k)
    import re
    for m in re.finditer(r"def (cute_compile|cute_run|compile|run)\b[^\n]*", src):
        print("  ", m.group(0)[:100])
    for name in ("cute_compile","cute_run"):
        if "def "+name in src:
            i = src.index("def "+name)
            print(f"\n--- src: {name} ---")
            print(src[i:i+400])
except Exception as e:
    print("  (cutlass_api not importable this way:)", e)
