#!/usr/bin/env python
"""Discover the CUTE DSL's reduction/scan/sync API surface on ds5.

We already proved the elementwise + tiled-GEMM patterns (smoke_vecadd,
mm_fp8_blockscale). The next op family (dual_rmsnorm, moe_sum_reduce,
mha_merge_state) needs BLOCK-WIDE REDUCTIONS (sum-of-squares over a row,
sum over top-k). That is a different CUTE primitive than copy/FMA.

This probe walks the installed ``cutlass.cute`` package and prints every
public symbol whose name mentions reduce / scan / sum / max / atomic / smem /
barrier / sync / cp_async / warp — the building blocks a row-reduction kernel
needs. Output drives the go/no-go for the reduction-class CUTE cards.
"""
from __future__ import annotations

import importlib
import pkgutil

import cutlass
import cutlass.cute as cute

KEYS = (
    "reduce", "scan", "sum", "max", "min", "atomic",
    "smem", "shared", "barrier", "sync", "cp_async", "async",
    "warp", "shfl", "wmma", "reduction",
)


def walk(pkg_name: str, seen: set[str]):
    try:
        pkg = importlib.import_module(pkg_name)
    except Exception as e:
        return
    if pkg_name in seen:
        return
    seen.add(pkg_name)
    names = [n for n in dir(pkg) if not n.startswith("_")]
    hits = [n for n in names if any(k in n.lower() for k in KEYS)]
    if hits:
        print(f"  {pkg_name}: {', '.join(sorted(hits))}")
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
            walk(m.name, seen)


print(f"cutlass {getattr(cutlass, '__version__', '?')} at {cutlass.__file__}")
print("=== reduction/smem/sync symbol surface (cutlass.cute.*) ===")
walk("cutlass.cute", set())

# Also check a few targeted submodules the GEMM examples import.
print()
print("=== targeted submodule attrs ===")
for mod in (
    "cutlass.cute",
    "cutlass.cute.algorithm",
    "cutlass.cute.nvgpu",
    "cutlass.cute.atom",
    "cutlass.cute.core",
):
    try:
        m = importlib.import_module(mod)
        names = [n for n in dir(m) if not n.startswith("_")]
        rdx = [n for n in names if any(k in n.lower() for k in KEYS)]
        print(f"  {mod}: {rdx if rdx else '(none of reduce/smem/sync)'}")
    except Exception as e:
        print(f"  {mod}: IMPORT FAILED {e}")
