#!/usr/bin/env python
"""Introspect the cutlass.cute DSL API surface for tiled-GEMM primitives:
shared memory tensors, copy atoms (cp.async), tile partitioning, thread maps.
Print the actual symbols so we author against the real API, not from memory."""
from __future__ import annotations
import cutlass.cute as cute
import cutlass.cute.algorithm as algo
import cutlass.cute.atom as atom
import cutlass.cute.core as core
import cutlass.cute.tensor as tensor
import cutlass.cute.nvgpu as nvgpu

def dump(name, mod):
    print(f"\n=== {name} ===")
    for a in sorted(dir(mod)):
        if a.startswith("_"):
            continue
        v = getattr(mod, a, None)
        kind = type(v).__name__
        if kind in ("module",):
            continue
        print(f"  {a}  [{kind}]")

dump("cutlass.cute", cute)
dump("algorithm", algo)
dump("atom", atom)
print("\n=== core (tile/layout helpers) ===")
for a in sorted(dir(core)):
    if a.startswith("_"): continue
    v = getattr(core, a)
    if type(v).__name__ == "module": continue
    print(f"  {a}  [{type(v).__name__}]")
print("\n=== tensor ===")
for a in sorted(dir(tensor)):
    if a.startswith("_"): continue
    v = getattr(tensor, a)
    if type(v).__name__ == "module": continue
    print(f"  {a}  [{type(v).__name__}]")
print("\n=== nvgpu (submodules + cp_async?) ===")
for a in sorted(dir(nvgpu)):
    if a.startswith("_"): continue
    print(f"  {a}  [{type(getattr(nvgpu,a)).__name__}]")
