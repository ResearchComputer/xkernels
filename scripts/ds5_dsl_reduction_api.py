#!/usr/bin/env python
"""Read the EXACT signatures/docstrings of the CUTE DSL reduction primitives a
row-reduction kernel (dual_rmsnorm / moe_sum_reduce) needs, straight from the
installed package source. Drives the kernel design — no guessing the API.
"""
from __future__ import annotations

import inspect

from cutlass.cute.core import reduce, ReductionOp
from cutlass.cute import ReductionKind
from cutlass.cute.arch import (
    warp_reduction, warp_reduction_sum, warp_reduction_max,
    warp_redux_sync, shuffle_sync_down, sync_threads, sync_warp,
    alloc_smem, get_dyn_smem,
)

TARGETS = [
    ("cute.core.reduce", reduce),
    ("cute.core.ReductionOp", ReductionOp),
    ("cute.ReductionKind", ReductionKind),
    ("cute.arch.warp_reduction", warp_reduction),
    ("cute.arch.warp_reduction_sum", warp_reduction_sum),
    ("cute.arch.warp_reduction_max", warp_reduction_max),
    ("cute.arch.warp_redux_sync", warp_redux_sync),
    ("cute.arch.shuffle_sync_down", shuffle_sync_down),
    ("cute.arch.sync_threads", sync_threads),
    ("cute.arch.sync_warp", sync_warp),
    ("cute.arch.alloc_smem", alloc_smem),
    ("cute.arch.get_dyn_smem", get_dyn_smem),
]

for name, obj in TARGETS:
    print("=" * 60)
    print(f"### {name}")
    print("=" * 60)
    print(f"type: {type(obj).__name__}")
    try:
        sig = inspect.signature(obj)
        print(f"signature: {name}{sig}")
    except (ValueError, TypeError) as e:
        print(f"signature: <no py signature: {e}>")
    doc = inspect.getdoc(obj)
    if doc:
        print(f"doc:\n{doc[:500]}")
    print()
