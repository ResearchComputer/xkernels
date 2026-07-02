# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The lowering subpackage: math IR -> backend kernels.

Phase 2.0b lowered both ``tiled_2d`` (bare GEMM) and ``rowwise`` (dual_rmsnorm)
from the SAME doc-10 math IR in ``mathbody.py``; the bespoke ``rowreduce.py`` is
retired. ``triton.py`` is the dispatch entry point (``lower_to_triton`` /
``register_dsl``); ``mathbody.py`` holds the torch evaluator (auto-reference) and
the per-pattern Triton codegen. CUDA/HIP native overrides are Phase 2.1.
"""

from . import mathbody, triton

__all__ = ["mathbody", "triton"]
