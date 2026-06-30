# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) backend for the GEMM op family on NVIDIA.

Imports ``nvidia-cutlass-dsl`` (the ``cute`` extra). If absent, this package's
import raises and the backend is not registered — the same graceful-degradation
pattern as ``ops/ffn/cuda/__init__.py``. The CUTE DSL JITs via nvcc, so callers
must export ``CUDA_HOME`` to a CUDA toolkit before invoking compiled kernels.
"""
from __future__ import annotations

from . import entry  # noqa: F401  (registers the backend on import, NVIDIA-only)
