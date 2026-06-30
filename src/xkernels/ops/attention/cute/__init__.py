# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) backend for the attention op family on NVIDIA.

Imports ``nvidia-cutlass-dsl`` (the ``cute`` extra). If absent, this package's
import raises and the backend is not registered — the same graceful-degradation
pattern as ``ops/gemm/cute/__init__.py``.
"""
from __future__ import annotations

from . import entry  # noqa: F401  (registers the backend on import, NVIDIA-only)
