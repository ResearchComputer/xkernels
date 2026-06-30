# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) backend for xkernels.

NVIDIA-only. Imports `nvidia-cutlass-dsl` (the `cute` extra); if absent, this
package's import raises and the backend is simply not registered — the same
graceful-degradation pattern as `ops/ffn/cuda/__init__.py`. The CUTE DSL JITs
CUTLASS C++/MLIR via nvcc, so callers must export ``CUDA_HOME`` to a CUDA
toolkit before invoking any compiled kernel.
"""
from __future__ import annotations

try:  # pragma: no cover - exercised only where the DSL is installed
    import cutlass  # noqa: F401  (presence gates registration)
    import cutlass.cute  # noqa: F401
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "xkernels CUTE DSL backend requires `nvidia-cutlass-dsl` "
        "(install the `cute` extra: `uv pip install -e .[cute]`)."
    ) from _exc
