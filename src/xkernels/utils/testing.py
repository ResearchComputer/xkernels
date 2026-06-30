"""Correctness-test helpers.

Tolerances are defined ONCE per op in its Op Spec ``numerics`` block
(meta/docs/library.md §5.1). ``assert_close`` consults the Op Spec when given an
``op_id``; otherwise it falls back to the legacy per-dtype presets so existing
kernel tests keep working unchanged.

``gpu_device_or_skip`` is the single source of truth for the device a Triton /
optimized kernel should run on across the kernel tests: CPU under
``TRITON_INTERPRET=1`` (the interpreter), ``cuda`` when a GPU is present, and a
skip otherwise (a no-GPU box that is not in interpreter mode cannot compile /
launch the kernel).
"""
from __future__ import annotations

import os

import torch

__all__ = ["assert_close", "gpu_device_or_skip", "tolerance", "tolerance_for_op"]


def gpu_device_or_skip() -> str:
    """Return the device to run a kernel on, or skip the test if none is usable.

    * ``'cpu'`` when ``TRITON_INTERPRET=1`` is set (run kernels through the
      Triton CPU interpreter);
    * ``'cuda'`` when a CUDA/ROCm GPU is available;
    * otherwise the test is skipped — a no-GPU box outside interpreter mode
      cannot launch the compiled kernel (it would raise
      ``RuntimeError: 0 active drivers`` rather than testing anything).
    """
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    import pytest

    pytest.skip("no GPU and TRITON_INTERPRET!=1")

# Legacy per-dtype presets (used when no Op Spec is known / not yet seeded).
_TOL: dict[torch.dtype, dict[str, float]] = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-3, "atol": 1e-3},
    torch.bfloat16: {"rtol": 1.6e-2, "atol": 1e-2},
}


def tolerance(dtype: torch.dtype) -> dict[str, float]:
    """Return {'rtol', 'atol'} for a dtype (defaults to float32 tolerances)."""
    return _TOL.get(dtype, _TOL[torch.float32])


def tolerance_for_op(op_id: str, dtype: torch.dtype) -> dict[str, float]:
    """Return the Op Spec's (rtol, atol) for an op + dtype, falling back to presets."""
    try:
        from ..registry import get_spec
        from ..registry.dtypes import to_short_dtype

        numerics = get_spec(op_id).numerics
        rtol, atol = numerics.tolerance_for(to_short_dtype(dtype))
        return {"rtol": rtol, "atol": atol}
    except Exception:  # op not seeded in registry, or registry unavailable
        return tolerance(dtype)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, op_id: str | None = None) -> None:
    """torch.testing.assert_close with Op Spec tolerances (if ``op_id`` given)
    else per-dtype presets keyed off ``expected.dtype``."""
    if op_id is not None:
        tol = tolerance_for_op(op_id, expected.dtype)
    else:
        tol = tolerance(expected.dtype)
    torch.testing.assert_close(actual, expected, rtol=tol["rtol"], atol=tol["atol"])
