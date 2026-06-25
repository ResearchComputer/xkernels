"""Correctness-test helpers.

Tolerances are defined ONCE per op in its Op Spec ``numerics`` block
(docs/library.md §5.1). ``assert_close`` consults the Op Spec when given an
``op_id``; otherwise it falls back to the legacy per-dtype presets so existing
kernel tests keep working unchanged.
"""
from __future__ import annotations

import torch

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
