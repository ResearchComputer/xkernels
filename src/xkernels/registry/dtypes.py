"""Dtype short-name <-> torch mapping, shared across the harness."""
from __future__ import annotations

import torch

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
    "fp8_e4m3fnuz": torch.float8_e4m3fnuz,
    "int32": torch.int32,
    "int8": torch.int8,
}


def to_torch_dtype(short: str | torch.dtype) -> torch.dtype:
    if isinstance(short, torch.dtype):
        return short
    key = short.lower()
    if key not in _DTYPE_MAP:
        raise KeyError(f"unknown dtype short name {short!r}; have {sorted(_DTYPE_MAP)}")
    return _DTYPE_MAP[key]


def to_short_dtype(dtype: torch.dtype) -> str:
    for short, dt in _DTYPE_MAP.items():
        if dt == dtype:
            return short
    raise KeyError(f"no short name for torch dtype {dtype}")


# Bytes per element for the dtype short names (for analytical byte-cost models).
# Computed via torch's element_size() on a zero-element tensor — guaranteed
# across torch versions (unlike ``dtype.itemsize``) and stays correct as new
# dtypes are added to _DTYPE_MAP (no parallel table to drift).
_DTYPE_BYTES: dict[str, int] = {
    short: torch.empty(0, dtype=dt).element_size()
    for short, dt in _DTYPE_MAP.items()
}


def dtype_bytes(short: str) -> int:
    """Bytes per element for a dtype short name (e.g. 'bf16' -> 2).

    Used by the analytical cost models (registry/cost_model.py) to compute
    memory traffic from a sweep point. Raises KeyError on an unknown name so a
    typo in a model fails loudly rather than silently undercounting bytes.
    """
    key = short.lower()
    if key not in _DTYPE_BYTES:
        raise KeyError(f"unknown dtype short name {short!r}; have {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]
