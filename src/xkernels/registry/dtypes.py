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
