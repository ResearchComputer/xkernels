# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Minimal torch-backed tile helpers for ``@kernel`` bodies (Phase 1).

On CPU these are thin wrappers over torch — they let a body be written in a
kernel-flavored style (``next_pow2``, dtype casts) that mirrors the device
kernel's arithmetic, so the body is a *meaningful* reference rather than a copy
of the hand-written one. The per-program tiling / masking API that lowers to
Triton (``load(row, cols, mask=...)``, ``program_id``) is the GPU-gated path
(docs/brainstorm/04 Ex.1) and lands in Phase 1.5.
"""
from __future__ import annotations

import torch

# Dtype short-name constants (mirror registry/dtypes.py short names).
fp32 = "fp32"
bf16 = "bf16"
fp16 = "fp16"
int32 = "int32"  # index tensors (e.g. RoPE positions) — a data-ADDRESSING input
bool = "bool"  # attention masks (the where() mask operand)


def next_pow2(n: int) -> int:
    """Smallest power of two >= n (mirrors ``triton.next_power_of_2``)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def cast(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """Cast by short dtype name — the ``.to(fp32)`` of the tile API."""
    from ..registry.dtypes import to_torch_dtype

    return x.to(to_torch_dtype(dtype))
