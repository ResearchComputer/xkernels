# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Shared mxfp4 (OCP MX, FP4 E2M1 + E8M0 block scale) helpers for the DSA-indexer
paged KV gather (issue #27 / DeepSeek-V4 on gfx942).

mxfp4 packs two FP4 E2M1 elements per ``uint8`` (low nibble = lower index) and
shares one E8M0 power-of-two scale across a contiguous group of ``group_size``
elements along the gathered (head_dim) axis.

* **E2M1** (1 sign, 2 exp, 1 mantissa). The 8 magnitudes are
  ``{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}``; bit 3 is the sign.
* **E8M0** scale: ``uint8`` biased exponent, value ``2**(byte - 127)``; ``0xFF``
  is the reserved NaN code (treated as ``0`` scale here — padded/unused groups).
"""

from __future__ import annotations

import torch

__all__ = [
    "E2M1_LUT",
    "MXFP4_GROUP_SIZE",
    "dequant_mxfp4",
    "make_mxfp4_kv",
]

MXFP4_GROUP_SIZE = 32

# Magnitude for each of the 8 unsigned E2M1 codes (index = abs nibble & 0x7).
_E2M1_ABS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# Full 16-entry signed lookup: index = nibble (0..15), bit 3 = sign.
E2M1_LUT: tuple[float, ...] = tuple(_E2M1_ABS) + tuple(-v for v in _E2M1_ABS)


def _e2m1_lut_tensor(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(E2M1_LUT, device=device, dtype=dtype)


def dequant_mxfp4(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int = MXFP4_GROUP_SIZE
) -> torch.Tensor:
    """Unpack + dequantize mxfp4 to fp32 along the last (head_dim) axis.

    Args:
        packed: ``[..., D // 2]`` uint8, two E2M1 nibbles per byte
            (low nibble = even index, high nibble = odd index).
        scale: ``[..., D // group_size]`` uint8 E8M0 block scales.
        group_size: shared-scale group length along ``D``.

    Returns:
        ``[..., D]`` fp32 dequantized values.
    """
    *lead, half = packed.shape
    D = half * 2
    lut = _e2m1_lut_tensor(packed.device)
    p = packed.to(torch.int64)
    lo = lut[(p & 0xF)]  # [..., D//2]
    hi = lut[((p >> 4) & 0xF)]
    vals = torch.stack((lo, hi), dim=-1).reshape(*lead, D)  # interleave even/odd
    # E8M0 -> fp32 multiplier; 0xFF (NaN code) -> 0.
    sb = scale.to(torch.int64)
    mult = torch.where(
        sb == 0xFF,
        torch.zeros_like(sb, dtype=torch.float32),
        torch.exp2((sb - 127).to(torch.float32)),
    )  # [..., D//group]
    mult = mult.repeat_interleave(group_size, dim=-1)  # [..., D]
    return vals * mult


def make_mxfp4_kv(
    num_blocks: int,
    block_size: int,
    head_dim: int,
    *,
    group_size: int = MXFP4_GROUP_SIZE,
    device="cuda",
    seed: int = 0,
):
    """Random valid mxfp4 paged KV cache + its exact fp32 dequantization.

    Returns ``(packed [B,blk,D//2] uint8, scale [B,blk,D//g] uint8,
    deq [B,blk,D] fp32)`` where ``deq`` is ``dequant_mxfp4(packed, scale)``.
    """
    assert head_dim % 2 == 0 and head_dim % group_size == 0
    g = torch.Generator(device=device).manual_seed(seed)
    packed = torch.randint(
        0, 256, (num_blocks, block_size, head_dim // 2),
        generator=g, device=device, dtype=torch.uint8,
    )
    # E8M0 in a sane range around 1.0 (2^-4 .. 2^3); avoid the 0xFF NaN code.
    scale = torch.randint(
        123, 131, (num_blocks, block_size, head_dim // group_size),
        generator=g, device=device, dtype=torch.uint8,
    )
    deq = dequant_mxfp4(packed, scale, group_size)
    return packed, scale, deq
