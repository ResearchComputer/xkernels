# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton ``mxfp4_paged_gather`` for AMD MI300A (gfx942, CDNA3).

Portable Triton replacement for the CUDA-only ``indexer_mxfp4_paged_gather`` used
by the DeepSeek-V4 DSA indexer (issue #27). One program per ``(seq, topk-slot)``
output row: resolve the selected sequence position through the block table, load
the row's ``D // 2`` packed mxfp4 bytes, decode both E2M1 nibbles arithmetically,
apply the broadcast E8M0 group scale, and store ``D`` values in ``out_dtype``.
Padded slots (``sel_pos < 0``) store a zero row.

E2M1 decode (nibble ``n`` in ``0..15``): sign ``= n >> 3``; on the 3-bit
magnitude code ``c = n & 7`` with exponent ``e = (c >> 1) & 3`` and mantissa
``m = c & 1``::

    |x| = m * 0.5                     if c < 2   (subnormal: 0, 0.5)
    |x| = (1 + m * 0.5) * 2**(e - 1)  otherwise  (1, 1.5, 2, 3, 4, 6)

which is exact for all 8 magnitudes. The E8M0 scale byte ``b`` maps to
``2**(b - 127)``; the reserved NaN code ``0xFF`` maps to ``0`` (padded groups).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["mxfp4_paged_gather_triton", "paged_gather_kernel"]


@triton.jit
def _decode_e2m1(nib):
    """Decode a vector of E2M1 nibbles (int32 in 0..15) to fp32 values."""
    sign = (nib >> 3) & 1
    c = nib & 7
    e = (c >> 1) & 3
    m = c & 1
    mf = m.to(tl.float32)
    sub = mf * 0.5  # c < 2
    nrm = (1.0 + mf * 0.5) * tl.exp2((e - 1).to(tl.float32))  # c >= 2
    mag = tl.where(c < 2, sub, nrm)
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def paged_gather_kernel(
    kv_packed_ptr,  # uint8 [num_blocks, block_size, D // 2]
    kv_scale_ptr,  # uint8 [num_blocks, block_size, D // group]
    block_table_ptr,  # int32 [num_seqs, max_blocks]
    sel_pos_ptr,  # int32 [num_seqs, topk]
    out_ptr,  # out_dtype [num_seqs, topk, D]
    max_blocks,
    topk,
    BLOCK_SIZE: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HALF: tl.constexpr,  # D // 2  (power of two >= actual)
    NGROUP: tl.constexpr,  # D // GROUP_SIZE
):
    s = tl.program_id(0)
    j = tl.program_id(1)

    pos = tl.load(sel_pos_ptr + s * topk + j)
    out_row = out_ptr + (s * topk + j) * D
    half_cols = tl.arange(0, HALF)

    if pos < 0:
        # Padded slot -> zero the whole D row (store as even/odd halves).
        zero = tl.zeros([HALF], dtype=out_ptr.dtype.element_ty)
        tl.store(out_row + half_cols * 2, zero, mask=half_cols < HALF)
        tl.store(out_row + half_cols * 2 + 1, zero, mask=half_cols < HALF)
        return

    logical_blk = pos // BLOCK_SIZE
    within = pos % BLOCK_SIZE
    phys = tl.load(block_table_ptr + s * max_blocks + logical_blk)

    row_base = (phys * BLOCK_SIZE + within) * HALF
    bytes_ = tl.load(kv_packed_ptr + row_base + half_cols).to(tl.int32)
    lo = _decode_e2m1(bytes_ & 0xF)  # even index value
    hi = _decode_e2m1((bytes_ >> 4) & 0xF)  # odd index value

    # E8M0 group scale for each of the D columns; byte 0xFF -> 0.
    scale_base = (phys * BLOCK_SIZE + within) * NGROUP
    # even index 2*k belongs to group (2*k)//GROUP_SIZE; odd 2*k+1 same group.
    grp_lo = (half_cols * 2) // GROUP_SIZE
    grp_hi = (half_cols * 2 + 1) // GROUP_SIZE
    sb_lo = tl.load(kv_scale_ptr + scale_base + grp_lo).to(tl.int32)
    sb_hi = tl.load(kv_scale_ptr + scale_base + grp_hi).to(tl.int32)
    mul_lo = tl.where(sb_lo == 0xFF, 0.0, tl.exp2((sb_lo - 127).to(tl.float32)))
    mul_hi = tl.where(sb_hi == 0xFF, 0.0, tl.exp2((sb_hi - 127).to(tl.float32)))

    val_lo = (lo * mul_lo).to(out_ptr.dtype.element_ty)
    val_hi = (hi * mul_hi).to(out_ptr.dtype.element_ty)
    # Interleave back into D: even cols = lo, odd cols = hi.
    tl.store(out_row + half_cols * 2, val_lo, mask=half_cols < HALF)
    tl.store(out_row + half_cols * 2 + 1, val_hi, mask=half_cols < HALF)


def mxfp4_paged_gather_triton(
    kv_packed: torch.Tensor,
    kv_scale: torch.Tensor,
    block_table: torch.Tensor,
    sel_pos: torch.Tensor,
    *,
    block_size: int,
    group_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    kv_packed = kv_packed.contiguous()
    kv_scale = kv_scale.contiguous()
    block_table = block_table.contiguous().to(torch.int32)
    sel_pos = sel_pos.contiguous().to(torch.int32)

    num_seqs, topk = sel_pos.shape
    max_blocks = block_table.shape[1]
    half = kv_packed.shape[-1]
    D = half * 2
    ngroup = D // group_size
    assert D % group_size == 0

    out = torch.empty(num_seqs, topk, D, device=kv_packed.device, dtype=out_dtype)
    paged_gather_kernel[(num_seqs, topk)](
        kv_packed,
        kv_scale,
        block_table,
        sel_pos,
        out,
        max_blocks,
        topk,
        BLOCK_SIZE=block_size,
        D=D,
        GROUP_SIZE=group_size,
        HALF=half,
        NGROUP=ngroup,
    )
    return out


register("mxfp4_paged_gather", Backend.TRITON)(mxfp4_paged_gather_triton)
