# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public ``mxfp4_paged_gather`` op: dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)
from .mxfp4 import MXFP4_GROUP_SIZE


def mxfp4_paged_gather(
    kv_packed: torch.Tensor,
    kv_scale: torch.Tensor,
    block_table: torch.Tensor,
    sel_pos: torch.Tensor,
    *,
    block_size: int,
    group_size: int = MXFP4_GROUP_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Gather + dequantize DSA-selected mxfp4 KV positions from a paged cache.

    The DeepSeek-V4 DSA indexer picks the top-k KV positions per query; this op
    collects them from a block-table-indexed mxfp4 cache and dequantizes to
    ``out_dtype``. Padded slots (``sel_pos < 0``) yield a zero row. This is the
    gfx942 (Triton) replacement for the CUDA-only ``indexer_mxfp4_paged_gather``
    that has no Triton variant (issue #27).

    Args:
        kv_packed: ``[num_blocks, block_size, D // 2]`` uint8 mxfp4 nibbles.
        kv_scale: ``[num_blocks, block_size, D // group_size]`` uint8 E8M0 scales.
        block_table: ``[num_seqs, max_blocks]`` int32 logical->physical block map.
        sel_pos: ``[num_seqs, topk]`` int32 selected positions (``<0`` = padding).
        block_size: KV positions per physical block.
        group_size: mxfp4 shared-scale group length along ``D``.
        out_dtype: output dtype (bf16 in production).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[num_seqs, topk, D]`` gathered + dequantized KV in ``out_dtype``.
    """
    return dispatch(
        "mxfp4_paged_gather",
        kv_packed,
        kv_scale,
        block_table,
        sel_pos,
        block_size=block_size,
        group_size=group_size,
        out_dtype=out_dtype,
        backend=backend,
    )
