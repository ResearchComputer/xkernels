# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for ``mxfp4_paged_gather`` (issue #27 / DeepSeek-V4 DSA
indexer on gfx942) — numerical oracle and default (CPU / no-Triton) backend.

The DSA indexer selects, per query, the top-k KV positions; this op gathers those
positions out of a **paged** (block-table indexed) mxfp4 KV cache and dequantizes
them to the attention compute dtype. Padded selection slots (sentinel ``< 0``)
produce a zero row, matching the CUDA-only ``indexer_mxfp4_paged_gather``.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register
from .mxfp4 import MXFP4_GROUP_SIZE, dequant_mxfp4

__all__ = ["mxfp4_paged_gather_ref"]


def mxfp4_paged_gather_ref(
    kv_packed: torch.Tensor,
    kv_scale: torch.Tensor,
    block_table: torch.Tensor,
    sel_pos: torch.Tensor,
    *,
    block_size: int,
    group_size: int = MXFP4_GROUP_SIZE,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather + dequantize selected mxfp4 KV positions from a paged cache.

    Args:
        kv_packed: ``[num_blocks, block_size, D // 2]`` uint8 mxfp4 nibbles.
        kv_scale: ``[num_blocks, block_size, D // group_size]`` uint8 E8M0 scales.
        block_table: ``[num_seqs, max_blocks]`` int32, logical-block -> physical
            block id.
        sel_pos: ``[num_seqs, topk]`` int32 selected *sequence* positions; entries
            ``< 0`` are padding and yield a zero output row.
        block_size: KV positions per physical block.
        group_size: mxfp4 shared-scale group length along ``D``.
        out_dtype: output dtype (bf16 in production).

    Returns:
        ``[num_seqs, topk, D]`` gathered + dequantized KV in ``out_dtype``.
    """
    num_seqs, topk = sel_pos.shape
    D = kv_packed.shape[-1] * 2
    out = torch.zeros(num_seqs, topk, D, device=kv_packed.device, dtype=out_dtype)
    for s in range(num_seqs):
        for j in range(topk):
            pos = int(sel_pos[s, j])
            if pos < 0:
                continue
            logical_blk = pos // block_size
            within = pos % block_size
            phys = int(block_table[s, logical_blk])
            deq = dequant_mxfp4(
                kv_packed[phys, within], kv_scale[phys, within], group_size
            )  # [D] fp32
            out[s, j] = deq.to(out_dtype)
    return out


register("mxfp4_paged_gather", Backend.REFERENCE)(mxfp4_paged_gather_ref)
