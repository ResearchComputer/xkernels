# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Decode entry point ``flash_mla_with_kvcache`` for sparse-MLA on gfx942 (#32).

Gathers each query's DSA-selected positions from the paged fp8_ds_mla primary
cache (SWA) and the optional compressed (CSA) ``extra_k_cache``, dequantizes to
the compute dtype, flattens to the shared ``(kv, indices)`` form, and runs the
sparse-MLA compute kernel. The two index sets share one softmax — the hybrid
attention is realized here as a *union* of selections, not separate passes.

Cache contract (xkernels-clean, unit-testable): paged caches are passed as a
**value tensor** ``[num_blocks, block_size, value_bytes]`` uint8 plus a **scale
tensor** ``[num_blocks, block_size, scale_bytes]`` uint8 (the on-device adapter
that splits tokenspeed's single 2D pool buffer into these views is pinned by the
beverin validation job). Selected positions are resolved through ``block_table``.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from .sparse_mla import dequant_fp8_ds_mla


def _gather_dequant(value_cache, scale_cache, block_table, indices, lengths, block_size):
    """Gather + dequant selected positions to ``([T, topk, D] fp32, valid [T, topk])``.

    ``value_cache``/``scale_cache``: ``[num_blocks, block_size, *bytes]`` uint8.
    ``indices`` ``[T, topk]`` are positions into the cache; ``<0`` or
    ``>= lengths`` mark padding. When ``block_table`` is given, the indices are
    per-seq **logical** positions resolved through ``block_table``
    ``[T, max_blocks]``. When ``block_table is None`` the indices are **physical**
    token positions into the flattened ``[num_blocks * block_size]`` token space
    (the tokenspeed DeepSeek-V4 caller resolves logical->physical itself and
    passes ``block_table=None``).
    """
    if scale_cache is None:
        raise ValueError("fp8_ds_mla decode requires a scale_cache")
    T, topk = indices.shape
    dev = value_cache.device
    pos = torch.arange(topk, device=dev)
    valid = indices >= 0
    if lengths is not None:
        valid = valid & (pos.unsqueeze(0) < lengths.unsqueeze(1))
    safe = indices.clamp_min(0).long()
    within = safe % block_size
    if block_table is None:
        # Physical token positions: block = pos // block_size (no block_table gather).
        blk = safe // block_size
    else:
        logical_blk = safe // block_size
        blk = torch.gather(block_table.long(), 1, logical_blk)  # [T, topk]
    flat_blk = blk.reshape(-1)
    flat_within = within.reshape(-1)
    # Direct 2-axis gather of T*topk rows — works on non-contiguous as_strided
    # cache views without materializing the whole cache.
    vsel = value_cache[flat_blk, flat_within]  # [T*topk, value_bytes]
    ssel = scale_cache[flat_blk, flat_within]  # [T*topk, scale_bytes]
    deq = dequant_fp8_ds_mla(vsel, ssel)  # [T*topk, D]
    return deq.reshape(T, topk, deq.shape[-1]), valid


def flash_mla_with_kvcache(
    q,
    k_cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    tile_scheduler_metadata,
    *,
    softmax_scale,
    is_fp8_kvcache=True,
    indices,
    attn_sink=None,
    extra_k_cache=None,
    extra_indices_in_kvcache=None,
    topk_length=None,
    extra_topk_length=None,
    scale_cache=None,
    extra_scale_cache=None,
    block_size=None,
    backend: Backend | str = "auto",
):
    """Decode sparse-MLA over paged fp8_ds_mla cache(s). Returns ``(out, lse)``.

    Args (the V4-bound subset; see module docstring for the cache contract):
        q: ``[B, 1, H, D]`` latent queries (seq_q=1 at decode).
        k_cache / scale_cache: primary (SWA) paged value/scale caches.
        block_table: ``[B, max_blocks]`` logical->physical block map.
        head_dim_v: value/output dim (first ``head_dim_v`` latent dims).
        indices: ``[B, 1, topk]`` selected positions into the primary cache.
        extra_k_cache / extra_scale_cache / extra_indices_in_kvcache: optional
            compressed (CSA) cache + its selection, unioned into one softmax.
        topk_length / extra_topk_length: optional ``[B]`` valid counts.
        attn_sink: optional ``[H]`` per-head sink logit.

    Returns:
        ``(out [B, 1, H, head_dim_v], lse [B, H] fp32)``.
    """
    from .interface import sparse_mla_attention  # local import (avoid cycle)

    q2 = q.squeeze(1) if q.dim() == 4 else q  # [T, H, D]
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    T, H, D = q2.shape
    dev = q2.device
    if block_size is None:
        block_size = k_cache.shape[1]

    kvs, valids = [], []
    kv1, valid1 = _gather_dequant(k_cache, scale_cache, block_table, idx, topk_length, block_size)
    kvs.append(kv1)
    valids.append(valid1)

    if extra_k_cache is not None:
        eidx = (
            extra_indices_in_kvcache.squeeze(1)
            if extra_indices_in_kvcache.dim() == 3
            else extra_indices_in_kvcache
        )
        kv2, valid2 = _gather_dequant(
            extra_k_cache,
            extra_scale_cache,
            block_table,
            eidx,
            extra_topk_length,
            block_size,
        )
        kvs.append(kv2)
        valids.append(valid2)

    kv = torch.cat(kvs, dim=1)  # [T, total, D]
    valid = torch.cat(valids, dim=1)  # [T, total]
    total = kv.shape[1]
    kv_flat = kv.reshape(T * total, D).to(q2.dtype)
    # Per-token slots index a contiguous [t*total, (t+1)*total) block of kv_flat;
    # invalid slots get the -1 sentinel (masked by the compute kernel).
    base = (torch.arange(T, device=dev) * total).view(T, 1)
    slot = torch.arange(total, device=dev).view(1, total).expand(T, total)
    idx_flat = torch.where(valid, base + slot, torch.full_like(slot, -1)).to(torch.int32)

    out, lse, _ = sparse_mla_attention(
        q2,
        kv_flat,
        idx_flat,
        sm_scale=softmax_scale,
        attn_sink=attn_sink,
        d_v=head_dim_v,
        backend=backend,
    )
    return out.unsqueeze(1), lse
