# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 DSA indexer logits (issue #27) for AMD MI300A (gfx942).

Portable Triton replacement for the NVIDIA-only ``deep_gemm.fp8_fp4_mqa_logits``
used by the V4 sparse-attention indexer. Computes, per query token ``t`` and KV
position ``j``, the weighted ReLU MQA score

    logits[t, j] = sum_h weights[t, h] * relu( q[t, h, :] . k[j, :] )

then masks columns outside the causal window ``[row_starts, row_starts+lengths)``
to ``-inf``. Pairs with ``torch.topk`` (see ``dsa_indexer_topk``) to select the
top-512 (Flash) / 1024 (Pro) KV per query.

One program handles one ``(query, KV-tile)`` pair: it loads the full ``[H, D]``
query (``H = index_n_heads = 64``, ``D = index_head_dim = 128`` fit in registers),
streams a ``BLOCK_K``-row tile of the shared MQA key, and reduces over heads in
fp32 on the hardware. The bf16/fp16 inputs are upcast to fp32 for the dot product
and ReLU, matching the reference oracle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["dsa_indexer_logits_triton", "dsa_indexer_logits_kernel"]


@triton.jit
def dsa_indexer_logits_kernel(
    q_ptr,  # [T, H, D]
    k_ptr,  # [K, D]
    w_ptr,  # [T, H]
    lengths_ptr,  # [T] int32 or nullptr
    starts_ptr,  # [T] int32 or nullptr
    out_ptr,  # [T, K] fp32
    T,
    K,
    H,
    D,
    stride_qt,
    stride_qh,
    stride_kk,
    stride_wt,
    stride_ot,
    HAS_MASK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(axis=0)
    k_tile = tl.program_id(axis=1)

    k_idx = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]
    k_mask = k_idx < K

    d_idx = tl.arange(0, BLOCK_D)  # [BLOCK_D]
    d_mask = d_idx < D
    h_idx = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_idx < H

    # KV tile: [BLOCK_K, BLOCK_D], shared across heads (MQA).
    k_blk = tl.load(
        k_ptr + k_idx[:, None] * stride_kk + d_idx[None, :],
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Per-head queries and weights for this token: [BLOCK_H, BLOCK_D], [BLOCK_H].
    q_blk = tl.load(
        q_ptr + t * stride_qt + h_idx[:, None] * stride_qh + d_idx[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    w_blk = tl.load(
        w_ptr + t * stride_wt + h_idx, mask=h_mask, other=0.0
    ).to(tl.float32)

    # scores[h, j] = relu(q[h] . k[j]); accumulate weighted sum over heads.
    # [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_K] -> [BLOCK_H, BLOCK_K]
    scores = tl.dot(q_blk, tl.trans(k_blk))  # [BLOCK_H, BLOCK_K] fp32
    scores = tl.maximum(scores, 0.0)
    scores = scores * w_blk[:, None]
    # Zero out padding heads so they do not contribute.
    scores = tl.where(h_mask[:, None], scores, 0.0)
    logits = tl.sum(scores, axis=0)  # [BLOCK_K]

    if HAS_MASK:
        length = tl.load(lengths_ptr + t).to(tl.int32)
        start = tl.load(starts_ptr + t).to(tl.int32)
        valid = (k_idx >= start) & (k_idx < (start + length))
        # NOTE: inline literal, not a module global — the Triton *compiler* (unlike
        # TRITON_INTERPRET=1) rejects reading non-constexpr globals from @jit code.
        logits = tl.where(valid, logits, float("-inf"))

    tl.store(out_ptr + t * stride_ot + k_idx, logits, mask=k_mask)


def dsa_indexer_logits_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton DSA indexer logits. See ``dsa_reference.dsa_indexer_logits_ref``."""
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    q = q.contiguous()
    k = k.contiguous()
    weights = weights.contiguous()
    T, H, D = q.shape
    K = k.shape[0]
    out = torch.empty((T, K), dtype=torch.float32, device=q.device)

    has_mask = lengths is not None
    if has_mask:
        if row_starts is None:
            row_starts = torch.zeros_like(lengths)
        lengths_i = lengths.to(torch.int32).contiguous()
        starts_i = row_starts.to(torch.int32).contiguous()
    else:
        # Dummy 1-element tensors; kernel never dereferences them when HAS_MASK=False.
        lengths_i = torch.zeros(1, dtype=torch.int32, device=q.device)
        starts_i = lengths_i

    block_k = 64
    block_d = triton.next_power_of_2(D)
    block_h = triton.next_power_of_2(H)
    grid = (T, triton.cdiv(K, block_k))
    dsa_indexer_logits_kernel[grid](
        q,
        k,
        weights,
        lengths_i,
        starts_i,
        out,
        T,
        K,
        H,
        D,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        out.stride(0),
        HAS_MASK=has_mask,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
    )
    return out


register("dsa_indexer_logits", Backend.TRITON)(dsa_indexer_logits_triton)


# --------------------------------------------------------------------------- #
# Fused DSA indexer top-k (issue #54): logits + top-k in one launch.          #
# --------------------------------------------------------------------------- #


@triton.jit
def dsa_indexer_topk_kernel(
    q_ptr,  # [T, H, D]
    k_ptr,  # [K, D]
    w_ptr,  # [T, H]
    lengths_ptr,  # [T] int32 or nullptr
    starts_ptr,  # [T] int32 or nullptr
    logits_ptr,  # [T, K] fp32 scratch (per-query; stays in L2)
    indices_ptr,  # [T, topk] int32 output
    T,
    K,
    H,
    D,
    stride_qt,
    stride_qh,
    stride_kk,
    stride_wt,
    stride_lt,
    stride_it,
    HAS_MASK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K_TOTAL: tl.constexpr,
    TOPK: tl.constexpr,
):
    """One program per query token. Phase 1 streams KV in BLOCK_K chunks,
    computing the weighted-ReLU MQA logits and writing them to a per-query
    scratch row (stays in L2). Phase 2 loads the full [BLOCK_K_TOTAL] row and
    does an iterative argmax top-k (tl.static_range — unrolled, matching the
    topk_softmax_kernel pattern; tl.range does not correctly accumulate the
    loop-carried `selected` mask in this Triton version), writing [TOPK]
    int32 indices in descending-logit order (ties by ascending KV id, since
    tl.argmax returns the lowest index on ties)."""
    t = tl.program_id(axis=0)

    d_idx = tl.arange(0, BLOCK_D)  # [BLOCK_D]
    d_mask = d_idx < D
    h_idx = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    h_mask = h_idx < H

    # Load query [BLOCK_H, BLOCK_D] and weights [BLOCK_H] (loop-invariant).
    q_blk = tl.load(
        q_ptr + t * stride_qt + h_idx[:, None] * stride_qh + d_idx[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    w_blk = tl.load(
        w_ptr + t * stride_wt + h_idx, mask=h_mask, other=0.0
    ).to(tl.float32)

    # --- Phase 1: compute logits, stream KV in BLOCK_K chunks ---
    for k_start in tl.range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < K
        k_blk = tl.load(
            k_ptr + k_idx[:, None] * stride_kk + d_idx[None, :],
            mask=k_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.maximum(tl.dot(q_blk, tl.trans(k_blk)), 0.0)  # [H, BLOCK_K]
        scores = scores * w_blk[:, None]
        scores = tl.where(h_mask[:, None], scores, 0.0)
        logits_chunk = tl.sum(scores, axis=0)  # [BLOCK_K]

        if HAS_MASK:
            length = tl.load(lengths_ptr + t).to(tl.int32)
            start = tl.load(starts_ptr + t).to(tl.int32)
            valid = (k_idx >= start) & (k_idx < (start + length))
            logits_chunk = tl.where(valid, logits_chunk, -1e30)

        tl.store(logits_ptr + t * stride_lt + k_idx, logits_chunk, mask=k_mask)

    # --- Phase 2: iterative argmax top-k over the full [BLOCK_K_TOTAL] row ---
    k_offs = tl.arange(0, BLOCK_K_TOTAL)
    k_valid = k_offs < K
    logits = tl.load(
        logits_ptr + t * stride_lt + k_offs, mask=k_valid, other=float("-inf")
    )

    selected = tl.zeros([BLOCK_K_TOTAL], dtype=tl.int1)
    base_i = t * TOPK
    for j in tl.static_range(TOPK):
        cand = tl.where(selected | (~k_valid), float("-inf"), logits)
        idx = tl.argmax(cand, axis=0)
        tl.store(indices_ptr + base_i + j, idx.to(tl.int32))
        selected = selected | (k_offs == idx)


def dsa_indexer_topk_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Triton DSA indexer top-k. See ``dsa_reference.dsa_indexer_topk_ref``."""
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    q = q.contiguous()
    k = k.contiguous()
    weights = weights.contiguous()
    T, H, D = q.shape
    K = k.shape[0]
    if not (1 <= int(topk) <= K):
        raise ValueError(
            f"topk must satisfy 1 <= topk <= K (got topk={topk}, K={K})"
        )
    device = q.device
    # Per-query logits scratch (stays in L2: written then read by the same program).
    logits = torch.full((T, K), -1e30, dtype=torch.float32, device=device)
    indices = torch.empty((T, int(topk)), dtype=torch.int32, device=device)

    has_mask = lengths is not None
    if has_mask:
        if row_starts is None:
            row_starts = torch.zeros_like(lengths)
        lengths_i = lengths.to(torch.int32).contiguous()
        starts_i = row_starts.to(torch.int32).contiguous()
    else:
        lengths_i = torch.zeros(1, dtype=torch.int32, device=device)
        starts_i = lengths_i

    block_k = 64
    block_d = triton.next_power_of_2(D)
    block_h = triton.next_power_of_2(H)
    block_k_total = triton.next_power_of_2(K)
    grid = (T,)
    dsa_indexer_topk_kernel[grid](
        q,
        k,
        weights,
        lengths_i,
        starts_i,
        logits,
        indices,
        T,
        K,
        H,
        D,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        weights.stride(0),
        logits.stride(0),
        indices.stride(0),
        HAS_MASK=has_mask,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        BLOCK_K_TOTAL=block_k_total,
        TOPK=int(topk),
        num_warps=4,
    )
    return indices


register("dsa_indexer_topk", Backend.TRITON)(dsa_indexer_topk_triton)
