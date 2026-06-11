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
