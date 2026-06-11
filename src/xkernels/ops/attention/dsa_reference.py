# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 DSA (DeepSeek Sparse Attention)
indexer forward path (issue #27) — numerical oracle and default (CPU / no-Triton)
backend for AMD MI300A (gfx942).

The DSA indexer is an MQA (multi-query) scorer that selects, per query token, the
top-``topk`` KV positions for the sparse attention that follows. Upstream
``deepseek_v4`` computes the indexer logits with NVIDIA-only fp8/fp4 kernels
(``deep_gemm.fp8_fp4_mqa_logits`` + a CUDA mxfp4 paged gather). The numerically
meaningful operation — the one that *selects* which KV survive — is a weighted
ReLU MQA dot-product followed by a masked top-k:

    logits[t, j] = sum_h weights[t, h] * relu( q[t, h, :] . k[j, :] )

with ``q : [T, H, D]`` (``H = index_n_heads``, ``D = index_head_dim``), a single
shared ``k : [K, D]`` per KV position (MQA), and per-head combine ``weights :
[T, H]``. An optional causal window ``[row_starts, row_starts + lengths)`` masks
out-of-range columns to ``-inf`` before the top-k. This mirrors the upstream
torch oracle ``_indexer_topk_reference`` exactly (``einsum('thd,kd->thk').relu()``,
weight, sum over heads, mask, ``torch.topk``).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["dsa_indexer_logits_ref", "dsa_indexer_topk"]

_NEG_INF = float("-inf")


def _apply_causal_mask(
    logits: torch.Tensor,
    lengths: torch.Tensor | None,
    row_starts: torch.Tensor | None,
) -> torch.Tensor:
    """Mask columns outside ``[row_starts, row_starts + lengths)`` to ``-inf``."""
    if lengths is None:
        return logits
    K = logits.shape[-1]
    if row_starts is None:
        row_starts = torch.zeros_like(lengths)
    cols = torch.arange(K, device=logits.device)
    valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
        cols.unsqueeze(0) < (row_starts + lengths).unsqueeze(1)
    )
    return logits.masked_fill(~valid, _NEG_INF)


def dsa_indexer_logits_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted ReLU MQA indexer logits.

    Args:
        q: ``[T, H, D]`` indexer queries (``H`` index heads, ``D`` index_head_dim).
        k: ``[K, D]`` single shared indexer key per KV position (MQA).
        weights: ``[T, H]`` (or ``[T, H, 1]``) per-head combine weights.
        lengths: optional ``[T]`` int — number of valid KV columns per query.
        row_starts: optional ``[T]`` int — first valid KV column per query
            (defaults to 0). Columns outside ``[row_starts, row_starts+lengths)``
            are masked to ``-inf``.

    Returns:
        ``logits [T, K]`` fp32.
    """
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    logits = torch.einsum("thd,kd->thk", q.float(), k.float()).relu()
    logits = (logits * weights.float().unsqueeze(-1)).sum(dim=1)
    return _apply_causal_mask(logits, lengths, row_starts)


def dsa_indexer_topk(
    logits: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` KV indices per query from indexer logits.

    Returns ``[T, topk]`` int32 (unsorted, matching the upstream
    ``torch.topk(..., sorted=False)`` indexer selection).
    """
    k = min(topk, logits.shape[-1])
    return torch.topk(logits, k=k, dim=-1, sorted=False).indices.to(torch.int32)


register("dsa_indexer_logits", Backend.REFERENCE)(dsa_indexer_logits_ref)
