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

The **fused** ``dsa_indexer_topk`` (issue #54) computes the same weighted ReLU
MQA logits and selects the top-k KV indices in ONE pass, without materializing
the full ``[T, K]`` fp32 logits tensor. Its reference (:func:`dsa_indexer_topk_ref`)
uses a canonical descending argsort (ties by ascending KV id), NOT
``torch.topk(sorted=False)`` whose tie-break is unspecified.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["dsa_indexer_logits_ref", "dsa_indexer_topk_ref", "dsa_indexer_topk_from_logits"]

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


def dsa_indexer_topk_from_logits(
    logits: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` KV indices per query from **precomputed** indexer logits.

    This is the *diagnostics* selection path kept for reference parity (issue #54):
    callers that already hold the full ``[T, K]`` logits (e.g. from
    :func:`dsa_indexer_logits`) can select the top-k without re-running the
    scorer. Returns ``[T, topk]`` int32 in **canonical descending order** (ties
    broken by ascending KV id), NOT ``torch.topk(sorted=False)`` whose tie-break
    is unspecified and diverges from the Triton kernel's ``tl.argmax`` (see
    :func:`dsa_indexer_topk_ref` numerics notes).
    """
    k = min(topk, logits.shape[-1])
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    return order[:, :k].to(torch.int32).contiguous()


def dsa_indexer_topk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused DSA indexer top-k: weighted ReLU MQA logits + canonical top-k selection.

    Computes :func:`dsa_indexer_logits_ref` ``(q, k, weights, lengths, row_starts)``
    then selects the top-``topk`` KV indices per query using a **canonical
    descending argsort** (ties broken by ascending KV id), NOT ``torch.topk``
    (whose tie-break is unspecified — see registry/ops/dsa_indexer_topk.spec.json
    numerics.notes). This is the backend-neutral oracle every Impl Card is
    checked against (issue #54).

    Args:
        q: ``[T, H, D]`` indexer queries (``H = index_n_heads``, ``D = index_head_dim``).
        k: ``[K, D]`` single shared indexer key per KV position (MQA).
        weights: ``[T, H]`` (or ``[T, H, 1]``) per-head combine weights.
        topk: number of KV positions to select per query (``1 <= topk <= K``).
        lengths: optional ``[T]`` int — valid KV columns per query.
        row_starts: optional ``[T]`` int — first valid KV column per query.

    Returns:
        ``indices [T, topk]`` int32 in descending-logit order (ties by ascending
        KV id). Out-of-window columns (masked to ``-inf``) are never selected.
    """
    logits = dsa_indexer_logits_ref(q, k, weights, lengths=lengths, row_starts=row_starts)
    K = logits.shape[-1]
    if not (1 <= int(topk) <= K):
        raise ValueError(f"topk must satisfy 1 <= topk <= K (got topk={topk}, K={K})")
    order = torch.argsort(logits, dim=1, descending=True, stable=True)
    return order[:, : int(topk)].to(torch.int32).contiguous()


register("dsa_indexer_logits", Backend.REFERENCE)(dsa_indexer_logits_ref)
register("dsa_indexer_topk", Backend.REFERENCE)(dsa_indexer_topk_ref)
