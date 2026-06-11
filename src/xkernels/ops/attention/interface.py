# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public attention ops (``mha_merge_state``, ``dsa_indexer_logits``): each
dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import (
    dsa_reference,  # noqa: F401  (registers dsa_indexer_logits REFERENCE)
    reference,  # noqa: F401  (registers REFERENCE backend)
)
from .dsa_reference import dsa_indexer_topk  # noqa: F401  (re-export thin helper)


def dsa_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """DeepSeek-V4 DSA indexer logits (issue #27): weighted ReLU MQA scores that
    select the top-k KV for sparse attention. Portable gfx942 replacement for the
    NVIDIA-only ``deep_gemm.fp8_fp4_mqa_logits``.

    ``logits[t, j] = sum_h weights[t, h] * relu(q[t, h, :] . k[j, :])``.

    Args:
        q: ``[T, H, D]`` indexer queries (``H = index_n_heads``, ``D = index_head_dim``).
        k: ``[K, D]`` single shared indexer key per KV position (MQA).
        weights: ``[T, H]`` (or ``[T, H, 1]``) per-head combine weights.
        lengths: optional ``[T]`` int — valid KV columns per query.
        row_starts: optional ``[T]`` int — first valid KV column per query
            (defaults to 0). Columns outside the window are masked to ``-inf``.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``logits [T, K]`` fp32. Pair with :func:`dsa_indexer_topk` to obtain the
        top-k KV indices.
    """
    return dispatch(
        "dsa_indexer_logits",
        q,
        k,
        weights,
        lengths=lengths,
        row_starts=row_starts,
        backend=backend,
    )


def mha_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials by their log-sum-exp (online softmax).

    For chunked-prefill / split-KV MLA: combine per-KV-chunk partial outputs and
    LSEs into a single output + merged LSE.

    Args:
        out_a, out_b: ``[T, H, D]`` partial outputs (bf16 or fp32).
        lse_a, lse_b: ``[T, H]`` fp32 log-sum-exp (natural-log basis).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out [T, H, D] in out_a.dtype, lse [T, H] fp32)``.
    """
    return dispatch("mha_merge_state", out_a, lse_a, out_b, lse_b, backend=backend)
