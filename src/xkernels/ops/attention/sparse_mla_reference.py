# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 sparse-MLA attention compute
(issue #32) — numerical oracle and default (CPU / no-Triton) backend on gfx942.

This is the kernel that *consumes* the DSA indexer's top-k KV selection (#27/#31)
and runs the actual attention softmax over V4's latent KV. MLA in latent form is
MQA: a single shared latent KV per position of dim ``D = kv_lora_rank + rope``
(V4: 512 = 448 + 64). The score uses the full ``D``; the value is the first
``d_v`` (the kv_lora / nope part). An optional per-head attention **sink** logit
joins the softmax denominator and contributes no value.

    s[t,h,j] = sm_scale * (q[t,h] . kv[idx[t,j]])      over selected idx
    p        = softmax([s..., sink[h]])                 (sink column has zero value)
    out[t,h] = sum_j p[j] * kv[idx[t,j], :d_v]

Validity of a column ``j`` is ``idx >= 0`` AND (when given) ``j < topk_length[t]``.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["sparse_mla_attention_ref"]

_NEG_INF = float("-inf")


def sparse_mla_attention_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    d_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse latent-MLA attention. See module docstring.

    Args:
        q: ``[T, H, D]`` latent queries.
        kv: ``[Kv, D]`` shared latent MQA cache (bf16/fp32).
        indices: ``[T, topk]`` int32 columns into ``kv`` (``<0`` = padding).
        sm_scale: softmax scale applied to the q.k score.
        topk_length: optional ``[T]`` int — valid column count per query.
        attn_sink: optional ``[H]`` (or scalar) per-head sink logit.
        d_v: value/output dim (first ``d_v`` latent dims). Defaults to ``D``.

    Returns:
        ``(out [T, H, d_v] in q.dtype, lse [T, H] fp32, max_logits [T, H] fp32)``.
    """
    T, H, D = q.shape
    Kv = kv.shape[0]
    topk = indices.shape[1]
    d_v = D if d_v is None else d_v
    qf, kvf = q.float(), kv.float()
    out = q.new_zeros(T, H, d_v)
    lse = q.new_zeros(T, H, dtype=torch.float32)
    maxl = q.new_zeros(T, H, dtype=torch.float32)
    pos = torch.arange(topk, device=q.device)

    sink_vec = None
    if attn_sink is not None:
        s = attn_sink.float().reshape(-1)
        sink_vec = (s.expand(H) if s.numel() == 1 else s[:H]).reshape(H, 1)

    for t in range(T):
        idx = indices[t].long()
        valid = idx >= 0
        if topk_length is not None:
            valid = valid & (pos < int(topk_length[t]))
        safe = idx.clamp(0, Kv - 1)
        ksel = kvf[safe]  # [topk, D]
        scores = sm_scale * (qf[t] @ ksel.t())  # [H, topk]
        scores = scores.masked_fill(~valid.unsqueeze(0), _NEG_INF)
        aug = scores if sink_vec is None else torch.cat([scores, sink_vec], dim=1)
        m = aug.amax(dim=1)  # [H]
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = (aug - m_safe.unsqueeze(1)).exp()
        denom = p.sum(dim=1)
        pv = p[:, :topk]  # sink column excluded from value
        ov = (pv @ ksel[:, :d_v]) / denom.clamp_min(1e-20).unsqueeze(1)
        out[t] = ov.to(q.dtype)
        lse[t] = torch.where(
            denom > 0,
            m_safe + denom.clamp_min(1e-20).log(),
            torch.full_like(m_safe, _NEG_INF),
        )
        maxl[t] = m_safe
    return out, lse, maxl


register("sparse_mla_attention", Backend.REFERENCE)(sparse_mla_attention_ref)
