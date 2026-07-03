# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Batched paged grouped-query attention -- DECODE path (issue #71).

The throughput-critical kernel for serving any dense GQA model (Qwen3, Llama):
every decode step, each active request has ONE new query token and attends to its
own growing KV history, which lives in a paged KV cache (the RadixCache / vLLM /
SGLang paged pool). ``paged_attention`` is the fused batched kernel that does the
gather + GQA attention for the whole batch in one launch -- replacing the
per-request Python loop + per-request KV-gather materialization that the portable
SDPA fallback currently pays on every decode step (the dominant decode
bottleneck, per issue #71's scope update).

Contract -- batched paged GQA DECODE (one q-token per request):

    q          : [B, H_q, D]      (one query token per request; bf16/fp16)
    k_cache    : [Nb, block, H_kv, D]   paged K pool (same dtype as q)
    v_cache    : [Nb, block, H_kv, D]   paged V pool (same dtype as q)
    block_table: [B, max_blocks]  int32 -- request b's pages are block_table[b, :]
    seq_lens   : [B]              int32 -- number of VALID KV tokens per request
    scale      : float            (typically head_dim**-0.5)
    -> out     : [B, H_q, D]      (input dtype)

GQA: ``H_kv <= H_q``, group = ``H_q // H_kv``. Query head ``h`` attends to KV head
``h // group`` (the standard contiguous GQA head mapping). Decoding the single
new token attends to ALL of its request's past KV positions ``[0, seq_len)`` --
there is no causal mask (the new token IS the last position, so every past
position is in-causal). Scores + softmax are fp32; the output is cast to the
input dtype.

This module ships the backend-neutral reference (per-request SDPA oracle, written
for clarity) + the public dispatch. The Triton device kernel is a separate card;
this reference is what makes ``verify`` runnable with no GPU (the gateway skill's
CPU-satisfiable gate). PREFILL (variable-length multi-token, ragged
``cu_seqlens``) is a SEPARATE op -- ``paged_attention_prefill`` -- with a
different q shape (``[num_tokens, H_q, D]``) and a varlen flash kernel; it is the
documented follow-up, not this increment.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register

__all__ = ["paged_attention", "paged_attention_decode_ref"]


def paged_attention_decode_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    workspace=None,
) -> torch.Tensor:
    """Reference: batched paged GQA decode. See module docstring.

    ``workspace`` is accepted for signature uniformity with the Triton backend
    but ignored (the reference always returns a fresh tensor).

    For each request ``b``, gathers its valid KV pages from the pool, computes
    fp32 GQA attention of that request's single query token against its full KV
    history, and writes the output. Written for CLARITY (a per-request Python
    loop), not speed -- the device kernel fuses this whole batch into one launch.
    """
    if q.dim() != 3:
        raise ValueError(f"q must be 3-D [B, H_q, D], got shape {tuple(q.shape)}")
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache must share shape; got {tuple(k_cache.shape)} vs "
            f"{tuple(v_cache.shape)}"
        )
    B, H_q, D = q.shape
    _Nb, block_size, H_kv, _D = k_cache.shape
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
        )
    group = H_q // H_kv

    qf = q.float()  # [B, H_q, D]
    out = q.new_empty(B, H_q, D)

    for b in range(B):
        sl = int(seq_lens[b].item())
        if sl <= 0:
            out[b] = 0
            continue
        # Gather this request's valid KV from the paged pool. The first
        # ceil(sl / block_size) pages are valid; reshape to [sl, H_kv, D].
        n_valid_blocks = (sl + block_size - 1) // block_size
        page_ids = block_table[b, :n_valid_blocks].long()  # [n_valid_blocks]
        k = k_cache[page_ids].reshape(-1, H_kv, D)[:sl].float()  # [sl, H_kv, D]
        v = v_cache[page_ids].reshape(-1, H_kv, D)[:sl].float()  # [sl, H_kv, D]
        qh = qf[b]  # [H_q, D]
        # GQA: expand the H_kv KV heads to H_q by repeating each kv head `group`
        # times (kv_head qo_head//group is the contiguous GQA mapping).
        ke = k.repeat_interleave(group, dim=1)  # [sl, H_q, D]
        ve = v.repeat_interleave(group, dim=1)  # [sl, H_q, D]
        # scores[h, s] = scale * sum_d q[h,d] * k[s,h,d]   -> [H_q, sl]
        scores = scale * torch.einsum("hd,shd->hs", qh, ke)
        p = torch.softmax(scores, dim=-1)  # [H_q, sl]
        # out[h, d] = sum_s p[h, s] * v[s, h, d]
        out[b] = torch.einsum("hs,shd->hd", p, ve).to(q.dtype)
    return out


register("paged_attention", Backend.REFERENCE)(paged_attention_decode_ref)


def paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    backend: Backend | str = "auto",
    workspace=None,
) -> torch.Tensor:
    """Batched paged grouped-query attention (DECODE). See
    :func:`paged_attention_decode_ref`.

    ``backend="auto"`` picks the fastest registered backend (Triton device kernel
    when available, else the pure-torch reference).

    ``workspace`` (optional :class:`PagedAttentionWorkspace`): reuse a
    preallocated output buffer across decode steps to avoid per-call
    allocation and enable CUDA/HIP graph capture (issue #52). Ignored by the
    reference backend.
    """
    return dispatch(
        "paged_attention",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        scale=scale,
        workspace=workspace,
        backend=backend,
    )
