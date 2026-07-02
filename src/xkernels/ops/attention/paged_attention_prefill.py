# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Variable-length paged grouped-query attention -- PREFILL / EXTEND path
(issue #71 prefill half).

The complement of ``paged_attention`` (decode): where decode has ONE new q-token
per request, prefill has a *packed batch* of ``num_tokens`` query tokens spanning
``num_seqs`` variable-length sequences, and each query token attends CAUSALLY to
its sequence's growing KV history in the paged pool. This is the
flashinfer ``BatchPrefillWithPagedKVCache`` shape -- the make-or-break serving
kernel for any dense GQA model (Qwen3, Llama) per issue #71, whose primary target
is ``amd_cdna3`` (gfx942).

Contract -- varlen paged GQA PREFILL/EXTEND:

    q            : [num_tokens, H_q, D]      packed query tokens (bf16/fp16/fp32)
    k_cache      : [Nb, block, H_kv, D]      paged K pool (same dtype as q)
    v_cache      : [Nb, block, H_kv, D]      paged V pool (same dtype as q)
    block_table  : [num_seqs, max_blocks] int32  -- seq s's pages are block_table[s,:]
    cu_seqlens_q : [num_seqs+1] int32        cumulative q-token counts (cu[0]==0)
    cu_seqlens_k : [num_seqs+1] int32        cumulative kv-length counts (cu[0]==0)
    scale        : float                     (typically head_dim**-0.5)
    -> out       : [num_tokens, H_q, D]      (input dtype)

GQA: ``H_kv <= H_q``, group = ``H_q // H_kv``; qo head ``h`` reads kv head
``h // group``. CAUSAL within each sequence: for seq ``s`` with ``nq`` new q-tokens
and ``nk`` total kv positions, the p-th new token (0-indexed) attends to kv
positions ``[0, (nk - nq) + p + 1)`` -- i.e. the new tokens are the LAST ``nq``
positions of the sequence. This is *extend* semantics and covers both pure
prefill (``nk == nq``, prefix = 0) and chunked/extend prefill (``nk > nq``,
prefix = ``nk - nq`` already-cached KV). Scores + softmax are fp32; output cast
back to the input dtype.

This module ships the backend-neutral reference (a per-sequence, per-token SDPA
oracle, written for clarity) + the public dispatch. The reference is what makes
``verify`` runnable with no GPU (the gateway skill's CPU-satisfiable gate).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register

__all__ = ["paged_attention_prefill", "paged_attention_prefill_ref"]


def paged_attention_prefill_ref(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Reference: varlen paged GQA prefill. See module docstring.

    For each sequence, gathers its valid KV pages from the pool, then for each
    new query token computes fp32 GQA attention against the token's CAUSAL kv
    range ``[0, prefix + p + 1)``. Written for CLARITY (per-seq, per-token Python
    loop), not speed -- the device kernel fuses the whole packed batch into one
    launch.
    """
    if q.dim() != 3:
        raise ValueError(
            f"q must be 3-D [num_tokens, H_q, D], got shape {tuple(q.shape)}"
        )
    if k_cache.shape != v_cache.shape:
        raise ValueError(
            f"k_cache and v_cache must share shape; got {tuple(k_cache.shape)} vs "
            f"{tuple(v_cache.shape)}"
        )
    num_tokens, H_q, D = q.shape
    _Nb, block_size, H_kv, _D = k_cache.shape
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
        )
    group = H_q // H_kv
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_k must have equal length; got "
            f"{cu_seqlens_q.numel()} vs {cu_seqlens_k.numel()}"
        )
    num_seqs = cu_seqlens_q.numel() - 1
    if num_seqs < 1:
        raise ValueError("need at least one sequence (cu_seqlens len >= 2)")

    qf = q.float()  # [num_tokens, H_q, D]
    out = q.new_empty(num_tokens, H_q, D)

    for s in range(num_seqs):
        q_start = int(cu_seqlens_q[s].item())
        q_end = int(cu_seqlens_q[s + 1].item())
        k_start = int(cu_seqlens_k[s].item())
        k_end = int(cu_seqlens_k[s + 1].item())
        nq = q_end - q_start
        nk = k_end - k_start
        if nq <= 0:
            continue
        if nk < nq:
            raise ValueError(
                f"seq {s}: kv length ({nk}) < q length ({nq}); the new q-tokens "
                f"must be a SUFFIX of the kv (nk >= nq required for causal extend)"
            )
        prefix = nk - nq  # already-cached kv before this batch's new tokens

        # Gather this seq's valid KV from the paged pool -> [nk, H_kv, D].
        n_valid_blocks = (nk + block_size - 1) // block_size
        page_ids = block_table[s, :n_valid_blocks].long()
        k = k_cache[page_ids].reshape(-1, H_kv, D)[:nk].float()
        v = v_cache[page_ids].reshape(-1, H_kv, D)[:nk].float()
        # GQA: expand kv heads (kv_head qo_head//group is the contiguous mapping).
        ke = k.repeat_interleave(group, dim=1)  # [nk, H_q, D]
        ve = v.repeat_interleave(group, dim=1)

        qs = qf[q_start:q_end]  # [nq, H_q, D]
        for p in range(nq):
            causal_end = prefix + p + 1  # this token attends to kv [0, causal_end)
            qh = qs[p]  # [H_q, D]
            scores = scale * torch.einsum("hd,shd->hs", qh, ke[:causal_end])
            pp = torch.softmax(scores, dim=-1)  # [H_q, causal_end]
            out[q_start + p] = torch.einsum(
                "hs,shd->hd", pp, ve[:causal_end]
            ).to(q.dtype)
    return out


register("paged_attention_prefill", Backend.REFERENCE)(paged_attention_prefill_ref)


def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Variable-length paged grouped-query attention (PREFILL/EXTEND). See
    :func:`paged_attention_prefill_ref`.

    ``backend="auto"`` picks the fastest registered backend (Triton device kernel
    when available, else the pure-torch reference).
    """
    return dispatch(
        "paged_attention_prefill",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        scale=scale,
        backend=backend,
    )
