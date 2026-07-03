# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton varlen paged GQA prefill kernel (issue #71 prefill half).

One program per ``(global q-token, qo_head)``: load that token's query head into
registers, then stream its sequence's paged KV in ``BLOCK_N`` chunks with the
flash (online) softmax, bounded by the CAUSAL range. The page indirection lives
INSIDE the chunk loop: position ``j`` -> ``block_id = block_table[s, j//bs]``,
``offset = j % bs``; load ``k_cache[block_id, offset, kv_head, :]``. GQA is free
(each program owns one qo head, reading exactly its mapped kv head ``h // group``).

CAUSAL extend semantics: for seq ``s`` with ``nq`` new q-tokens and ``nk`` total
kv positions, the p-th new token (0-indexed) attends to kv ``[0, (nk-nq)+p+1)``.
Because ``causal_end <= nk`` always, the causal mask doubles as the valid-kv mask
(positions in a partial last page beyond ``nk`` are never reached). This covers
pure prefill (``nk == nq``) and chunked/extend prefill (``nk > nq``).

Finding each token's sequence: the host launcher precomputes ``seq_ids[t]`` via
``torch.searchsorted(cu_seqlens_q, t, right=True) - 1`` (one O(num_tokens log
num_seqs) pass), so the device program loads its seq index from a small tensor
instead of doing a per-program scan.

v1 PERF (honest): this is the *per-token* flash kernel (BLOCK_M = 1) -- each kv
position is loaded once per attending q-token, i.e. O(L^2/2) kv loads per seq,
with no Q-reuse tiling across BLOCK_M q-rows. That is the naive flash cost, not
the BLOCK_M-tiled flash cost that amortizes each kv load across a q-tile. It is
CORRECT and fuses the packed batch into one launch (killing the per-token Python
loop + per-token KV-gather materialization of the SDPA fallback, the issue's
unblock target), but for LONG prefills (4k+ tokens) the BLOCK_M-tiled varlen
flash kernel is the documented perf follow-up -- correct-first per the issue's
"Triton card to unblock" directive. Portable (manual FMA + tl.exp, no vendor
intrinsics); the AMD gfx942 card (the issue's primary target) is a
port-across-arch step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["paged_attention_prefill_triton", "paged_attention_prefill_kernel"]


@triton.jit
def paged_attention_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    cuq_ptr,
    cuk_ptr,
    sids_ptr,
    out_ptr,
    scale,
    H_q,
    H_kv,
    D,
    block_size,
    # q strides [num_tokens, H_q, D]
    stride_qt,
    stride_qh,
    stride_qd,
    # kv strides [Nb, block, H_kv, D]
    stride_kn,
    stride_kb,
    stride_kh,
    stride_kd,
    # block_table strides [num_seqs, max_blocks]
    stride_bts,
    stride_btm,
    # out strides [num_tokens, H_q, D]
    stride_ot,
    stride_oh,
    stride_od,
    GROUP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)  # global q-token index
    h = tl.program_id(1)  # qo head index
    kv_head = h // GROUP  # GQA head mapping (runtime int division)

    # Resolve this token's sequence + local position via the host-computed
    # seq_ids tensor (no per-program scan over cu_seqlens).
    s = tl.load(sids_ptr + t)
    q_start = tl.load(cuq_ptr + s)
    q_end = tl.load(cuq_ptr + s + 1)
    k_end = tl.load(cuk_ptr + s + 1)
    nq = q_end - q_start
    nk = k_end - tl.load(cuk_ptr + s)
    prefix = nk - nq
    p = t - q_start  # local q position within the seq's new tokens
    causal_end = prefix + p + 1  # attend to kv [0, causal_end); <= nk always

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Load the query head vector [BLOCK_D] (fp32).
    q = tl.load(
        q_ptr + t * stride_qt + h * stride_qh + d_off * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # Online (flash) softmax accumulators.
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start in range(0, causal_end, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < causal_end  # causal + valid-kv combined (causal_end<=nk)
        # Page indirection within seq s: j -> (block_id, offset).
        page_idx = cols // block_size       # [BLOCK_N]
        offs_in_block = cols % block_size    # [BLOCK_N]
        block_id = tl.load(
            bt_ptr + s * stride_bts + page_idx * stride_btm,
            mask=col_mask, other=0,
        )
        k_ptrs = (
            k_ptr
            + block_id[:, None] * stride_kn
            + offs_in_block[:, None] * stride_kb
            + kv_head * stride_kh
            + d_off[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        # scores[j] = scale * sum_d q[d] * k[j, d]  -> [BLOCK_N]
        scores = scale * tl.sum(q[None, :] * k, axis=1)
        scores = tl.where(col_mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p_attn = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)  # rescale; ==1 on first chunk
        l_i = l_i * alpha + tl.sum(p_attn, axis=0)
        v_ptrs = (
            v_ptr
            + block_id[:, None] * stride_kn
            + offs_in_block[:, None] * stride_kb
            + kv_head * stride_kh
            + d_off[None, :] * stride_kd
        )
        v = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p_attn[:, None] * v, axis=0)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out_vec = acc / l_safe
    tl.store(
        out_ptr + t * stride_ot + h * stride_oh + d_off * stride_od,
        out_vec.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


def paged_attention_prefill_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    workspace=None,
) -> torch.Tensor:
    """Host launcher for the varlen paged GQA prefill kernel.

    Args match the reference
    (:func:`xkernels.ops.attention.paged_attention_prefill.paged_attention_prefill_ref`);
    returns ``out [num_tokens, H_q, D]``.

    ``workspace`` (optional :class:`PagedAttentionPrefillWorkspace`): if provided
    and matching, write into its ``out``/``seq_ids`` buffers (``[:num_tokens]``
    slices when allocated at ``num_tokens_max >= num_tokens``) instead of
    allocating. Enables graph capture (issue #52).
    """
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous().to(torch.int32)
    cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(torch.int32)

    num_tokens, H_q, D = q.shape
    _Nb, block_size, H_kv, _D = k_cache.shape
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
        )
    group = H_q // H_kv

    # Precompute seq_ids[t]: the sequence owning global q-token t. searchsorted
    # with right=True gives the first index where cu_seqlens_q[idx] > t, so idx-1
    # is the seq with cu_seqlens_q[s] <= t < cu_seqlens_q[s+1]. O(num_tokens log
    # num_seqs) once on the host; the device program then does one gather.
    token_idx = torch.arange(num_tokens, device=q.device, dtype=torch.int32)
    if workspace is not None:
        if not workspace.matches(num_tokens, H_q, D, device=q.device, dtype=q.dtype):
            raise ValueError(
                f"workspace buffer is shape {tuple(workspace.out.shape)} on "
                f"{workspace.out.device}/{workspace.out.dtype}; need >= "
                f"({num_tokens}, {H_q}, {D}) on {q.device}/{q.dtype}."
            )
        out = workspace.out[:num_tokens]
        seq_ids = workspace.seq_ids[:num_tokens]
        # searchsorted returns int64; compute then copy into the int32 workspace slice
        seq_ids.copy_(torch.searchsorted(cu_seqlens_q, token_idx, right=True) - 1)
    else:
        out = torch.empty(num_tokens, H_q, D, device=q.device, dtype=q.dtype)
        seq_ids = (
            torch.searchsorted(cu_seqlens_q, token_idx, right=True) - 1
        ).to(torch.int32)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_N = 64  # flash-chunk size; the flash reduction is exact for any size.
    grid = (num_tokens, H_q)
    paged_attention_prefill_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_ids,
        out,
        scale,
        H_q,
        H_kv,
        D,
        block_size,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        GROUP=group,
        BLOCK_D=BLOCK_D,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
    return out


register("paged_attention_prefill", Backend.TRITON)(paged_attention_prefill_triton)
