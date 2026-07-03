# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton batched paged GQA decode kernel (issue #71) -- the decode half.

One program per ``(request, qo_head)``: load that head's query vector into
registers, then stream the request's paged KV in ``BLOCK_N`` chunks with the
flash (online) softmax. The page indirection lives INSIDE the chunk loop:
position ``j`` -> ``block_id = block_table[b, j // block_size]``,
``offset = j % block_size``; load ``k_cache[block_id, offset, kv_head, :]``.
GQA is free (each program owns one qo head, so it reads exactly its one mapped
kv head ``h // group`` -- no expansion needed). Decoding the single new token
attends to ALL past positions ``[0, seq_len)`` -- no causal mask.

This is the vLLM ``paged_attention`` / flashinfer ``BatchDecodeWithPagedKVCache``
shape. Replacing the per-request Python loop + per-request KV-gather
materialization in the SDPA fallback is the single highest-impact throughput win
for mini-sglang decode (issue #71 scope update).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["paged_attention_triton", "paged_attention_decode_kernel"]


@triton.jit
def paged_attention_decode_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    bt_ptr,
    sl_ptr,
    out_ptr,
    scale,
    H_q,
    H_kv,
    D,
    block_size,
    max_blocks,
    # q strides [B, H_q, D]
    stride_qb,
    stride_qh,
    stride_qd,
    # kv strides [Nb, block, H_kv, D]
    stride_kn,
    stride_kb,
    stride_kh,
    stride_kd,
    # block_table strides [B, max_blocks]
    stride_btb,
    stride_btm,
    # out strides [B, H_q, D]
    stride_ob,
    stride_oh,
    stride_od,
    GROUP: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)  # request index
    h = tl.program_id(1)  # qo head index
    kv_head = h // GROUP  # the GQA head mapping (runtime int division)

    sl = tl.load(sl_ptr + b)  # valid KV length for this request
    if sl <= 0:
        # nothing to attend to -- zero the output head
        d_off = tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tl.store(out_ptr + b * stride_ob + h * stride_oh + d_off * stride_od,
                 tl.zeros([BLOCK_D], dtype=tl.float32).to(out_ptr.dtype.element_ty),
                 mask=d_mask)
        return

    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    # Load the query head vector [BLOCK_D] (fp32).
    q = tl.load(
        q_ptr + b * stride_qb + h * stride_qh + d_off * stride_qd,
        mask=d_mask, other=0.0,
    ).to(tl.float32)

    # Online (flash) softmax accumulators.
    m_i = -float("inf")  # running max score
    l_i = 0.0            # running denominator
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)  # running weighted value

    for start in range(0, sl, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < sl
        # Page indirection: j -> (block_id, offset).
        page_idx = cols // block_size       # [BLOCK_N]
        offs_in_block = cols % block_size    # [BLOCK_N]
        block_id = tl.load(
            bt_ptr + b * stride_btb + page_idx * stride_btm,
            mask=col_mask, other=0,
        )  # [BLOCK_N] -- the physical page index for each logical position
        # k[j, d] = k_cache[block_id[j], offs_in_block[j], kv_head, d]
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

        # Flash softmax update.
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)  # rescale running acc/denom; ==1 on first chunk
        l_i = l_i * alpha + tl.sum(p, axis=0)
        # load v[j, d] and fold into the value accumulator
        v_ptrs = (
            v_ptr
            + block_id[:, None] * stride_kn
            + offs_in_block[:, None] * stride_kb
            + kv_head * stride_kh
            + d_off[None, :] * stride_kd
        )
        v = tl.load(v_ptrs, mask=col_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # Normalize and store.
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out_vec = acc / l_safe
    tl.store(
        out_ptr + b * stride_ob + h * stride_oh + d_off * stride_od,
        out_vec.to(out_ptr.dtype.element_ty),
        mask=d_mask,
    )


def paged_attention_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    scale: float,
    workspace=None,
) -> torch.Tensor:
    """Host launcher for the batched paged GQA decode kernel.

    Args match the reference (:func:`xkernels.ops.attention.paged_attention.paged_attention_decode_ref`);
    returns ``out [B, H_q, D]``.

    ``workspace`` (optional :class:`PagedAttentionWorkspace`): if provided and
    matching, write into its ``out`` buffer (a ``[:B]`` slice when the buffer
    was allocated at ``B_max >= B``) instead of allocating. Enables graph
    capture (issue #52).
    """
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_table = block_table.contiguous().to(torch.int32)
    seq_lens = seq_lens.contiguous().to(torch.int32)

    B, H_q, D = q.shape
    _Nb, block_size, H_kv, _D = k_cache.shape
    if H_q % H_kv != 0:
        raise ValueError(
            f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
        )
    group = H_q // H_kv
    max_blocks = block_table.shape[1]

    if workspace is not None:
        if not workspace.matches(B, H_q, D, device=q.device, dtype=q.dtype):
            raise ValueError(
                f"workspace buffer is shape {tuple(workspace.out.shape)} on "
                f"{workspace.out.device}/{workspace.out.dtype}; need >= ({B}, "
                f"{H_q}, {D}) on {q.device}/{q.dtype}."
            )
        out = workspace.out[:B]  # valid slice; strides unchanged
    else:
        out = torch.empty(B, H_q, D, device=q.device, dtype=q.dtype)
    BLOCK_D = triton.next_power_of_2(D)
    # BLOCK_N: how many KV positions per flash chunk. 64 is the vLLM/flash
    # default; the flash reduction is exact for any chunk size (tune later).
    BLOCK_N = 64
    grid = (B, H_q)
    paged_attention_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        out,
        scale,
        H_q,
        H_kv,
        D,
        block_size,
        max_blocks,
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


register("paged_attention", Backend.TRITON)(paged_attention_triton)
