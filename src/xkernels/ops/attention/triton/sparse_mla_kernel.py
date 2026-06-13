# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton sparse-MLA attention compute for AMD MI300A (gfx942, CDNA3), issue #32.

One program per ``(token, head)``: stream the top-k selected latent KV in
``BLOCK_N`` chunks with online (flash) softmax. The score uses all ``D`` dims; the
value accumulator stores the first ``d_v`` dims (the kv_lora / nope part). An
optional per-head attention **sink** logit folds into the denominator after the
stream and contributes no value. Columns with ``idx < 0`` or beyond
``topk_length`` are masked to ``-inf``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from .sparse_mla_config import resolve_sparse_mla_config

__all__ = ["sparse_mla_attention_triton", "sparse_mla_kernel"]


@triton.jit
def sparse_mla_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    sink_ptr,
    len_ptr,
    out_ptr,
    lse_ptr,
    maxl_ptr,
    sm_scale,
    H,
    Kv,
    topk,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kk,
    stride_kd,
    stride_it,
    stride_ik,
    stride_ot,
    stride_oh,
    stride_od,
    HAS_SINK: tl.constexpr,
    HAS_LEN: tl.constexpr,
    D: tl.constexpr,
    D_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    q = tl.load(
        q_ptr + t * stride_qt + h * stride_qh + d * stride_qd, mask=d_mask, other=0.0
    ).to(tl.float32)
    n_valid = tl.load(len_ptr + t) if HAS_LEN else topk

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start in range(0, topk, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < topk
        idx = tl.load(idx_ptr + t * stride_it + cols * stride_ik, mask=col_mask, other=-1)
        valid = (idx >= 0) & col_mask
        if HAS_LEN:
            valid = valid & (cols < n_valid)
        safe = tl.where(valid, idx, 0)
        kvb = tl.load(
            kv_ptr + safe[:, None] * stride_kk + d[None, :] * stride_kd,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * kvb, axis=1) * sm_scale  # [BLOCK_N]
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.where(valid, tl.exp(scores - m_new), 0.0)
        # Guard the all-masked chunk: m_i == m_new == -inf would make exp(-inf -
        # -inf) = exp(nan). alpha=1 there keeps the (still-zero) running state.
        alpha = tl.where(m_new == -float("inf"), 1.0, tl.exp(m_i - m_new))
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kvb, axis=0)  # [BLOCK_D]
        m_i = m_new

    if HAS_SINK:
        sink = tl.load(sink_ptr + h).to(tl.float32)
        m_new = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.exp(sink - m_new)
        acc = acc * alpha
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe
    dv_mask = d < D_V
    tl.store(
        out_ptr + t * stride_ot + h * stride_oh + d * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=dv_mask,
    )
    lse_val = tl.where(l_i > 0.0, m_i + tl.log(l_safe), -float("inf"))
    tl.store(lse_ptr + t * H + h, lse_val)
    tl.store(maxl_ptr + t * H + h, m_i)


def sparse_mla_attention_triton(
    q,
    kv,
    indices,
    *,
    sm_scale,
    topk_length=None,
    attn_sink=None,
    d_v=None,
):
    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous().to(torch.int32)
    T, H, D = q.shape
    Kv = kv.shape[0]
    topk = indices.shape[1]
    d_v = D if d_v is None else d_v

    out = torch.empty(T, H, d_v, device=q.device, dtype=q.dtype)
    lse = torch.empty(T, H, device=q.device, dtype=torch.float32)
    maxl = torch.empty(T, H, device=q.device, dtype=torch.float32)

    has_sink = attn_sink is not None
    has_len = topk_length is not None
    dummy = torch.empty(1, device=q.device, dtype=torch.float32)
    sink = dummy
    if has_sink:
        s = attn_sink.contiguous().float().reshape(-1)
        sink = s.expand(H).contiguous() if s.numel() == 1 else s[:H].contiguous()
    length = topk_length.contiguous().to(torch.int32) if has_len else dummy.to(torch.int32)

    BLOCK_D = triton.next_power_of_2(D)
    # Perf pass (#39): BLOCK_N + CDNA3 lowering knobs are resolved from a config
    # (env-overridable for the on-device sweep). The default reproduces the #33
    # launch (BLOCK_N=64). BLOCK_N is a pure perf knob — the flash reduction is
    # exact for any chunk size (see sparse_mla_config.py).
    cfg = resolve_sparse_mla_config()
    # AMD-only lowering kwargs: read by the Triton AMD backend, ignored elsewhere
    # (and under TRITON_INTERPRET=1), so the same call stays portable.
    amd_knobs = {
        "waves_per_eu": int(cfg.get("waves_per_eu", 0)),
        "matrix_instr_nonkdim": int(cfg.get("matrix_instr_nonkdim", 16)),
        "kpack": int(cfg.get("kpack", 2)),
    }
    sparse_mla_kernel[(T, H)](
        q,
        kv,
        indices,
        sink,
        length,
        out,
        lse,
        maxl,
        sm_scale,
        H,
        Kv,
        topk,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HAS_SINK=has_sink,
        HAS_LEN=has_len,
        D=D,
        D_V=d_v,
        BLOCK_D=BLOCK_D,
        BLOCK_N=int(cfg["BLOCK_N"]),
        num_warps=int(cfg.get("num_warps", 4)),
        num_stages=int(cfg.get("num_stages", 1)),
        **amd_knobs,
    )
    return out, lse, maxl


register("sparse_mla_attention", Backend.TRITON)(sparse_mla_attention_triton)
