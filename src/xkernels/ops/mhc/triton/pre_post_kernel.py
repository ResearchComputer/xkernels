# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton ``mhc_pre`` / ``mhc_post`` full fusion for AMD MI300A (gfx942), issue #44.

The NVIDIA TileLang ``mhc_pre`` fusion mislowers the ``layer_input`` combine on
gfx942: it copies a strided ``(hc_mult, hidden_block)`` slab out of
``residual[t, 0, off]`` (which needs a ``hidden``-element stride between the
``hc_mult`` rows) as a contiguous block, so ~97% of ``layer_input`` is wrong and
every layer's hidden state is corrupted. These Triton kernels recompute the same
math with explicit strides so element ``residual[t, n, h]`` is always addressed
as ``t*stride_rt + n*stride_rn + h*stride_rh``.

``mhc_pre`` runs one program per token: it streams ``K = hc_mult*hidden`` in
``BLOCK_K`` chunks to accumulate both the prenorm projection ``mixes[j] =
sum_k x[k] * fn[j, k]`` and ``sum_k x[k]**2`` (RMS prenorm), forms the
``pre``/``post`` sigmoid gates and the ``comb`` sinkhorn matrix (``hc_mult`` is
tiny, e.g. 4), then streams ``hidden`` in ``BLOCK_H`` chunks for the
``pre``-weighted residual combine ``layer_input[t, h] = sum_n pre[n] *
residual[t, n, h]``.

``mhc_post`` runs one program per ``(token, hidden-tile)``: it loads the
``hc_mult x hc_mult`` ``comb`` and the ``hc_mult`` ``post`` gates and computes
``out[t, m, h] = sum_n comb[t, n, m] * residual[t, n, h] + post[t, m] *
hidden[t, h]``. All math in fp32 (CDNA3 has no TF32).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = [
    "mhc_pre_triton",
    "mhc_post_triton",
    "mhc_pre_kernel",
    "mhc_post_kernel",
]


@triton.jit
def mhc_pre_kernel(
    residual_ptr,  # [T, hc_mult, hidden] bf16/fp32
    fn_ptr,  # [hc_mult3, K] fp32, K = hc_mult*hidden
    hc_scale_ptr,  # [3] fp32
    hc_base_ptr,  # [hc_mult3] fp32
    li_ptr,  # [T, hidden] out
    post_ptr,  # [T, hc_mult] fp32 out
    comb_ptr,  # [T, hc_mult*hc_mult] fp32 out
    T,
    hidden,
    K,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    stride_rt,
    stride_rn,
    stride_rh,
    stride_fnj,
    stride_fnk,
    stride_lt,
    stride_lh,
    stride_pt,
    stride_ct,
    HC_MULT: tl.constexpr,
    HC_MULT_PAD: tl.constexpr,
    HC_MULT_PAD2: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)

    # hc_mult (and hc_mult**2) are NOT powers of 2 in general (DeepSeek-V4 and
    # the sweep use hc_mult=3), but tl.arange requires a power-of-2 length. Pad
    # to the next power of 2 and mask the out-of-range elements everywhere they
    # feed a reduction or a store (issue #83).
    arn = tl.arange(0, HC_MULT_PAD)  # [HC_MULT_PAD]
    mask_n = arn < HC_MULT
    j_pre = arn  # rows 0..hc_mult of fn
    j_post = arn + HC_MULT  # rows hc_mult..2*hc_mult
    # comb is a [hc_mult, hc_mult] sinkhorn mixing matrix. Pad it to HC_MULT_PAD**2
    # (a power of 2) but lay the valid hc_mult**2 entries at stride HC_MULT_PAD --
    # i.e. flat index (r*HC_MULT_PAD + c) -- so reshape([HC_MULT_PAD, HC_MULT_PAD])
    # is ALIGNED with the top-left hc_mult x hc_mult mask. (A naive pad to
    # next_pow2(hc_mult**2) and reshape leaves the valid entries on a HC_MULT
    # stride, misaligning them with the mask for non-power-of-2 hc_mult -- #83.)
    arn2d = tl.arange(0, HC_MULT_PAD)[:, None]  # n (source row) -- used for store
    acm2d = tl.arange(0, HC_MULT_PAD)[None, :]  # m (dest col)
    comb_valid = mask_n[:, None] & mask_n[None, :]  # [HC_MULT_PAD, HC_MULT_PAD]
    arnn = tl.arange(0, HC_MULT_PAD2)  # flat [HC_MULT_PAD**2]
    arnn_row = arnn // HC_MULT_PAD
    arnn_col = arnn % HC_MULT_PAD
    mask_nn = (arnn_row < HC_MULT) & (arnn_col < HC_MULT)
    j_comb = 2 * HC_MULT + arnn_row * HC_MULT + arnn_col  # fn row per padded (r,c)

    # ---- prenorm GEMM: mixes_head[j] = sum_k x[k]*fn[j,k], sqsum = sum_k x[k]^2 ----
    pre_acc = tl.zeros([HC_MULT_PAD], dtype=tl.float32)
    post_acc = tl.zeros([HC_MULT_PAD], dtype=tl.float32)
    comb_acc = tl.zeros([HC_MULT_PAD2], dtype=tl.float32)
    sqsum = tl.zeros([], dtype=tl.float32)
    # x is residual[t] flattened over (hc_mult, hidden) in row-major order:
    # x[k] = residual[t, k // hidden, k % hidden].
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = ks < K
        n_idx = ks // hidden
        h_idx = ks % hidden
        x = tl.load(
            residual_ptr + t * stride_rt + n_idx * stride_rn + h_idx * stride_rh,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        sqsum += tl.sum(x * x, axis=0)
        fn_pre = tl.load(
            fn_ptr + j_pre[:, None] * stride_fnj + ks[None, :] * stride_fnk,
            mask=k_mask[None, :] & mask_n[:, None], other=0.0,
        ).to(tl.float32)
        fn_post = tl.load(
            fn_ptr + j_post[:, None] * stride_fnj + ks[None, :] * stride_fnk,
            mask=k_mask[None, :] & mask_n[:, None], other=0.0,
        ).to(tl.float32)
        fn_comb = tl.load(
            fn_ptr + j_comb[:, None] * stride_fnj + ks[None, :] * stride_fnk,
            mask=k_mask[None, :] & mask_nn[:, None], other=0.0,
        ).to(tl.float32)
        pre_acc += tl.sum(fn_pre * x[None, :], axis=1)
        post_acc += tl.sum(fn_post * x[None, :], axis=1)
        comb_acc += tl.sum(fn_comb * x[None, :], axis=1)

    rsqrt = tl.rsqrt(sqsum / K + rms_eps)
    pre_acc = pre_acc * rsqrt
    post_acc = post_acc * rsqrt
    comb_acc = comb_acc * rsqrt

    scale0 = tl.load(hc_scale_ptr + 0).to(tl.float32)
    scale1 = tl.load(hc_scale_ptr + 1).to(tl.float32)
    scale2 = tl.load(hc_scale_ptr + 2).to(tl.float32)
    base_pre = tl.load(hc_base_ptr + j_pre, mask=mask_n, other=0.0).to(tl.float32)
    base_post = tl.load(hc_base_ptr + j_post, mask=mask_n, other=0.0).to(tl.float32)
    base_comb = tl.load(hc_base_ptr + j_comb, mask=mask_nn, other=0.0).to(tl.float32)

    pre = tl.sigmoid(pre_acc * scale0 + base_pre) + hc_eps  # [HC_MULT_PAD]
    post = tl.sigmoid(post_acc * scale1 + base_post) * 2.0  # [HC_MULT_PAD]
    tl.store(post_ptr + t * stride_pt + arn, post, mask=mask_n)

    # ---- comb: sinkhorn over [hc_mult, hc_mult] (row = source, col = dest) ----
    # comb_acc is [HC_MULT_PAD**2] with valid entries at stride HC_MULT_PAD, so
    # reshape to [HC_MULT_PAD, HC_MULT_PAD] is ALIGNED with comb_valid (top-left
    # hc_mult x hc_mult). Force padding to -inf for the row softmax then to 0 for
    # every reduction, so padding never perturbs the valid rows/cols.
    cm = tl.reshape(comb_acc * scale2 + base_comb, [HC_MULT_PAD, HC_MULT_PAD])
    cm = tl.where(comb_valid, cm, -float("inf"))

    # softmax over last dim (row)
    row_max = tl.max(cm, axis=1)
    cm = tl.exp(cm - row_max[:, None])
    cm = tl.where(comb_valid, cm, 0.0)  # padding cols -> 0; kills padding-row nan
    row_sum = tl.sum(cm, axis=1)
    cm = cm / row_sum[:, None] + hc_eps
    cm = tl.where(comb_valid, cm, 0.0)  # padding rows -> 0 before col_sum
    # first column normalization
    col_sum = tl.sum(cm, axis=0)
    cm = cm / (col_sum[None, :] + hc_eps)
    cm = tl.where(comb_valid, cm, 0.0)
    # remaining (iters-1) alternating row/col passes
    for _ in range(sinkhorn_iters - 1):
        row_sum = tl.sum(cm, axis=1)
        cm = cm / (row_sum[:, None] + hc_eps)
        cm = tl.where(comb_valid, cm, 0.0)
        col_sum = tl.sum(cm, axis=0)
        cm = cm / (col_sum[None, :] + hc_eps)
        cm = tl.where(comb_valid, cm, 0.0)
    # store the valid hc_mult x hc_mult block at its true flat stride (HC_MULT)
    tl.store(comb_ptr + t * stride_ct + arn2d * HC_MULT + acm2d, cm, mask=comb_valid)

    # ---- layer_input[t, h] = sum_n pre[n] * residual[t, n, h] (the defect branch) ----
    for h0 in range(0, hidden, BLOCK_H):
        hs = h0 + tl.arange(0, BLOCK_H)  # [BLOCK_H]
        h_mask = hs < hidden
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for n in tl.static_range(HC_MULT):
            xn = tl.load(
                residual_ptr + t * stride_rt + n * stride_rn + hs * stride_rh,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            pre_n = tl.sum(tl.where(arn == n, pre, 0.0), axis=0)
            acc += pre_n * xn
        tl.store(li_ptr + t * stride_lt + hs * stride_lh, acc, mask=h_mask)


@triton.jit
def mhc_post_kernel(
    hidden_ptr,  # [T, hidden]
    residual_ptr,  # [T, hc_mult, hidden]
    post_ptr,  # [T, hc_mult] fp32
    comb_ptr,  # [T, hc_mult, hc_mult] fp32
    out_ptr,  # [T, hc_mult, hidden]
    T,
    hidden,
    stride_ht,
    stride_hh,
    stride_rt,
    stride_rn,
    stride_rh,
    stride_pt,
    stride_ct,
    stride_cn,
    stride_cm,
    stride_ot,
    stride_om,
    stride_oh,
    HC_MULT: tl.constexpr,
    HC_MULT_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    VEC: tl.constexpr,
):
    t = tl.program_id(0)
    hb = tl.program_id(1)
    # Vectorized memory access: each thread handles VEC consecutive elements.
    thread_idx = tl.arange(0, BLOCK_H // VEC)
    hs = hb * BLOCK_H + thread_idx[:, None] * VEC + tl.arange(0, VEC)[None, :]
    h_mask = hs < hidden

    # hc_mult may be non-power-of-2 (issue #83): pad arn and mask the padding so
    # it contributes 0 to the reduce over n and is not stored.
    arn = tl.arange(0, HC_MULT_PAD)
    mask_n = arn < HC_MULT
    # load hidden[t, h] tile [BLOCK_H//VEC, VEC]
    hid = tl.load(
        hidden_ptr + t * stride_ht + hs * stride_hh, mask=h_mask, other=0.0
    ).to(tl.float32)
    # residual tile [HC_MULT_PAD, BLOCK_H//VEC, VEC]
    res = tl.load(
        residual_ptr
        + t * stride_rt
        + arn[:, None, None] * stride_rn
        + hs[None, :, :] * stride_rh,
        mask=h_mask[None, :, :] & mask_n[:, None, None],
        other=0.0,
    ).to(tl.float32)
    # comb[t, n, m] and post[t, m]  (padding n/m masked -> 0)
    cm_mask2 = mask_n[:, None] & mask_n[None, :]
    comb = tl.load(
        comb_ptr
        + t * stride_ct
        + arn[:, None] * stride_cn
        + arn[None, :] * stride_cm,
        mask=cm_mask2,
        other=0.0,
    ).to(tl.float32)  # [n, m] = [HC_MULT_PAD, HC_MULT_PAD]
    post = tl.load(
        post_ptr + t * stride_pt + arn, mask=mask_n, other=0.0
    ).to(tl.float32)  # [HC_MULT_PAD]

    # out[m, h] = sum_n comb[n, m] * res[n, h] + post[m] * hid[h]
    # mixed[m, h] = sum_n comb[n, m] * res[n, h]  (padding n: res=0 -> contributes 0)
    mixed = tl.sum(comb[:, :, None, None] * res[:, None, :, :], axis=0)  # [m, BLOCK_H//VEC, VEC]
    out = mixed + post[:, None, None] * hid[None, :, :]  # [HC_MULT_PAD, BLOCK_H//VEC, VEC]
    tl.store(
        out_ptr + t * stride_ot + arn[:, None, None] * stride_om + hs[None, :, :] * stride_oh,
        out,
        mask=h_mask[None, :, :] & mask_n[:, None, None],
    )


def mhc_pre_triton(
    residual, fn, hc_scale, hc_base, rms_eps, hc_eps, sinkhorn_iters
):
    if residual.dim() != 3:
        raise ValueError(
            f"residual must be [T, hc_mult, hidden], got {tuple(residual.shape)}"
        )
    num_tokens, hc_mult, hidden = residual.shape
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    K = hc_mult * hidden
    if fn.shape != (hc_mult3, K):
        raise ValueError(f"fn must be [{hc_mult3}, {K}], got {tuple(fn.shape)}")
    fn = fn.contiguous()
    hc_scale = hc_scale.contiguous().float()
    hc_base = hc_base.contiguous().float()

    li = torch.empty(num_tokens, hidden, device=residual.device, dtype=residual.dtype)
    post = torch.empty(num_tokens, hc_mult, device=residual.device, dtype=torch.float32)
    comb = torch.empty(
        num_tokens, hc_mult2, device=residual.device, dtype=torch.float32
    )
    if num_tokens == 0:
        return (
            li,
            post.view(0, hc_mult, 1),
            comb.view(0, hc_mult, hc_mult),
        )

    block_k = min(triton.next_power_of_2(K), 256)
    block_h = min(triton.next_power_of_2(hidden), 1024)
    mhc_pre_kernel[(num_tokens,)](
        residual,
        fn,
        hc_scale,
        hc_base,
        li,
        post,
        comb,
        num_tokens,
        hidden,
        K,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        fn.stride(0),
        fn.stride(1),
        li.stride(0),
        li.stride(1),
        post.stride(0),
        comb.stride(0),
        HC_MULT=hc_mult,
        HC_MULT_PAD=triton.next_power_of_2(hc_mult),
        HC_MULT_PAD2=triton.next_power_of_2(hc_mult) ** 2,
        BLOCK_K=block_k,
        BLOCK_H=block_h,
    )
    return li, post.view(num_tokens, hc_mult, 1), comb.view(num_tokens, hc_mult, hc_mult)


def mhc_post_triton(hidden_states, residual, post, comb):
    if residual.dim() != 3:
        raise ValueError(
            f"residual must be [T, hc_mult, hidden], got {tuple(residual.shape)}"
        )
    num_tokens, hc_mult, hidden = residual.shape
    out = torch.empty(
        num_tokens, hc_mult, hidden, device=hidden_states.device, dtype=hidden_states.dtype
    )
    if num_tokens == 0:
        return out

    hidden_states = hidden_states.reshape(num_tokens, hidden).contiguous()
    post = post.reshape(num_tokens, hc_mult).contiguous().float()
    comb = comb.reshape(num_tokens, hc_mult, hc_mult).contiguous().float()

    block_h = min(triton.next_power_of_2(hidden), 1024)
    vec = 4 if block_h >= 4 else (2 if block_h == 2 else 1)
    grid = (num_tokens, triton.cdiv(hidden, block_h))
    mhc_post_kernel[grid](
        hidden_states,
        residual,
        post,
        comb,
        out,
        num_tokens,
        hidden,
        hidden_states.stride(0),
        hidden_states.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post.stride(0),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        HC_MULT=hc_mult,
        HC_MULT_PAD=triton.next_power_of_2(hc_mult),
        BLOCK_H=block_h,
        VEC=vec,
    )
    return out


register("mhc_pre", Backend.TRITON)(mhc_pre_triton)
register("mhc_post", Backend.TRITON)(mhc_post_triton)
