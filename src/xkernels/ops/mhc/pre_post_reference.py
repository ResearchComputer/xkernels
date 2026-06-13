# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 MHC full ``mhc_pre`` / ``mhc_post``
fusion (issue #44) — numerical oracle and default (CPU / no-Triton) backend.

Issue #36/#37 shipped only the GEMM + RMS-prenorm-squared-sum half of ``mhc_pre``
(``hc_prenorm_gemm``); the rest of ``mhc_pre`` (prenorm projection split into the
``pre``/``post``/``comb`` heads, sinkhorn, and the ``pre``-weighted residual
combine that produces ``layer_input``) and all of ``mhc_post`` lived in a
TileLang fusion. On gfx942 that TileLang kernel **mislowers the ``layer_input``
combine** (~97% wrong -> incoherent generation). This module ports the full math
to portable torch (oracle) and serves as the reference backend; the Triton
backend in ``triton/pre_post_kernel.py`` is the on-device gfx942 path.

``mhc_pre`` computes, per token ``t`` with ``hc_mult`` blocks of ``hidden``:

    x       = residual[t].flatten()                       # [hc_mult*hidden]
    rsqrt   = rsqrt(mean(x**2) + rms_eps)
    mixes   = (x @ fn.T) * rsqrt                           # [hc_mult3]
    pre     = sigmoid(mixes_pre  * hc_scale[0] + base) + hc_eps      # [hc_mult]
    post    = sigmoid(mixes_post * hc_scale[1] + base) * 2.0         # [hc_mult]
    comb    = sinkhorn(mixes_comb * hc_scale[2] + base, iters, hc_eps) # [hc_mult,hc_mult]
    layer_input[t, h] = sum_n pre[n] * residual[t, n, h]  # the defect branch

``mhc_post`` computes:

    out[t, m, h] = sum_n comb[t, n, m] * residual[t, n, h] + post[t, m] * hidden[t, h]

where ``fn`` has shape ``[hc_mult3, hc_mult*hidden]`` (Linear orientation) and
``hc_mult3 = 2*hc_mult + hc_mult**2``. All math in fp32 (CDNA3 has no TF32; the
parity target is this reference, not NVIDIA bit-equality).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["mhc_pre_ref", "mhc_post_ref", "mhc_sinkhorn"]


def mhc_sinkhorn(mixes: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """Sinkhorn normalization matching the TileLang ``comb`` branch.

    Softmax over the last (row) dim, then alternating column/row normalization for
    ``iters`` total passes; ``eps`` stabilizes each division. ``mixes`` is
    ``[..., hc_mult, hc_mult]`` (row = source block, col = dest block).
    """
    mixes = torch.softmax(mixes, dim=-1) + eps
    mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        mixes = mixes / (mixes.sum(dim=-1, keepdim=True) + eps)
        mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    return mixes


def mhc_pre_ref(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MHC ``mhc_pre`` reference. See module docstring.

    Args:
        residual: ``[T, hc_mult, hidden]`` (bf16; fp32 accepted).
        fn: ``[hc_mult3, hc_mult*hidden]`` fp32 hidden-compression weight.
        hc_scale: ``[3]`` fp32 per-head scales (pre/post/comb).
        hc_base: ``[hc_mult3]`` fp32 per-channel bias.
        rms_eps: RMS-prenorm epsilon.
        hc_eps: sigmoid/sinkhorn stabilizing epsilon.
        sinkhorn_iters: number of sinkhorn passes (>= 1).

    Returns:
        ``(layer_input [T, hidden] residual-dtype, post [T, hc_mult, 1] fp32,
        comb [T, hc_mult, hc_mult] fp32)``.
    """
    if residual.dim() != 3:
        raise ValueError(f"residual must be [T, hc_mult, hidden], got {tuple(residual.shape)}")
    num_tokens, hc_mult, hidden = residual.shape
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    if fn.shape != (hc_mult3, hc_mult * hidden):
        raise ValueError(
            f"fn must be [{hc_mult3}, {hc_mult * hidden}], got {tuple(fn.shape)}"
        )
    if num_tokens == 0:
        return (
            residual.new_empty(0, hidden),
            torch.empty(0, hc_mult, 1, dtype=torch.float32, device=residual.device),
            torch.empty(0, hc_mult, hc_mult, dtype=torch.float32, device=residual.device),
        )

    x = residual.reshape(num_tokens, hc_mult * hidden).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    pre_raw, post_raw, comb_raw = torch.split(mixes, [hc_mult, hc_mult, hc_mult2], dim=-1)
    pre_base, post_base, comb_base = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult2], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * hc_scale[1].float() + post_base) * 2.0).unsqueeze(-1)
    comb = mhc_sinkhorn(
        comb_raw.reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2].float()
        + comb_base.reshape(1, hc_mult, hc_mult),
        sinkhorn_iters,
        hc_eps,
    )
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(residual.dtype), post, comb


def mhc_post_ref(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """MHC ``mhc_post`` reference: sinkhorn-mixed residual + gated block update.

    Args:
        hidden_states: ``[T, hidden]`` block update (bf16; fp32 accepted).
        residual: ``[T, hc_mult, hidden]``.
        post: ``[T, hc_mult, 1]`` (or ``[T, hc_mult]``) gate.
        comb: ``[T, hc_mult, hc_mult]`` sinkhorn mixing.

    Returns:
        ``[T, hc_mult, hidden]`` in ``hidden_states`` dtype.
    """
    if residual.dim() != 3:
        raise ValueError(f"residual must be [T, hc_mult, hidden], got {tuple(residual.shape)}")
    num_tokens, hc_mult, hidden = residual.shape
    if num_tokens == 0:
        return hidden_states.new_empty(0, hc_mult, hidden)
    p = post.reshape(num_tokens, hc_mult, 1).float()
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = p * hidden_states.reshape(num_tokens, hidden).float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)


register("mhc_pre", Backend.REFERENCE)(mhc_pre_ref)
register("mhc_post", Backend.REFERENCE)(mhc_post_ref)
