# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public MHC ops: ``hc_prenorm_gemm`` (+ faithful-named in-place wrapper
``tf32_hc_prenorm_gemm``) and the full ``mhc_pre`` / ``mhc_post`` fusions. Each
dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import (
    pre_post_reference,  # noqa: F401  (registers mhc_pre/mhc_post REFERENCE)
    reference,  # noqa: F401  (registers hc_prenorm_gemm REFERENCE)
)


def hc_prenorm_gemm(
    a: torch.Tensor,
    fn: torch.Tensor,
    *,
    n_splits: int,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 MHC hidden-compression prenorm GEMM (issue #36): the GEMM +
    RMS-prenorm squared-sum half of ``mhc_pre``. Portable gfx942 replacement for
    the NVIDIA-only ``deep_gemm.tf32_hc_prenorm_gemm``.

    Computes, in a split-K layout summed over the split axis by the downstream
    TileLang post-fusion:

        gemm_out_mul.sum(0)    == F.linear(a.float(), fn.float())   ( = a @ fn.T )
        gemm_out_sqrsum.sum(0) == (a.float()**2).sum(-1)

    Args:
        a: ``[T, K]`` flattened residual (bf16; fp32 accepted). ``K = hc_mult*hidden``.
        fn: ``[N, K]`` fp32 hidden-compression weight (Linear orientation,
            ``N = hc_mult3 = 2*hc_mult + hc_mult**2``).
        n_splits: number of K-split partials to emit (``>= 1``).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(gemm_out_mul [n_splits, T, N] fp32, gemm_out_sqrsum [n_splits, T] fp32)``.
    """
    return dispatch("hc_prenorm_gemm", a, fn, n_splits=n_splits, backend=backend)


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    n_splits: int,
    *,
    backend: Backend | str = "auto",
) -> None:
    """Upstream-faithful in-place wrapper (the tokenspeed binding target).

    Exact ``deep_gemm.tf32_hc_prenorm_gemm`` positional signature: writes the
    pre-allocated ``gemm_out_mul [n_splits, T, N]`` / ``gemm_out_sqrsum
    [n_splits, T]`` fp32 buffers in place and returns ``None``.
    """
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=backend)
    if gemm_out_mul.shape != mul.shape or gemm_out_sqrsum.shape != sqr.shape:
        raise ValueError(
            f"out buffer shape mismatch: mul {tuple(gemm_out_mul.shape)} vs "
            f"{tuple(mul.shape)}, sqrsum {tuple(gemm_out_sqrsum.shape)} vs "
            f"{tuple(sqr.shape)}"
        )
    gemm_out_mul.copy_(mul)
    gemm_out_sqrsum.copy_(sqr)
    return None


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 MHC ``mhc_pre`` full fusion (issue #44).

    RMS-prenorms the flattened residual, projects through ``fn``, and splits the
    result into the ``pre`` (combine weights), ``post`` (block-update gate) and
    ``comb`` (sinkhorn mixing) heads, returning the ``pre``-weighted residual
    combine as ``layer_input``. Portable gfx942 replacement for the TileLang
    fusion whose ``layer_input`` branch mislowers on AMD.

    Args:
        residual: ``[T, hc_mult, hidden]`` (bf16; fp32 accepted).
        fn: ``[hc_mult3, hc_mult*hidden]`` fp32 weight (``hc_mult3 = 2*hc_mult +
            hc_mult**2``).
        hc_scale: ``[3]`` fp32 per-head scales (pre/post/comb).
        hc_base: ``[hc_mult3]`` fp32 per-channel bias.
        rms_eps: RMS-prenorm epsilon.
        hc_eps: sigmoid/sinkhorn stabilizing epsilon.
        sinkhorn_iters: number of sinkhorn passes (>= 1).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(layer_input [T, hidden] residual-dtype, post [T, hc_mult, 1] fp32,
        comb [T, hc_mult, hc_mult] fp32)``.
    """
    return dispatch(
        "mhc_pre",
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        backend=backend,
    )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """DeepSeek-V4 MHC ``mhc_post`` fusion (issue #44): sinkhorn-mixed residual
    plus gated block update.

        out[t, m, h] = sum_n comb[t, n, m] * residual[t, n, h]
                       + post[t, m] * hidden_states[t, h]

    Args:
        hidden_states: ``[T, hidden]`` block update (bf16; fp32 accepted).
        residual: ``[T, hc_mult, hidden]``.
        post: ``[T, hc_mult, 1]`` (or ``[T, hc_mult]``) gate.
        comb: ``[T, hc_mult, hc_mult]`` sinkhorn mixing.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[T, hc_mult, hidden]`` in ``hidden_states`` dtype.
    """
    return dispatch("mhc_post", hidden_states, residual, post, comb, backend=backend)
