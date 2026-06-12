# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public MHC op (``hc_prenorm_gemm``) + faithful-named in-place wrapper
(``tf32_hc_prenorm_gemm``): dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers hc_prenorm_gemm REFERENCE)


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
