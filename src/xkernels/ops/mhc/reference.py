# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 MHC hidden-compression prenorm GEMM
(issue #36) — numerical oracle and default (CPU / no-Triton) backend on gfx942.

This is the GEMM + RMS-prenorm-squared-sum half of V4's ``mhc_pre``. Upstream
computes it with the NVIDIA-only ``deep_gemm.tf32_hc_prenorm_gemm``; on AMD that
raises. The op takes the flattened residual ``A = residual.view(T, hc_mult*hidden)``
(bf16) and the fp32 hidden-compression weight ``fn`` of shape ``[N, K]`` (Linear
orientation, ``N = hc_mult3 = 2*hc_mult + hc_mult**2``), and produces, in a
split-K layout consumed by the TileLang post-fusion:

    gemm_out_mul[s, t, :]    partial of  F.linear(A, fn)[t]  ( = A @ fn.T )
    gemm_out_sqrsum[s, t]    partial of  (A.float()**2).sum(-1)[t]   (RMS prenorm)

summed over the split axis ``s``. The TileLang kernel only ever sums across
splits, so any complete disjoint K-partition is valid; the reference uses the
trivial one (full result in split 0, zeros elsewhere). All math in fp32 (CDNA3
has no TF32; the parity target is this reference, not NVIDIA bit-equality).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["hc_prenorm_gemm_ref"]


def hc_prenorm_gemm_ref(
    a: torch.Tensor,
    fn: torch.Tensor,
    *,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MHC prenorm GEMM reference. See module docstring.

    Args:
        a: ``[T, K]`` flattened residual (bf16; fp32 accepted). ``K = hc_mult*hidden``.
        fn: ``[N, K]`` fp32 hidden-compression weight (Linear orientation).
        n_splits: number of K-split partials to emit (``>= 1``).

    Returns:
        ``(gemm_out_mul [n_splits, T, N] fp32, gemm_out_sqrsum [n_splits, T] fp32)``
        with ``gemm_out_mul.sum(0) == F.linear(a.float(), fn.float())`` and
        ``gemm_out_sqrsum.sum(0) == (a.float()**2).sum(-1)``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    T, K = a.shape
    N = fn.shape[0]
    if fn.shape[1] != K:
        raise ValueError(f"fn must be [N, K] with K={K}, got {tuple(fn.shape)}")
    af = a.float()
    gemm_out_mul = af.new_zeros(n_splits, T, N)
    gemm_out_sqrsum = af.new_zeros(n_splits, T)
    if T > 0:
        gemm_out_mul[0] = F.linear(af, fn.float())
        gemm_out_sqrsum[0] = (af * af).sum(dim=-1)
    return gemm_out_mul, gemm_out_sqrsum


register("hc_prenorm_gemm", Backend.REFERENCE)(hc_prenorm_gemm_ref)
