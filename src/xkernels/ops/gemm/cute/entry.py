# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``Backend.CUDA`` registration for ``mm_fp8_blockscale`` via the CUTE DSL.

Signature matches the triton/reference entry: ``(a_fp8, a_scales, b_fp8,
b_scales, *, block, out_dtype, dot_bf16=, path=)``. Only the portable path is
honored on sm_121 (see the Impl Card): dequant both operands to fp32 in torch
(bit-identical to ``mm_fp8_blockscale_ref``), run the CUTE fp32 GEMM, cast to
``out_dtype``. ``path``/``dot_bf16`` are accepted for signature parity and
ignored (the portable path is always exact fp32).
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .mm_fp8_blockscale_kernel import fp32_matmul_cute

__all__ = ["mm_fp8_blockscale_cute"]


def _dequant_a(a_fp8: torch.Tensor, a_scales: torch.Tensor, block: int) -> torch.Tensor:
    """A [M,K] fp8 -> fp32, scaled per token-group along K (one scale per ``block``)."""
    M, K = a_fp8.shape
    a = a_fp8.to(torch.float32)
    scales = (
        a_scales.to(torch.float32)
        .repeat_interleave(block, dim=1)[:, :K]
    )
    return a * scales


def _dequant_b(b_fp8: torch.Tensor, b_scales: torch.Tensor, block: int) -> torch.Tensor:
    """B [N,K] fp8 -> fp32, scaled per ``block``x``block`` tile (Linear orientation)."""
    N, K = b_fp8.shape
    b = b_fp8.to(torch.float32)
    scales = (
        b_scales.to(torch.float32)
        .repeat_interleave(block, dim=0)[:N]
        .repeat_interleave(block, dim=1)[:, :K]
    )
    return b * scales


def mm_fp8_blockscale_cute(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,
    path: str = "portable",
) -> torch.Tensor:
    """fp8 block-scale GEMM via CUTE DSL (portable fp32 path) on sm_121.

    See module docstring. ``path`` and ``dot_bf16`` are accepted for signature
    parity with the triton backend; on sm_121 only the exact-fp32 portable path
    is honored (native fp8 MMA needs CTK >= 13.1).
    """
    a = _dequant_a(a_fp8, a_scales, block)
    b = _dequant_b(b_fp8, b_scales, block)
    out = fp32_matmul_cute(a, b)  # fp32 [M,N] = a @ b.T
    return out.to(out_dtype)


# Only register on an NVIDIA build (the CUTE DSL is NVIDIA-only). On AMD, the
# triton/reference card handles the op; this module simply doesn't register.
if detect_vendor() == "nvidia":
    register("mm_fp8_blockscale", Backend.CUDA)(mm_fp8_blockscale_cute)
