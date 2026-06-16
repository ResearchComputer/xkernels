# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public fp8 block-scale dense-GEMM op (``mm_fp8_blockscale``): dispatches to a
registered backend (issue #38)."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers mm_fp8_blockscale REFERENCE)
from .reference import FP8_BLOCK


def mm_fp8_blockscale(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = FP8_BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,
    path: str = "auto",
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """DeepSeek-V4 fp8 block-scale dense GEMM (issue #38): a portable gfx942
    replacement for the NVIDIA-only ``triton_mm_fp8_blockscale`` /
    ``deep_gemm_mm_fp8_blockscale``. On gfx942 the only previously selectable
    kernel was the slow ``torch_mm_fp8_blockscale`` reference (full fp32
    materialization + dense matmul, no MFMA).

    Computes ``out[M, N] = A_deq @ B_deq.T`` where ``A`` is per-token-group fp8
    e4m3 (``A_scales [M, ceil(K/block)]``) and ``B`` is per-block fp8 e4m3
    (``B_scales [ceil(N/block), ceil(K/block)]``), accumulating in fp32.

    Args:
        a_fp8: ``[M, K]`` activation, ``torch.float8_e4m3fn``.
        a_scales: ``[M, ceil(K/block)]`` fp32 per-token-group scales.
        b_fp8: ``[N, K]`` weight, ``torch.float8_e4m3fn`` (Linear orientation).
        b_scales: ``[ceil(N/block), ceil(K/block)]`` fp32 per-block scales.
        block: group/tile size along each quantized axis (default 128).
        out_dtype: output dtype (default bf16).
        dot_bf16: Triton-backend only, **opt-in** (default False). When True,
            casts the block-scaled operands to bf16 so ``tl.dot`` runs on the
            CDNA3 bf16 MFMA path (fp32 accumulate) — faster, but only ~bf16-bit
            accurate, so it can exceed a tight per-element tolerance on
            small-/near-zero outputs. Default False keeps the exact-fp32 dot.
            Ignored by the reference backend.
        path: Triton-backend only. ``"mfma"`` (native fp8 MFMA fast path, #41),
            ``"portable"`` (dequant-then-dot, #40), or ``"auto"`` (default ->
            mfma). ``dot_bf16=True`` forces the portable path. Ignored by the
            reference backend.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``out [M, N]`` of ``out_dtype``.
    """
    return dispatch(
        "mm_fp8_blockscale",
        a_fp8,
        a_scales,
        b_fp8,
        b_scales,
        block=block,
        out_dtype=out_dtype,
        dot_bf16=dot_bf16,
        path=path,
        backend=backend,
    )
