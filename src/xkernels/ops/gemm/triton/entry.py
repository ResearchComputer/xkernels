# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Single ``Backend.TRITON`` registration for ``mm_fp8_blockscale`` on gfx942:
routes between the native fp8 MFMA fast path (#41) and the portable
dequant-then-dot fallback (#40)."""
from __future__ import annotations

import torch

from ...._backends import Backend
from ...._dispatch import register
from .mm_fp8_blockscale_kernel import mm_fp8_blockscale_triton as _portable
from .mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton as _mfma

__all__ = ["mm_fp8_blockscale_triton"]


def mm_fp8_blockscale_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,
    path: str = "auto",
) -> torch.Tensor:
    """Dispatch the gfx942 Triton fp8 block-scale GEMM.

    ``path``: ``"mfma"`` (native fp8 MFMA, #41), ``"portable"`` (dequant-then-dot,
    #40), or ``"auto"``. ``dot_bf16=True`` is a portable-only knob and forces the
    portable path. ``"auto"`` selects the mfma fast path; the portable path is the
    explicit / ``dot_bf16`` / non-128-block fallback.
    """
    if path not in ("auto", "mfma", "portable"):
        raise ValueError(f"path must be auto|mfma|portable, got {path!r}")
    if dot_bf16 or path == "portable" or block != 128:
        # The mfma path is 128-quant-block only; dot_bf16 is a portable-only knob.
        return _portable(
            a_fp8, a_scales, b_fp8, b_scales,
            block=block, out_dtype=out_dtype, dot_bf16=dot_bf16,
        )
    return _mfma(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)


register("mm_fp8_blockscale", Backend.TRITON)(mm_fp8_blockscale_triton)
