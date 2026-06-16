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

# fp8 encodings the gfx942 CDNA3 MFMA decodes natively (``v_mfma_*_fp8_fp8``).
# Operands in these dtypes hit the fast path; the OCP ``fn`` family instead
# upcasts to an f16 MFMA (measured slower than the portable kernel), so ``auto``
# only routes to the mfma path for fnuz operands.
_FNUZ_FP8 = {
    getattr(torch, n)
    for n in ("float8_e4m3fnuz", "float8_e5m2fnuz")
    if hasattr(torch, n)
}


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
    portable path. ``"auto"`` routes to the mfma fast path only for fnuz operands
    (the gfx942-native fp8 MFMA encoding, where it wins 3-9x); fn operands upcast
    to an f16 MFMA that is slower than the portable kernel, so ``auto`` keeps them
    on the portable path. The mfma path is 128-quant-block only.
    """
    if path not in ("auto", "mfma", "portable"):
        raise ValueError(f"path must be auto|mfma|portable, got {path!r}")
    want_mfma = path == "mfma" or (
        path == "auto"
        and not dot_bf16
        and block == 128
        and a_fp8.dtype in _FNUZ_FP8
    )
    if want_mfma and block == 128:
        return _mfma(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)
    return _portable(
        a_fp8, a_scales, b_fp8, b_scales,
        block=block, out_dtype=out_dtype, dot_bf16=dot_bf16,
    )


register("mm_fp8_blockscale", Backend.TRITON)(mm_fp8_blockscale_triton)
