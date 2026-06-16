# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 fp8 block-scale dense GEMM (issue #38).

Ships ``mm_fp8_blockscale`` — a portable gfx942 replacement for the NVIDIA-only
``triton_mm_fp8_blockscale`` / ``deep_gemm_mm_fp8_blockscale``. The only kernel
previously selectable on gfx942 was the slow ``torch_mm_fp8_blockscale`` reference
(full fp32 materialization + dense matmul, no MFMA), which dominates the MLA
projection hot path on both prefill and decode.

Also re-exports the quant helpers (``per_token_group_quant_fp8`` /
``per_block_quant_fp8``) used to produce the fp8 block-scale operands.
"""
from .interface import mm_fp8_blockscale
from .reference import (
    FP8_BLOCK,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)

# Import the Triton backend for its registration side effect (optional). Routed
# through the optional ``_triton_compat`` redirect so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import mm_fp8_blockscale_kernel  # noqa: F401
except Exception:
    pass

__all__ = [
    "mm_fp8_blockscale",
    "per_token_group_quant_fp8",
    "per_block_quant_fp8",
    "FP8_BLOCK",
]
