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
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .interface import mm_fp8_blockscale
from .reference import (
    FP8_BLOCK,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    preferred_fp8_dtype,
)

# Import the Triton backend for its registration side effect (optional). Routed
# through the optional ``_triton_compat`` redirect so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
with backend_registration_guard(
    "mm_fp8_blockscale", Backend.TRITON, source="xkernels.ops.gemm.triton.entry"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import entry  # noqa: F401  (registers TRITON: mfma + portable)

# Register the DSL-generated Triton backends for the fp8 quant helpers (issue
# #57) for their dispatch side effect. Unlike the hand-written mm_fp8_blockscale
# kernel above (which imports ``triton`` at top level, so it routes through
# ``triton_import_ctx``), the DSL path builds only a lazy host launcher at import
# time -- no triton import happens until the kernel is actually called -- so no
# import-context redirect is needed here (same pattern as ops.norm #66). The
# cards operate on the grouped [G,B] view; the public [M,K] helpers stay the
# reference path (reshape+dtype glue is a tracked follow-up).
with backend_registration_guard(
    ("per_token_group_quant_fp8", "per_block_quant_fp8"),
    Backend.TRITON,
    source="xkernels.ops.gemm.triton.quant_kernel",
):  # pragma: no cover - hardware dependent
    from .triton import quant_kernel  # noqa: F401

# Import the CUTE DSL (native CUDA) backend for its registration side effect
# (optional). NVIDIA-only; gated on `nvidia-cutlass-dsl` (the `cute` extra). On a
# box without the DSL (CI, AMD) the guard records the failure and the op is
# still served by the triton/reference card.
with backend_registration_guard(
    "mm_fp8_blockscale", Backend.CUDA, source="xkernels.ops.gemm.cute.entry"
):  # pragma: no cover - requires nvidia-cutlass-dsl + NVIDIA GPU
    from .cute import entry  # noqa: F401  (registers CUDA: CUTE fp32 path)

__all__ = [
    "mm_fp8_blockscale",
    "per_token_group_quant_fp8",
    "per_block_quant_fp8",
    "preferred_fp8_dtype",
    "FP8_BLOCK",
]
