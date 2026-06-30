# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 MHC (multi-head hidden-compression) kernels.

Ships ``hc_prenorm_gemm`` (issue #36): the GEMM + RMS-prenorm squared-sum half of
V4's ``mhc_pre`` — a portable gfx942 replacement for the NVIDIA-only
``deep_gemm.tf32_hc_prenorm_gemm``. Re-exported under that faithful name so
tokenspeed binds it drop-in.

Also ships the full ``mhc_pre`` / ``mhc_post`` fusions (issue #44): a portable
gfx942 replacement for the TileLang fusion whose ``layer_input`` (pre-weighted
residual combine) branch mislowers on AMD (~97% wrong -> incoherent generation).
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .interface import hc_prenorm_gemm, mhc_post, mhc_pre, tf32_hc_prenorm_gemm

# Import the Triton backends for their registration side effect (optional).
# Routed through the optional ``_triton_compat`` redirect so the kernels bind
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
with backend_registration_guard(
    "hc_prenorm_gemm",
    Backend.TRITON,
    source="xkernels.ops.mhc.triton.prenorm_gemm_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import prenorm_gemm_kernel  # noqa: F401

# Import the CUTE DSL (native CUDA) backend for hc_prenorm_gemm (optional).
# NVIDIA-only; gated on `nvidia-cutlass-dsl` (the `cute` extra).
with backend_registration_guard(
    "hc_prenorm_gemm", Backend.CUDA, source="xkernels.ops.mhc.cute.entry"
):  # pragma: no cover - requires nvidia-cutlass-dsl + NVIDIA GPU
    from .cute import entry  # noqa: F401  (registers CUDA: CUTE fp32 path)

with backend_registration_guard(
    ("mhc_pre", "mhc_post"),
    Backend.TRITON,
    source="xkernels.ops.mhc.triton.pre_post_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import pre_post_kernel  # noqa: F401

__all__ = ["hc_prenorm_gemm", "tf32_hc_prenorm_gemm", "mhc_pre", "mhc_post"]
