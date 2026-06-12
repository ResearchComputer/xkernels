# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 MHC (multi-head hidden-compression) kernels.

Ships ``hc_prenorm_gemm`` (issue #36): the GEMM + RMS-prenorm squared-sum half of
V4's ``mhc_pre`` — a portable gfx942 replacement for the NVIDIA-only
``deep_gemm.tf32_hc_prenorm_gemm``. Re-exported under that faithful name so
tokenspeed binds it drop-in; the TileLang post-fusion that consumes its outputs
is already portable on AMD and is untouched.
"""
from .interface import hc_prenorm_gemm, tf32_hc_prenorm_gemm

# Import the Triton backend for its registration side effect (optional). Routed
# through the optional ``_triton_compat`` redirect so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import prenorm_gemm_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["hc_prenorm_gemm", "tf32_hc_prenorm_gemm"]
