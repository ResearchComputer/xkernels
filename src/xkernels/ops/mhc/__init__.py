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
from .interface import hc_prenorm_gemm, mhc_post, mhc_pre, tf32_hc_prenorm_gemm

# Import the Triton backends for their registration side effect (optional).
# Routed through the optional ``_triton_compat`` redirect so the kernels bind
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import (
            pre_post_kernel,  # noqa: F401
            prenorm_gemm_kernel,  # noqa: F401
        )
except Exception:
    pass

__all__ = ["hc_prenorm_gemm", "tf32_hc_prenorm_gemm", "mhc_pre", "mhc_post"]
