# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``Backend.CUDA`` registration for ``hc_prenorm_gemm`` via the CUTE DSL.

Signature matches the triton/reference entry:
``(a, fn, *, n_splits) -> (gemm_out_mul, gemm_out_sqrsum)``.
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .prenorm_gemm_kernel import hc_prenorm_gemm_cute

__all__ = ["hc_prenorm_gemm_cute"]


# NVIDIA-only registration (the CUTE DSL is NVIDIA-only). On AMD the
# triton/reference card serves the op; this module simply doesn't register.
if detect_vendor() == "nvidia":
    register("hc_prenorm_gemm", Backend.CUDA)(hc_prenorm_gemm_cute)
