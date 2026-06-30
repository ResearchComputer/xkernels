# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``Backend.CUDA`` registration for ``moe_sum_reduce`` via the CUTE DSL.

Signature matches the triton/reference entry:
``(y, w=None, routed_scaling_factor=1.0) -> out``.
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .sum_reduce_kernel import moe_sum_reduce_cute

__all__ = ["moe_sum_reduce_cute_entry"]


def moe_sum_reduce_cute_entry(
    y: torch.Tensor,
    w: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    """Weighted top-k reduction via CUTE DSL (fp32 path) on sm_121.

    ``y`` is upcast to fp32 on the host (bit-identical to the reference's
    ``y.float()``); the kernel runs pure fp32; the result is cast back to
    ``y.dtype``. See ``sum_reduce_kernel.py`` for the design.
    """
    if w is None:
        # Plain sum (weight == 1): the reference folds this as ``yf * 1``.
        w = torch.ones(y.shape[0], y.shape[1], device=y.device, dtype=torch.float32)
    out = moe_sum_reduce_cute(y, w, routed_scaling_factor)
    return out.to(y.dtype)


# NVIDIA-only registration (the CUTE DSL is NVIDIA-only). On AMD the
# triton/reference card serves the op; this module simply doesn't register.
if detect_vendor() == "nvidia":
    register("moe_sum_reduce", Backend.CUDA)(moe_sum_reduce_cute_entry)
