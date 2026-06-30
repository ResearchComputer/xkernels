# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``Backend.CUDA`` registration for ``dual_rmsnorm`` via the CUTE DSL.

Signature matches the triton/reference entry: ``(x1, w1, x2, w2, *, eps=1e-6)``
returning ``(out1, out2)``. Each latent is normalized by an independent launch of
the single-row RMSNorm CUTE primitive (``rmsnorm_cute``); the two norms are
numerically independent (the reference does them independently too), so this is
bit-faithful to ``dual_rmsnorm_ref``. A single fused launch (one CTA covering
both latents, as the triton card does) is a documented perf follow-up, not a
correctness concern.
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .rmsnorm_kernel import rmsnorm_cute

__all__ = ["dual_rmsnorm_cute"]


def dual_rmsnorm_cute(
    x1: torch.Tensor,
    w1: torch.Tensor,
    x2: torch.Tensor,
    w2: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused parallel dual RMSNorm via CUTE DSL (fp32 path) on sm_121.

    Two independent single-launch RMSNorms; numerically identical to
    ``dual_rmsnorm_ref`` (the reference also normalizes the two latents
    independently). See module docstring.
    """
    out1 = rmsnorm_cute(x1, w1, eps)
    out2 = rmsnorm_cute(x2, w2, eps)
    return out1.to(x1.dtype), out2.to(x2.dtype)


# NVIDIA-only registration (the CUTE DSL is NVIDIA-only). On AMD the
# triton/reference card serves the op; this module simply doesn't register.
if detect_vendor() == "nvidia":
    register("dual_rmsnorm", Backend.CUDA)(dual_rmsnorm_cute)
