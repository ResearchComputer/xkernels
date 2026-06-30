# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``Backend.CUDA`` registration for ``mha_merge_state`` via the CUTE DSL.

Signature matches the triton/reference entry:
``(out_a, lse_a, out_b, lse_b) -> (out, lse)``.
"""
from __future__ import annotations

import torch

from ...._backends import Backend, detect_vendor
from ...._dispatch import register
from .merge_state_kernel import merge_state_cute

__all__ = ["mha_merge_state_cute"]


def mha_merge_state_cute(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax merge of two attention partials via CUTE DSL (fp32 path).

    ``out_a``/``out_b`` are upcast to fp32 on the host (bit-identical to the
    reference); the kernel runs pure fp32; ``out`` is cast back to ``out_a.dtype``
    (``lse`` stays fp32). See ``merge_state_kernel.py`` for the design.
    """
    out, lse = merge_state_cute(out_a, lse_a, out_b, lse_b)
    return out.to(out_a.dtype), lse


# NVIDIA-only registration (the CUTE DSL is NVIDIA-only). On AMD the
# triton/reference card serves the op; this module simply doesn't register.
if detect_vendor() == "nvidia":
    register("mha_merge_state", Backend.CUDA)(mha_merge_state_cute)
