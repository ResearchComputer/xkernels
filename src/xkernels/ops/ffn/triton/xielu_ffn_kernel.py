# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton backend for the Apertus non-gated xIELU FFN (issue #80).

``out = down_proj( xIELU( up_proj(x) ) )`` — the FFN of
``swiss-ai/Apertus-8B-Instruct-2509``. There is no gate (NOT SwiGLU); the
parametric xIELU activation (arXiv:2411.13010) sits between the two
projections.

The two projections are ``torch.matmul`` in both the reference and this backend
(identical -> they contribute no cross-backend divergence). The xIELU
activation reuses the existing hand-written Triton kernel
(:func:`xkernels.ops.activation.triton.xielu_kernel.xielu_triton`) so this card
shares ONE activation precision path with the standalone ``xielu@1.0.0`` op and
the reference — exactly the discipline ``fused_ffn`` uses for SwiGLU (issue
#82): bit-identical activation, only the triton-vs-torch xIELU fp32-ULP differs,
amplified by the down-projection's cancellation over the contracted dim ``N``.

Why hand-written (not DSL): xIELU's ``where(x > 0, ...)`` branch-on-sign and the
param ``softplus`` need a value-comparison primitive and ``log`` — neither is in
the vkl math IR (see the standalone ``xielu_kernel.py`` for the full rationale).
This file mirrors ``ffn_kernel.py`` (projections = torch.matmul, only the
activation is a Triton kernel) and is portable across NVIDIA and AMD (gfx942).
"""
from __future__ import annotations

import torch

from ...._backends import Backend
from ...._dispatch import register
from ...activation.triton.xielu_kernel import xielu_triton


def xielu_ffn_triton(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    beta: float = 0.5,
    eps: float = -1e-6,
) -> torch.Tensor:
    """Non-gated xIELU FFN on the triton backend.

    The projections are ``torch.matmul`` (unchanged from the reference); only the
    xIELU activation is the hand-written Triton kernel, reused from
    ``xielu@1.0.0`` so the activation precision path is shared exactly.
    """
    h = x @ w_up
    a = xielu_triton(h, alpha_p, alpha_n, beta=beta, eps=eps)
    return a @ w_down


register("xielu_ffn", Backend.TRITON)(xielu_ffn_triton)
