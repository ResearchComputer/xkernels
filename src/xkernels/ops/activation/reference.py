# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Hand-written reference for the parametric ``xielu`` activation (issue #80).

Apertus (``swiss-ai/Apertus-8B-Instruct-2509``) uses a non-gated FFN
``up_proj -> xIELU -> down_proj`` with the parametric xIELU activation
(arXiv:2411.13010). xIELU carries learned parameters ``alpha_p`` / ``alpha_n``
(stored log-space in the checkpoint under ``mlp.act_fn.*``) plus fixed buffers
``beta`` / ``eps`` and is piecewise on the sign of ``x``.

This is the backend-neutral oracle every card is checked against (the
``numerics.reference`` for ``xielu@1.0.0``). Pure torch, nonlinearity evaluated in
fp32 (the flashinfer/vLLM convention), written for clarity not speed. It is
bit-identical to ``transformers.activations.XIELUActivation`` (verified to 0.0
max abs err on the mini-sglang CPU probe), so a card matching this reference
matches the HF/vLLM math exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["xielu"]


def xielu(
    x: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    beta: float = 0.5,
    eps: float = -1e-6,
) -> torch.Tensor:
    """Parametric xIELU activation — the numerical oracle.

    ``alpha_p`` / ``alpha_n`` are the raw log-space parameters (1-element tensors,
    exactly as stored in the Apertus checkpoint); the effective scales are
    ``softplus(alpha_p)`` and ``beta + softplus(alpha_n)``. The nonlinearity is
    evaluated in fp32 then cast to the input dtype — the only divergence vs an
    fp32 oracle is that final cast (same as the ``silu_and_mul`` family).

    Args:
        x: ``[..., K]`` activations (fp32 / bf16 / fp16).
        alpha_p: 1-element tensor, the positive-branch log-space scale.
        alpha_n: 1-element tensor, the negative-branch log-space scale offset.
        beta: fixed buffer (Apertus default ``0.5``).
        eps: fixed buffer (Apertus default ``-1e-6``); caps the exponent in the
            negative branch (``min(x, eps)``) for numerical stability.

    Returns:
        ``[..., K]`` output in the input dtype.
    """
    xf = x.float()
    ap = F.softplus(alpha_p.float())               # = log(1 + exp(alpha_p))
    an = beta + F.softplus(alpha_n.float())
    pos = ap * xf * xf + beta * xf
    # torch.clamp(xf, max=eps) == torch.min(xf, eps) (elementwise); the vLLM/HF spelling.
    neg = (torch.expm1(torch.clamp(xf, max=eps)) - xf) * an + beta * xf
    return torch.where(xf > 0, pos, neg).to(x.dtype)


register("xielu", Backend.REFERENCE)(xielu)
