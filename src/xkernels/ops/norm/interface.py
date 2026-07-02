# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public RMSNorm ops: ``dual_rmsnorm`` (MLA paired latents) and plain
``rmsnorm`` (single tensor, issue #66). Each dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)

__all__ = ["dual_rmsnorm", "rmsnorm"]


def _single(result: object) -> torch.Tensor:
    """Collapse a single-output backend result to the bare tensor.

    The DSL backends return the outputs as a tuple in spec order; ``rmsnorm``
    has exactly one output, so a 1-tuple comes back. A defensively-bare tensor
    (the hand reference) is passed through unchanged.
    """
    if isinstance(result, (tuple, list)):
        tensors = [t for t in result if isinstance(t, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    return result  # type: ignore[return-value]


def dual_rmsnorm(
    x1: torch.Tensor,
    w1: torch.Tensor,
    x2: torch.Tensor,
    w2: torch.Tensor,
    *,
    eps: float = 1e-6,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused parallel dual RMSNorm of two MLA latents (``q_a`` / ``kv_a``).

    Computes ``(rmsnorm(x1, w1), rmsnorm(x2, w2))`` in a single launch on the
    Triton backend (rows are independent; the win is one kernel/one pass vs two
    sequential RMSNorm launches).

    Args:
        x1: ``[T, d1]`` activations, x2: ``[T, d2]`` activations (must share T).
        w1: ``[d1]`` per-feature weight, w2: ``[d2]`` per-feature weight.
        eps: variance epsilon.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out1 [T, d1], out2 [T, d2])`` in the input dtypes.
    """
    return dispatch("dual_rmsnorm", x1, w1, x2, w2, eps=eps, backend=backend)


def rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Plain single-tensor RMSNorm (flashinfer-compatible), issue #66.

    Computes ``out = (x * rsqrt(mean(x.float()**2, -1) + eps)).to(x.dtype) * w``
    with the variance reduced in fp32 (the load-bearing numerical invariant;
    identical math to one branch of :func:`dual_rmsnorm`). mini-sglang's
    ``RMSNorm`` (ported to AMD ROCm) calls this where it used to call
    ``flashinfer.rmsnorm`` (no ROCm wheel).

    ``eps`` is baked at ``1e-6`` in the DSL body (the Llama / DeepSeek default;
    the math IR carries it as a literal, not a runtime scalar), so it is not a
    parameter here -- a non-default eps needs a DSL body change, not a call-site
    override.

    Args:
        x: ``[..., d]`` activations (fp32 / bf16 / fp16); reduction over the last
            axis in fp32.
        w: ``[d]`` per-feature weight (same dtype as ``x``).
        backend: ``"auto"`` (triton when available, else reference) or a
            ``Backend`` / its string value.

    Returns:
        ``[..., d]`` output in the input dtype.
    """
    return _single(dispatch("rmsnorm", x=x, w=w, backend=backend))
