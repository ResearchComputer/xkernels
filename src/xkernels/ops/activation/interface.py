# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public gated-activation ops (issue #67): standalone SwiGLU / GELU-gated multiply.

These are the bare ``act(gate) * up`` kernels factored out of ``fused_ffn`` — the
ops mini-sglang / vLLM / flashinfer call as ``silu_and_mul`` / ``gelu_and_mul``
when the FFN is split across a separately-fused GEMM (no flashinfer ROCm wheel).
Two input conventions, matching the two contract families in the vkl DSL
(``xkernels.vkl.examples.activation``):

* :func:`silu_and_mul` / :func:`gelu_and_mul` — the mathematically-honest
  two-tensor form ``act(gate[M,K]) * up[M,K]``.
* :func:`packed_silu_and_mul` / :func:`packed_gelu_and_mul` — the flashinfer/vLLM
  single-buffer convention: one ``x[M, 2K]`` tensor with the gate in the first
  ``K`` columns and ``up`` in the remaining ``K``.

Each function dispatches to a registered backend:

* ``REFERENCE`` — the DSL auto-reference (the ``@kernel`` body run on torch; the
  numerical oracle, also used by ``verify``).
* ``TRITON`` — the DSL-generated flat-1D elementwise kernel (registered in
  ``xkernels.ops.activation.triton.activation_kernel``).

The contract — Op Spec, reference, tolerances, shape sweep, Impl Cards — is
authored once in the DSL; this module is the thin dispatch surface that makes
the ops callable as ``xkernels.silu_and_mul(gate, up)``. Keyword arguments are
used throughout because the DSL auto-reference is a ``(**inputs)`` callable.
"""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch

__all__ = ["silu_and_mul", "gelu_and_mul", "packed_silu_and_mul", "packed_gelu_and_mul"]


def _single(result: object) -> torch.Tensor:
    """Collapse a single-output backend result to the bare tensor.

    Both DSL backends return the outputs as a tuple in spec order; these ops have
    exactly one output, so a 1-tuple comes back. A defensively-bare tensor (any
    future hand backend) is passed through unchanged.
    """
    if isinstance(result, (tuple, list)):
        tensors = [t for t in result if isinstance(t, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0]
    return result  # type: ignore[return-value]


def silu_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """SwiGLU activation: ``out = silu(gate) * up``  where  ``silu(x) = x * sigmoid(x)``.

    The nonlinearity is evaluated in fp32 (gate upcast) then cast to the output
    dtype — the flashinfer/vLLM convention, and more accurate than a pure-bf16
    silu.

    Args:
        gate: ``[M, K]`` gate activations (fp32 / bf16 / fp16).
        up: ``[M, K]`` up activations (same dtype and shape as ``gate``).
        backend: ``"auto"`` (triton when available, else reference) or a
            ``Backend`` / its string value.

    Returns:
        ``[M, K]`` output in the input dtype.
    """
    return _single(dispatch("silu_and_mul", gate=gate, up=up, backend=backend))


def gelu_and_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """GELU(tanh)-gated multiply: ``out = gelu_tanh(gate) * up``.

    GELU uses the tanh approximation (``0.5 * x * (1 + tanh(sqrt(2/pi) * (x +
    0.044715 * x**3)))``) — the form flashinfer / vLLM's ``gelu_and_mul`` use —
    evaluated in fp32 then cast to the output dtype.

    Args:
        gate: ``[M, K]`` gate activations (fp32 / bf16 / fp16).
        up: ``[M, K]`` up activations (same dtype and shape as ``gate``).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[M, K]`` output in the input dtype.
    """
    return _single(dispatch("gelu_and_mul", gate=gate, up=up, backend=backend))


def packed_silu_and_mul(
    x: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Packed SwiGLU: ``out = silu(x[:, :K]) * x[:, K:]`` for ``x`` of shape ``[M, 2K]``.

    The flashinfer/vLLM single-buffer convention: the gate half is the first
    ``K`` columns and the up half is the remaining ``K`` columns of one packed
    ``[M, 2K]`` tensor (``2K`` must be even).

    Args:
        x: ``[M, 2K]`` packed gate/up tensor, contiguous row-major.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[M, K]`` output in the input dtype.
    """
    return _single(dispatch("packed_silu_and_mul", x=x, backend=backend))


def packed_gelu_and_mul(
    x: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Packed GELU(tanh)-gated multiply: ``out = gelu_tanh(x[:, :K]) * x[:, K:]``.

    Same packed single-buffer convention as :func:`packed_silu_and_mul`, with the
    tanh-approximation GELU as the gate nonlinearity (evaluated in fp32).

    Args:
        x: ``[M, 2K]`` packed gate/up tensor, contiguous row-major.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[M, K]`` output in the input dtype.
    """
    return _single(dispatch("packed_gelu_and_mul", x=x, backend=backend))
