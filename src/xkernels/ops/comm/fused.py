# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fused residual-add + RMSNorm epilogue + its composition with the all-reduce
(issue #12). Pure-torch reference oracle, a backend-selecting public op, and the
``hierarchical_all_reduce`` + fused-epilogue wrapper.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import backend_registration_guard, dispatch, register
from .hierarchical import hierarchical_all_reduce

__all__ = [
    "add_rmsnorm_ref",
    "residual_add",
    "residual_rmsnorm",
    "hierarchical_all_reduce_residual_rmsnorm",
]


def add_rmsnorm_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: ``new_residual = x + residual; out = rmsnorm(new_residual, weight)``.

    Returns ``(out, new_residual)`` both in ``x.dtype`` (fp32 reduction).
    """
    hf = x.float() + residual.float()
    new_residual = hf.to(x.dtype)
    inv = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    out = ((hf * inv).to(weight.dtype) * weight).to(x.dtype)
    return out, new_residual


def residual_add_ref(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Reference for the bare residual add (``residual_add@1.0.0``).

    ``out = (x.float() + residual.float()).to(x.dtype)`` — the add is promoted
    to fp32 then cast to the input dtype, exactly the residual convention
    :func:`add_rmsnorm_ref` uses (``hf = x.float() + residual.float()``). There is
    no reduction; the promotion is a precision choice, not an accumulation.

    This is both the correctness oracle AND the standalone implementation: a
    separate Triton kernel for ``x + residual`` would lose to ``torch.add``
    (launch overhead > the add), so the reference is what a standalone call runs
    and the *contract* is what a fused/persistent card references as an on-chip
    stage (megakernel-blockers.md (b) point 4). 'Compose over generate'
    (library.md §1.2): don't emit a kernel for what ``torch.add`` already does.
    """
    return (x.float() + residual.float()).to(x.dtype)


register("residual_add", Backend.REFERENCE)(residual_add_ref)


def residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Bare residual add: ``out = (x.float() + residual.float()).to(dtype)``.

    The standalone op for ``residual_add@1.0.0``. Dispatches through the
    registry so a fused/persistent card can replace the reference later; today
    only the ``REFERENCE`` backend is registered (a separate Triton kernel for
    ``x + residual`` would lose to ``torch.add`` — see :func:`residual_add_ref`).
    The op's value is the *contract*: a persistent-grid megakernel names this as
    an on-chip stage with a defined precision path (megakernel-blockers.md (b)).
    """
    return dispatch("residual_add", x, residual, backend=backend)


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    use_triton: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm. Uses the Triton kernel when available on GPU,
    else the torch reference. ``use_triton`` forces the choice."""
    if use_triton is None:
        use_triton = x.is_cuda
    if use_triton:
        with backend_registration_guard(
            "residual_rmsnorm",
            Backend.TRITON,
            source="xkernels.ops.comm.triton.add_rmsnorm_kernel",
        ):
            from .triton.add_rmsnorm_kernel import add_rmsnorm_triton

            return add_rmsnorm_triton(x, residual, weight, eps)
    return add_rmsnorm_ref(x, residual, weight, eps)


def hierarchical_all_reduce_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    intra_group,
    cross_group,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical all-reduce of ``x`` followed by the fused residual+RMSNorm epilogue.

    Returns ``(normed_output, new_residual)``.
    """
    reduced = hierarchical_all_reduce(x, intra_group, cross_group)
    return residual_rmsnorm(reduced, residual, weight, eps)
