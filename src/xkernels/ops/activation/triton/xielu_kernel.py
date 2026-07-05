# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Hand-written Triton backend for the parametric ``xielu`` activation (issue #80).

A flat-1D elementwise kernel: one program per ``BLOCK``-sized tile of the flattened
output. Each lane loads one ``x`` element (upcast to fp32), loads the two scalar
log-space params, applies the xIELU math in fp32, and stores the result in the
input dtype. The learned ``alpha_p`` / ``alpha_n`` arrive as 1-element device
tensors (the checkpoint form); ``softplus`` is computed in-kernel with the
**numerically stable** formulation ``where(z>0, z+log1p(e^-z), log1p(e^z))`` —
the naive ``log(1+exp(z))`` overflows fp32 for z > ~88, and the Apertus-8B
checkpoint stores ``alpha_p = 166.0``, so the naive form silently produces inf.
Matching the vLLM experimental CUDA call-out's signature so this is a drop-in.

Why hand-written (not DSL): the vkl math IR has no value-comparison primitive
(``> / <``) and no ``log`` (needed for ``softplus``), so xIELU's
``where(x > 0, ...)`` branch-on-sign and the param ``softplus`` are not
expressible in the current DSL. Per ``author-an-op-spec`` SKILL.md this routes to
the hand-written fallback (this file), mirroring the existing hand-written
``dual_rmsnorm_kernel``. Portable Triton runs on ``amd_cdna3`` (gfx942) and NVIDIA.

The nonlinearity is evaluated in fp32 (the only divergence vs an fp32 oracle is
the final fp32 -> bf16/fp16 cast), matching the reference and the
``silu_and_mul`` / ``gelu_and_mul`` family (issue #67).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["xielu_triton", "xielu_kernel"]


@triton.jit
def xielu_kernel(
    x_ptr,
    alpha_p_ptr,
    alpha_n_ptr,
    out_ptr,
    n,
    beta: tl.constexpr,       # python float -> constexpr scalar
    eps: tl.constexpr,        # python float -> constexpr scalar
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # scalar params: one element each. Compute softplus in fp32.
    ap_log = tl.load(alpha_p_ptr).to(tl.float32)
    an_log = tl.load(alpha_n_ptr).to(tl.float32)
    # Numerically stable softplus(z) = max(0,z) + log(1 + exp(-|z|)).
    # exp(-|z|) is always in (0, 1] so this NEVER exponentiates a large positive
    # number — critical because the Apertus-8B checkpoint stores alpha_p = 166.0
    # and the naive log(1+exp(z)) overflows fp32 (exp(166) = inf), silently
    # poisoning every positive activation. (tl.where(z>0, z+log1p(e^-z), ...) would
    # also work but evaluates both SIMD branches, still computing exp(166).)
    # tl has no abs, so |z| = max(z, -z).
    ap = tl.maximum(0.0, ap_log) + tl.log(1.0 + tl.exp(-tl.maximum(ap_log, -ap_log)))
    an = beta + tl.maximum(0.0, an_log) + tl.log(1.0 + tl.exp(-tl.maximum(an_log, -an_log)))

    pos = ap * x * x + beta * x
    neg = (tl.exp(tl.minimum(x, eps)) - 1.0 - x) * an + beta * x   # expm1(min(x,eps))
    y = tl.where(x > 0.0, pos, neg)

    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def xielu_triton(
    x: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    beta: float = 0.5,
    eps: float = -1e-6,
) -> torch.Tensor:
    """Launch the flat-1D xIELU kernel over a contiguous ``x`` of any rank."""
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    # alpha_p / alpha_n are 1-element tensors on the same device as x.
    assert alpha_p.numel() == 1 and alpha_n.numel() == 1, "alpha_p/alpha_n must be scalar tensors"
    block = 2048
    grid = (triton.cdiv(n, block),)
    # BLOCK tuned for a bandwidth-bound elementwise op (matches the gated
    # activations' Launch.elementwise knob range); a wider block is the
    # GPU-gated tune-for-cdna lever.
    xielu_kernel[grid](
        x,
        alpha_p,
        alpha_n,
        out,
        n,
        beta,
        eps,
        BLOCK=block,
        num_warps=8,
    )
    return out


register("xielu", Backend.TRITON)(xielu_triton)
