"""Triton FFN backend: torch matmuls for the projections, a fused Triton
kernel for the elementwise SwiGLU activation `silu(g) * u`."""
from __future__ import annotations

from functools import partial

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from .._activation import SwigluAct

# Default tile size for the fused SwiGLU kernel; overridable as a specialization
# knob (see registry/impls/fused_ffn.triton.card.json -> specialization_knobs.BLOCK).
DEFAULT_BLOCK = 1024


@triton.jit
def _swiglu_kernel(g_ptr, u_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # Accumulate the SwiGLU activation silu(g)*u = g*sigmoid(g)*u in fp32 per the
    # Op Spec's numerics.reduce_dtype, matching the reference bit-for-bit (the
    # reference upcasts to fp32 too). Storing auto-converts to the output dtype.
    # Computing in input dtype diverged from the reference by a last-ULP that the
    # down-projection's cancellation amplified to rel~2.5 (issue #82).
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    out = (g * tl.sigmoid(g)) * u
    tl.store(out_ptr + offs, out, mask=mask)


def _swiglu_triton(g: torch.Tensor, u: torch.Tensor, *, BLOCK: int = DEFAULT_BLOCK) -> torch.Tensor:
    g = g.contiguous()
    u = u.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
    _swiglu_kernel[grid](g, u, out, n, BLOCK=BLOCK)
    return out


def ffn_triton(x, w_gate, w_up, w_down, *, BLOCK: int = DEFAULT_BLOCK):
    """SwiGLU FFN on the triton backend.

    ``BLOCK`` is the fused-activation tile size (a declared specialization knob);
    the projections are torch matmuls and are unaffected by it.
    """
    g = x @ w_gate
    u = x @ w_up
    h = SwigluAct.apply(g, u, partial(_swiglu_triton, BLOCK=BLOCK))
    return h @ w_down


register("ffn", Backend.TRITON)(ffn_triton)
