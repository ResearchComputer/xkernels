"""Pure-torch SwiGLU FFN — the correctness oracle and default backend."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register


def ffn_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Compute (silu(x @ w_gate) * (x @ w_up)) @ w_down.

    The three projections are ``torch.matmul`` in the input dtype; the fused
    SwiGLU activation ``silu(g) * u`` is accumulated in fp32 per the Op Spec's
    ``numerics.reduce_dtype`` ("Accumulate the activation products in fp32").
    Computing the activation in input dtype (bf16/fp16) left the reference on a
    *different precision path* from the Triton kernel (which upcasts sigmoid to
    fp32), and the down-projection's catastrophic cancellation (K=N, mixed-sign
    ``h``) amplified the last-ULP difference to rel≈2.5 at near-zero outputs
    (issue #82). fp32 activation makes reference and every backend bit-identical
    for the activation -> identical projection outputs.
    """
    g = x @ w_gate
    u = x @ w_up
    # SwiGLU activation silu(g)*u = g*sigmoid(g)*u, accumulated in fp32 per the
    # Op Spec's numerics.reduce_dtype ("Accumulate the activation products in
    # fp32"). Written as the manual g*sigmoid(g) product (NOT F.silu, whose
    # fused impl differs at fp32-ULP) so the reference shares ONE precision path
    # with the Triton kernel -> bit-identical activation -> identical projections.
    # Divergent paths left rel~2.5 at near-zero outputs (the down-projection's
    # K=N cancellation amplifies a last-ULP activation diff, issue #82).
    gf = g.float()
    uf = u.float()
    h = (gf * torch.sigmoid(gf) * uf).to(g.dtype)
    return h @ w_down


register("ffn", Backend.REFERENCE)(ffn_reference)
