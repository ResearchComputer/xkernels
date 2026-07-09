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


def xielu_ffn_reference(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    beta: float = 0.5,
    eps: float = -1e-6,
) -> torch.Tensor:
    """Non-gated xIELU FFN: ``down_proj( xIELU( up_proj(x) ) )``.

    Apertus (``swiss-ai/Apertus-8B-Instruct-2509``) uses a non-gated FFN
    ``up_proj -> xIELU -> down_proj`` (issue #80) — there is no ``gate_proj``, so
    this is NOT SwiGLU (cf. :func:`ffn_reference`). The parametric xIELU
    nonlinearity (arXiv:2411.13010) is evaluated in fp32 then cast to the input
    dtype, exactly as the standalone ``xielu@1.0.0`` reference does.

    The activation is delegated to the standalone reference
    (:func:`xkernels.ops.activation.reference.xielu`) so this oracle and every
    backend card share ONE activation precision path -> bit-identical activation,
    and only the projections (identical ``torch.matmul``) plus the
    triton-vs-torch xIELU fp32-ULP differ. The down-projection sums a mixed-sign
    intermediate over the contracted dim ``N`` (Apertus ``N=21504``), so the same
    catastrophic-cancellation analysis as issue #82 applies (fp32 rtol=1e-4, the
    cancellation floor — not a looseness).
    """
    # Reuse the standalone xielu reference so the activation math is defined
    # exactly once (xkernels.ops.activation.reference:xielu) and cannot drift
    # from the xielu@1.0.0 op this FFN is built from.
    from ..activation.reference import xielu as _xielu

    h = x @ w_up
    a = _xielu(h, alpha_p, alpha_n, beta=beta, eps=eps)
    return a @ w_down


register("ffn", Backend.REFERENCE)(ffn_reference)
register("xielu_ffn", Backend.REFERENCE)(xielu_ffn_reference)
