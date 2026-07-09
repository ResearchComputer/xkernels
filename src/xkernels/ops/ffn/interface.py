"""Public `fused_ffn` op: normalizes leading dims, then dispatches."""
from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers REFERENCE backend)


def fused_ffn(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """SwiGLU FFN: (silu(x @ w_gate) * (x @ w_up)) @ w_down.

    `x` may have any number of leading dims; only the last (feature) dim must
    match `w_gate`/`w_up`. `backend` is "auto" or a `Backend` / its string value.
    """
    *lead, d_model = x.shape
    x2d = x.reshape(-1, d_model)
    out = dispatch("ffn", x2d, w_gate, w_up, w_down, backend=backend)
    return out.reshape(*lead, out.shape[-1])


def fused_xielu_ffn(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    alpha_p: torch.Tensor,
    alpha_n: torch.Tensor,
    *,
    beta: float = 0.5,
    eps: float = -1e-6,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Non-gated xIELU FFN: ``down_proj( xIELU( up_proj(x) ) )``.

    The Apertus FFN (issue #80): a single ``up_proj`` followed by the parametric
    xIELU activation (arXiv:2411.13010) and ``down_proj``. There is no gate, so
    this is NOT SwiGLU (cf. :func:`fused_ffn`). ``alpha_p`` / ``alpha_n`` are the
    raw **log-space** parameters (1-element tensors, exactly as stored in the
    Apertus checkpoint under ``mlp.act_fn.*``); the effective scales are
    ``softplus(alpha_p)`` and ``beta + softplus(alpha_n)``.

    Args:
        x: ``[..., K]`` input (fp32 / bf16 / fp16); leading dims are flattened.
        w_up: ``[K, N]`` up-projection weight.
        w_down: ``[N, K]`` down-projection weight.
        alpha_p: 1-element tensor, the positive-branch log-space scale.
        alpha_n: 1-element tensor, the negative-branch log-space scale offset.
        beta: fixed buffer (Apertus default ``0.5``).
        eps: fixed buffer (Apertus default ``-1e-6``); caps the negative-branch
            exponent (``min(x, eps)``) for numerical stability.
        backend: ``"auto"`` (triton when available, else reference) or a
            ``Backend`` / its string value.

    Returns:
        ``[..., K]`` output in the input dtype.
    """
    *lead, d_model = x.shape
    x2d = x.reshape(-1, d_model)
    out = dispatch(
        "xielu_ffn",
        x2d,
        w_up,
        w_down,
        alpha_p,
        alpha_n,
        beta=beta,
        eps=eps,
        backend=backend,
    )
    return out.reshape(*lead, out.shape[-1])
