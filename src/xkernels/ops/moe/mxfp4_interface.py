# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public ``fused_moe_mxfp4`` op: MXFP4 W4A16 grouped fused-MoE GEMM for
DeepSeek-V4 routed experts. Dispatches to a registered backend.

The op fuses the whole routed-expert FFN: topk gather -> gate_up GEMM ->
clamped SwiGLU (optional per-expert bias) -> down GEMM -> routed-weighted
combine, returning a ``[M, hidden]`` result. Experts stay **packed** in MXFP4;
only active experts are touched (a full bf16 dequant of all 256 V4 experts is
~138 GB/rank and OOMs the APU).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import mxfp4_reference  # noqa: F401  (registers REFERENCE backend)


def fused_moe_mxfp4(
    A: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    *,
    b13: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    swiglu_limit: float | None = 10.0,
    group_size: int = 32,
    mul_routed_weight: bool = True,
    expert_map: torch.Tensor | None = None,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """MXFP4 grouped fused-MoE GEMM for DeepSeek-V4 routed experts.

    Computes, for each token ``m`` routed to top-k experts ``e``::

        gate_up = A[m] @ w13[e]^T + b13[e]           # [2*ispp]
        gate, up = chunk(gate_up, 2)
        act = silu(clamp(gate, max=L)) * clamp(up, -L, L)   # [ispp]
        down = act @ w2[e]^T + b2[e]                  # [hidden]
        out[m] = sum_e routed_w[m, e] * down

    Args:
        A: ``[M, hidden]`` activations (bf16, or fp32 under the Triton interpreter).
        w13: ``[E, 2*ispp, hidden // 2]`` uint8 packed MXFP4 gate_up weights.
            Rows are concatenated ``[gate(ispp); up(ispp)]``. Under expert
            parallelism (``expert_map`` given) this is the **rank-local** slice.
        w13_scale: ``[E, 2*ispp, hidden // group_size]`` uint8 ue8m0 block scales.
        w2: ``[E, hidden, ispp // 2]`` uint8 packed MXFP4 down weights.
        w2_scale: ``[E, hidden, ispp // group_size]`` uint8 ue8m0 block scales.
        topk_ids: ``[M, top_k]`` int32 **global** expert ids.
        topk_w: ``[M, top_k]`` fp32 routing weights (routed scaling pre-baked).
        b13: optional ``[E, 2*ispp]`` gate_up bias (column-parallel, added locally).
        b2: optional ``[E, hidden]`` down bias (row-parallel; add on one rank only
            under TP — see the tokenspeed convention).
        swiglu_limit: SwiGLU clamp limit ``L`` (V4 default 10.0); ``None``/<=0
            disables the clamp.
        group_size: MXFP4 block size along the contracted dim (default 32).
        mul_routed_weight: fold the routing weight into the combine.
        expert_map: optional ``[num_global_experts]`` int tensor for expert
            parallelism. Entry ``g`` is the local weight-row of global expert ``g``
            (in ``[0, E_local)``), or ``-1`` if not on this rank. When given, the
            op returns this rank's **partial** output; the caller all-reduces.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``[M, hidden]`` output in ``A.dtype`` (a per-rank partial when
        ``expert_map`` is given).
    """
    return dispatch(
        "moe_mxfp4",
        A,
        w13,
        w13_scale,
        w2,
        w2_scale,
        topk_ids,
        topk_w,
        b13=b13,
        b2=b2,
        swiglu_limit=swiglu_limit,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        expert_map=expert_map,
        backend=backend,
    )
