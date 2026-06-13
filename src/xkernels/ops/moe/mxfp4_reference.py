# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the MXFP4 fused-MoE GEMM (DeepSeek-V4 routed experts)
— the numerical oracle and the default (CPU / no-Triton) backend for
``moe_mxfp4``.

Mirrors tokenspeed's ``Mxfp4DequantBackend._mxfp4_dequant_moe`` exactly: for each
active expert, dequant its packed MXFP4 ``w13``/``w2`` to bf16, run
``gate_up -> clamped SwiGLU -> down`` over the assigned tokens, fold the optional
per-expert biases and the routing weight, and scatter-add into the output.

Acceptance (issue #43): match the dequant-then-matmul stack within
``atol/rtol ~ 2e-2`` bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register
from .mxfp4 import MXFP4_GROUP_SIZE, dequant_mxfp4_weight

__all__ = ["moe_mxfp4_ref"]


def moe_mxfp4_ref(
    A: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    b13: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    swiglu_limit: float | None = 10.0,
    group_size: int = MXFP4_GROUP_SIZE,
    mul_routed_weight: bool = True,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference MXFP4 fused-MoE: per active expert, gate_up -> SwiGLU -> down.

    Args:
        A: ``[M, hidden]`` activations (bf16, or fp32 under the interpreter).
        w13: ``[E, 2*ispp, hidden // 2]`` uint8 packed gate_up weights
            (rank-local under EP). Rows are concatenated ``[gate(ispp); up(ispp)]``.
        w13_scale: ``[E, 2*ispp, hidden // group_size]`` uint8 ue8m0 scales.
        w2: ``[E, hidden, ispp // 2]`` uint8 packed down weights.
        w2_scale: ``[E, hidden, ispp // group_size]`` uint8 ue8m0 scales.
        topk_ids: ``[M, top_k]`` int32 **global** expert ids.
        topk_w: ``[M, top_k]`` fp32 routing weights.
        b13: optional ``[E, 2*ispp]`` gate_up bias (added before SwiGLU).
        b2: optional ``[E, hidden]`` down bias (added once per token).
        swiglu_limit: SwiGLU clamp limit ``L`` (V4 default 10.0); ``None``/<=0
            disables clamping.
        group_size: MXFP4 block size along the contracted dim (32).
        mul_routed_weight: fold the routing weight into the combine.
        expert_map: optional ``[num_global_experts]`` global->local row map for
            expert parallelism; ``-1`` = expert not on this rank. When given, only
            locally-held experts contribute (this rank's partial output).

    Returns:
        ``[M, hidden]`` output in ``A.dtype`` (fp32 accumulation).
    """
    M, top_k = topk_ids.shape
    hidden = A.shape[1]
    out = torch.zeros(M, hidden, dtype=torch.float32, device=A.device)
    emap = None if expert_map is None else expert_map.to(A.device)
    E_local = w13.shape[0]

    flat_ids = topk_ids.reshape(-1)
    flat_w = topk_w.reshape(-1).to(torch.float32)
    flat_tok = torch.arange(M, device=A.device).repeat_interleave(top_k)

    for e_global in torch.unique(flat_ids).tolist():
        e = int(e_global)
        if emap is not None:
            e = int(emap[e])
        if e < 0 or e >= E_local:
            continue
        sel = flat_ids == e_global
        tok = flat_tok[sel]
        weight = flat_w[sel]
        if tok.numel() == 0:
            continue

        xe = A[tok].to(torch.bfloat16)
        w13_e = dequant_mxfp4_weight(w13[e], w13_scale[e], group_size)  # [2*ispp, hidden]
        gate_up = torch.matmul(xe, w13_e.transpose(0, 1))  # [n, 2*ispp]
        if b13 is not None:
            gate_up = gate_up + b13[e].to(gate_up.dtype)

        gate, up = gate_up.float().chunk(2, dim=-1)
        if swiglu_limit is not None and swiglu_limit > 0:
            gate = torch.clamp(gate, max=swiglu_limit)
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        act = (F.silu(gate) * up).to(torch.bfloat16)  # [n, ispp]

        w2_e = dequant_mxfp4_weight(w2[e], w2_scale[e], group_size)  # [hidden, ispp]
        down = torch.matmul(act, w2_e.transpose(0, 1))  # [n, hidden]
        if b2 is not None:
            down = down + b2[e].to(down.dtype)

        contrib = down.float()
        if mul_routed_weight:
            contrib = contrib * weight.unsqueeze(-1)
        out.index_add_(0, tok, contrib)

    return out.to(A.dtype)


register("moe_mxfp4", Backend.REFERENCE)(moe_mxfp4_ref)
