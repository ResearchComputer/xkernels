# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Shared helpers for the MXFP4 fused-MoE GEMM (DeepSeek-V4 routed experts):
weight unpack/dequant and a random-weight generator for tests.

DeepSeek-V4-Flash routed experts are **MXFP4** (OCP MX): packed E2M1 4-bit values
(two nibbles per ``uint8``, low nibble = even element) plus one ``ue8m0`` block-32
scale per group. This is the same convention the DSA-indexer paged KV gather uses
(see :mod:`xkernels.ops.gather.mxfp4`); we re-export its dequant so the two paths
stay numerically identical.

The two MoE projections are stored **per expert** as:

* ``w13 = [E, 2*ispp, hidden // 2] uint8`` — fused, column-parallel gate_up proj,
  rows concatenated ``[gate(ispp); up(ispp)]``; scale ``[E, 2*ispp, hidden // 32]``.
* ``w2  = [E, hidden, ispp // 2]   uint8`` — row-parallel down proj; scale
  ``[E, hidden, ispp // 32]``.

The compute is the V4 clamped SwiGLU ``F.silu(clamp(g, max=L)) * clamp(u, -L, L)``
(``swiglu_limit=10.0``, no gpt-oss ``+1``) with optional per-expert biases
``b13 = [E, 2*ispp]`` and ``b2 = [E, hidden]``.
"""

from __future__ import annotations

import torch

from ..gather.mxfp4 import MXFP4_GROUP_SIZE, dequant_mxfp4

__all__ = [
    "MXFP4_GROUP_SIZE",
    "dequant_mxfp4",
    "dequant_mxfp4_weight",
    "make_mxfp4_moe_weights",
]


def dequant_mxfp4_weight(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int = MXFP4_GROUP_SIZE
) -> torch.Tensor:
    """Unpack + dequantize a per-expert MXFP4 weight tensor to bf16.

    Thin wrapper over :func:`xkernels.ops.gather.mxfp4.dequant_mxfp4` that returns
    bf16 (the GEMM operand dtype) instead of fp32, matching the tokenspeed
    ``dequant_mxfp4_to_bf16`` oracle.

    Args:
        packed: ``[..., N, K // 2]`` uint8, two E2M1 nibbles per byte (low nibble =
            even / lower K index).
        scale: ``[..., N, K // group_size]`` uint8 ue8m0 block scales.
        group_size: shared-scale group length along K (32).

    Returns:
        ``[..., N, K]`` bf16 dequantized weights.
    """
    return dequant_mxfp4(packed, scale, group_size).to(torch.bfloat16)


def make_mxfp4_moe_weights(
    E: int,
    hidden: int,
    ispp: int,
    *,
    group_size: int = MXFP4_GROUP_SIZE,
    with_bias: bool = False,
    device="cuda",
    seed: int = 0,
):
    """Generate random valid MXFP4 MoE expert weights + their exact dequant.

    Returns a dict with the packed ``w13``/``w2`` (uint8), their ``ue8m0`` scales
    (uint8), optional bf16 biases ``b13``/``b2``, and the reference bf16 dequants
    ``w13_ref``/``w2_ref`` (``dequant_mxfp4_weight`` of the packed tensors).

    Shapes (TP-sharded, ``ispp = moe_intermediate_size // tp_size``):

    * ``w13``       ``[E, 2*ispp, hidden // 2]`` uint8
    * ``w13_scale`` ``[E, 2*ispp, hidden // group_size]`` uint8
    * ``w2``        ``[E, hidden, ispp // 2]`` uint8
    * ``w2_scale``  ``[E, hidden, ispp // group_size]`` uint8
    * ``b13``       ``[E, 2*ispp]`` bf16 (if ``with_bias``)
    * ``b2``        ``[E, hidden]`` bf16 (if ``with_bias``)
    """
    assert hidden % 2 == 0 and hidden % group_size == 0
    assert ispp % 2 == 0 and ispp % group_size == 0
    g = torch.Generator(device=device).manual_seed(seed)

    def _packed(rows: int, cols: int):
        # Packed nibbles: any byte is valid E2M1x2.
        packed = torch.randint(
            0, 256, (E, rows, cols // 2), generator=g, device=device, dtype=torch.uint8
        )
        # Small ue8m0 block scales (2^-8 .. 2^-3) so the dequantized weights are
        # O(0.02 .. 0.75) — matching real (normalized) expert weights and keeping
        # the two-GEMM SwiGLU output O(1) for tight bf16 correctness checks. Avoids
        # the 0xFF NaN code. (Any byte 1..254 decodes correctly; this range just
        # keeps the test numerics well conditioned, like the INT4 generator.)
        scale = torch.randint(
            119, 125, (E, rows, cols // group_size),
            generator=g, device=device, dtype=torch.uint8,
        )
        return packed, scale

    w13, w13_scale = _packed(2 * ispp, hidden)
    w2, w2_scale = _packed(hidden, ispp)
    out = {
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
        "w13_ref": dequant_mxfp4_weight(w13, w13_scale, group_size),
        "w2_ref": dequant_mxfp4_weight(w2, w2_scale, group_size),
    }
    if with_bias:
        out["b13"] = (torch.rand(E, 2 * ispp, generator=g, device=device) * 0.2 - 0.1).to(
            torch.bfloat16
        )
        out["b2"] = (torch.rand(E, hidden, generator=g, device=device) * 0.2 - 0.1).to(
            torch.bfloat16
        )
    else:
        out["b13"] = None
        out["b2"] = None
    return out
