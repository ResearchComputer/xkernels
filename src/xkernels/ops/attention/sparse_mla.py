# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""fp8_ds_mla latent-KV format helpers for the DeepSeek-V4 decode path (issue #32).

The V4 paged KV cache stores each latent token (layout pinned from the tokenspeed
writer ``_deepseek_v4_fused_sparse_compress_cache_kernel``) as:

* a **value region** of ``nope_dim + rope_dim*2`` bytes: ``nope_dim`` fp8 e4m3
  (the kv_lora / value-bearing part) followed by ``rope_dim`` bf16 (the decoupled
  rope, score-only), and
* a **scale region** of ``nope_dim//quant_block`` uint8 exponents (``enc``) plus a
  pad byte, shared per ``quant_block`` group along nope.

Dequant: ``nope = fp8_e4m3(byte) * 2**(enc - 127)`` per group; ``rope = bf16``.
V4: ``nope_dim=448, rope_dim=64, quant_block=64`` → latent ``D=512``. The fp8 is
OCP e4m3 (``torch.float8_e4m3fn``), pinned by the writer's ``tl.float8e4nv`` cast.
"""

from __future__ import annotations

import torch

__all__ = [
    "FP8_DS_MLA_NOPE_DIM",
    "FP8_DS_MLA_ROPE_DIM",
    "FP8_DS_MLA_QUANT_BLOCK",
    "FP8_DS_MLA_HEAD_DIM",
    "dequant_fp8_ds_mla",
    "make_fp8_ds_mla_kv",
]

FP8_DS_MLA_NOPE_DIM = 448
FP8_DS_MLA_ROPE_DIM = 64
FP8_DS_MLA_QUANT_BLOCK = 64
FP8_DS_MLA_HEAD_DIM = FP8_DS_MLA_NOPE_DIM + FP8_DS_MLA_ROPE_DIM  # 512
_FP8_MAX = 448.0


def dequant_fp8_ds_mla(
    value_bytes: torch.Tensor,
    scale_bytes: torch.Tensor,
    *,
    nope_dim: int = FP8_DS_MLA_NOPE_DIM,
    rope_dim: int = FP8_DS_MLA_ROPE_DIM,
    quant_block: int = FP8_DS_MLA_QUANT_BLOCK,
) -> torch.Tensor:
    """Dequantize fp8_ds_mla rows to fp32 latent ``[..., nope_dim + rope_dim]``.

    Args:
        value_bytes: ``[..., nope_dim + rope_dim*2]`` uint8.
        scale_bytes: ``[..., nope_dim//quant_block (+pad)]`` uint8 exponents.
        nope_dim: fp8 (value-bearing) latent dim.
        rope_dim: bf16 (score-only) rope dim.
        quant_block: shared-scale group length along nope.
    """
    nope_fp8 = value_bytes[..., :nope_dim]
    rope_raw = value_bytes[..., nope_dim : nope_dim + rope_dim * 2]
    nope = nope_fp8.reshape(-1, nope_dim).view(torch.float8_e4m3fn).to(torch.float32)
    ng = nope_dim // quant_block
    enc = scale_bytes.reshape(-1, scale_bytes.shape[-1])[..., :ng].to(torch.int32) - 127
    # Broadcast multiply over quant groups instead of materializing a repeated
    # multiplier tensor (avoids a topk-sized intermediate on the decode path).
    mult = torch.exp2(enc.float()).unsqueeze(-1)  # [B, ng, 1]
    nope = nope.reshape(-1, ng, quant_block)
    nope = nope * mult
    nope = nope.reshape(-1, nope_dim)
    rope = rope_raw.reshape(-1, rope_dim * 2).view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope], dim=-1).reshape(value_bytes.shape[:-1] + (-1,))


def make_fp8_ds_mla_kv(
    num_rows: int,
    *,
    nope_dim: int = FP8_DS_MLA_NOPE_DIM,
    rope_dim: int = FP8_DS_MLA_ROPE_DIM,
    quant_block: int = FP8_DS_MLA_QUANT_BLOCK,
    device="cuda",
    seed: int = 0,
):
    """Random valid fp8_ds_mla rows + their exact dequantization.

    Returns ``(value_bytes [rows, nope+rope*2] uint8,
    scale_bytes [rows, nope//qb + 1] uint8, ref [rows, nope+rope] fp32)`` where
    ``ref == dequant_fp8_ds_mla(value_bytes, scale_bytes)``.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    nope = (torch.rand(num_rows, nope_dim, generator=g, device=device) * 2 - 1) * 4
    rope = torch.rand(num_rows, rope_dim, generator=g, device=device) * 2 - 1
    ng = nope_dim // quant_block
    nope_g = nope.reshape(num_rows, ng, quant_block)
    absmax = nope_g.abs().amax(dim=-1).clamp_min(1e-4)
    exps = torch.ceil(torch.log2(absmax / _FP8_MAX))  # [rows, ng]
    inv = torch.exp2(-exps).unsqueeze(-1)
    fp8 = (nope_g * inv).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    nope_deq = (fp8.to(torch.float32) * torch.exp2(exps).unsqueeze(-1)).reshape(
        num_rows, nope_dim
    )
    enc = (exps + 127).clamp(0, 255).to(torch.uint8)

    value_bytes = torch.empty(
        num_rows, nope_dim + rope_dim * 2, device=device, dtype=torch.uint8
    )
    value_bytes[:, :nope_dim] = fp8.reshape(num_rows, nope_dim).view(torch.uint8)
    value_bytes[:, nope_dim:] = rope.to(torch.bfloat16).view(torch.uint8)
    scale_bytes = torch.zeros(num_rows, ng + 1, device=device, dtype=torch.uint8)
    scale_bytes[:, :ng] = enc
    ref = torch.cat([nope_deq, rope.to(torch.bfloat16).to(torch.float32)], dim=-1)
    return value_bytes, scale_bytes, ref
