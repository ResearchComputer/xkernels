# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Shared helpers for the INT4 W4A16 fused-MoE GEMM (compressed-tensors
``pack-quantized``): weight unpack/dequant, a random-weight generator for tests,
and the ``moe_align_block_size`` sort/pad dispatch builder reused by both the
reference and the Triton backend so they consume identical routing.
"""

from __future__ import annotations

import torch

__all__ = [
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
    "moe_align_block_size_ep",
]


def dequant_w4a16(
    packed: torch.Tensor, scale: torch.Tensor, group_size: int = 32
) -> torch.Tensor:
    """Unpack + dequantize compressed-tensors W4A16 weights to bf16.

    Args:
        packed: ``[E, N, K // 8]`` int32 holding 8 ``uint4b8`` nibbles per int32,
            low nibble = lowest K index.
        scale: ``[E, N, K // group_size]`` group scales (bf16).
        group_size: quant group size along K.

    Returns:
        ``[E, N, K]`` bf16 dequantized weights, where
        ``W[e,n,k] = (((packed[e,n,k//8] >> (4*(k%8))) & 0xF) - 8) * scale[e,n,k//g]``.
    """
    E, N, kp = packed.shape
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    nib = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(E, N, kp * 8)  # [E,N,K]
    K = kp * 8
    w = (nib.float() - 8.0).view(E, N, K // group_size, group_size)
    w = w * scale.float().unsqueeze(-1)
    return w.view(E, N, K).to(torch.bfloat16)


def moe_align_block_size_ref(
    topk_ids: torch.Tensor, block_size: int, num_experts: int, truncate: bool = True
):
    """Sort/pad routed token-slots into per-expert blocks (issue #4 baseline).

    Reused so the optimized kernel and the reference consume identical dispatch.

    With ``truncate=True`` (default) ``expert_ids`` is trimmed to the used blocks;
    with ``truncate=False`` it is the full ``max_blocks`` length, unused trailing
    blocks set to sentinel 0 (fixed-shape, for graph capture / issue #18).

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``.
    """
    flat_e = topk_ids.flatten().long()
    total = flat_e.numel()
    pad = total
    max_pad = total + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.full((max_pad,), pad, dtype=torch.int32, device=topk_ids.device)
    order = torch.argsort(flat_e, stable=True)
    toks = torch.arange(total, device=topk_ids.device, dtype=torch.int32)[order]
    counts = torch.bincount(flat_e, minlength=num_experts + 1)
    expert_ids = []
    w = off = 0
    for e in range(num_experts):
        n = int(counts[e])
        grp = toks[off : off + n]
        nb = (n + block_size - 1) // block_size
        for b in range(nb):
            c = min(block_size, n - b * block_size)
            sorted_ids[w + b * block_size : w + b * block_size + c] = grp[
                b * block_size : b * block_size + c
            ]
            expert_ids.append(e)
        w += nb * block_size
        off += n
    if not expert_ids:
        expert_ids = [0]
    if not truncate:
        # Fixed-shape mode: pad to the full block count with sentinel 0 so the
        # output is graph-shaped. Unused blocks are past num_tokens_post_padded
        # and are never read by the GEMM consumer.
        max_blocks = (max_pad + block_size - 1) // block_size
        expert_ids = expert_ids + [0] * (max_blocks - len(expert_ids))
    return (
        sorted_ids,
        torch.tensor(expert_ids, dtype=torch.int32, device=topk_ids.device),
        torch.tensor([w], dtype=torch.int32, device=topk_ids.device),
    )


def moe_align_block_size_ep(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor,
    truncate: bool = True,
):
    """Expert-parallel (``ep_size > 1``) dispatch builder (issue #26).

    Under expert parallelism the router still emits **global** ``topk_ids`` over
    all ``num_experts`` experts, but each rank only holds a *subset* of the expert
    weights. ``expert_map`` is the per-rank lookup ``[num_experts]`` giving, for
    each global expert id, its **local** weight-row index in ``[0, E_local)`` —
    or ``-1`` if that expert lives on another rank.

    This remaps each routed slot's global expert id to its local row, sends every
    non-local slot to a sentinel id (``E_local``) that is *not* iterated by the
    block builder, and then reuses the standard per-expert sort/pad over the
    ``E_local`` local experts only. Non-local slots are therefore dropped from
    this rank's compute (the kernel never gathers them), so the GEMM produces this
    rank's **partial** MoE output; summing the partials across ranks (the
    production all-reduce) reconstructs the full dense result.

    Args:
        topk_ids: ``[M, top_k]`` int32 **global** expert ids.
        block_size: GEMM block size each (local) expert run is padded up to.
        num_experts: total number of **global** experts.
        expert_map: ``[num_experts]`` int (any width) mapping global id -> local
            row (``-1`` if not on this rank). ``E_local = (expert_map >= 0).sum()``
            and the local rows must be ``0..E_local-1`` (the standard contiguous
            per-rank expert slice).
        truncate: as in :func:`moe_align_block_size_ref`.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)`` where
        ``expert_ids`` holds **local** expert rows and indexes the rank-local
        ``[E_local, N, K // 8]`` weight tensor passed to the GEMM.
    """
    expert_map = expert_map.to(device=topk_ids.device)
    e_local = int((expert_map >= 0).sum().item())
    if e_local == 0:
        # This rank owns no experts: empty dispatch, the GEMM writes nothing.
        return moe_align_block_size_ref(
            torch.full_like(topk_ids, 0), block_size, 0, truncate=truncate
        )
    # Remap global -> local row; non-local global ids -> sentinel ``e_local`` so
    # they sort past the local experts and are skipped by the block builder
    # (which only iterates ``range(num_experts=e_local)``).
    sentinel = torch.full_like(expert_map, e_local)
    local_ids = torch.where(expert_map >= 0, expert_map, sentinel).to(torch.int64)
    remapped = local_ids[topk_ids.long()].to(torch.int32)
    return moe_align_block_size_ref(remapped, block_size, e_local, truncate=truncate)


def make_w4a16_weights(E: int, N: int, K: int, group_size: int = 32, *, device="cuda", seed=0):
    """Generate random valid W4A16 packed weights + scales for testing.

    Returns ``(packed [E,N,K//8] int32, scale [E,N,K//g] bf16, w_ref [E,N,K] bf16)``
    where ``w_ref`` is the exact dequantization of ``(packed, scale)``.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    assert K % 8 == 0 and K % group_size == 0
    # Random signed nibbles in [-8, 7] -> uint4b8 in [0, 15].
    q = torch.randint(0, 16, (E, N, K), generator=g, device=device, dtype=torch.int32)
    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    packed = (q.view(E, N, K // 8, 8) << shifts).sum(dim=-1).to(torch.int32)  # [E,N,K//8]
    scale = (
        torch.rand((E, N, K // group_size), generator=g, device=device, dtype=torch.float32)
        * 0.02
        + 0.005
    ).to(torch.bfloat16)
    w_ref = dequant_w4a16(packed, scale, group_size)
    return packed, scale, w_ref
