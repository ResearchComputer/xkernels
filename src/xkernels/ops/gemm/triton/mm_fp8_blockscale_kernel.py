# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton fp8 block-scale dense GEMM for AMD MI300A (gfx942, CDNA3), issue #38.

Portable gfx942 replacement for the NVIDIA-only ``triton_mm_fp8_blockscale`` /
``deep_gemm_mm_fp8_blockscale``. Computes

    out[M, N] = A_deq @ B_deq.T

with ``A [M, K]`` per-token-group fp8 e4m3 (scale ``A_scales [M, kt]``, one scale
per contiguous ``BLOCK`` of K) and ``B [N, K]`` per-block fp8 e4m3 (scale
``B_scales [nt, kt]``, one scale per ``BLOCK×BLOCK`` tile), Linear orientation.

One program per ``(row-tile, col-tile)``. The compute tile sizes ``BLOCK_M`` /
``BLOCK_N`` / ``BLOCK_K`` are decoupled from the quant ``block`` but constrained so
each tile lands inside exactly one quant block on both the N and K axes
(``block % BLOCK_N == 0`` and ``block % BLOCK_K == 0``). That keeps every tile's
``B`` block-scale a single scalar (``bs[col_block, k_block]``) and its ``A``
group-scale per-row (``as[row, k_block]``), so dequant is a cheap broadcast.

The K loop streams ``BLOCK_K`` columns at a time: load the fp8 ``A``/``B`` tiles,
upcast to fp32, multiply by the two block scales, and accumulate ``tl.dot`` in
fp32. fp32 ``tl.dot`` lands on the MFMA path on CDNA3 and — unlike the
``torch_mm_fp8_blockscale`` reference — never materializes the full dequantized
operands in DRAM (dequant happens per tile, in registers). Tiles are kept small
(64³) so the fp32 LDS footprint stays under the 64 KB CDNA3 limit.

Non-block-aligned M/N/K are handled by masking; a trailing partial block uses its
(correct) per-block scale just like a full one.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["mm_fp8_blockscale_triton", "mm_fp8_blockscale_kernel"]


@triton.jit
def mm_fp8_blockscale_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_bn, stride_bk,
    stride_bsn, stride_bsk,
    stride_cm, stride_cn,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    # N-block this tile lives in (BLOCK_N divides BLOCK, so it is constant).
    nb = (pid_n * BLOCK_N) // BLOCK

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        kb = k0 // BLOCK  # quant K-block (BLOCK_K divides BLOCK, so constant here)

        # A tile [BLOCK_M, BLOCK_K] fp8 -> fp32, scaled by per-row group scale.
        a_tile = tl.load(
            a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        a_scale = tl.load(
            as_ptr + rows * stride_asm + kb * stride_ask,
            mask=row_mask, other=0.0,
        )
        a_tile = a_tile * a_scale[:, None]

        # B tile [BLOCK_K, BLOCK_N] fp8 -> fp32; B is [N, K] so gather K on axis0,
        # N on axis1 (transposed load). Scaled by the single per-block scalar.
        b_tile = tl.load(
            b_ptr + cols[None, :] * stride_bn + ks[:, None] * stride_bk,
            mask=col_mask[None, :] & k_mask[:, None], other=0.0,
        ).to(tl.float32)
        b_scale = tl.load(bs_ptr + nb * stride_bsn + kb * stride_bsk)
        b_tile = b_tile * b_scale

        # Cast the (already block-scaled) operands to bf16 so tl.dot lands on the
        # bf16 MFMA path on CDNA3 — far faster than an fp32 dot — while the
        # accumulator stays fp32. ``DOT_BF16=False`` keeps the fp32 dot (exact
        # parity with the reference) for environments that prefer it.
        if DOT_BF16:
            acc += tl.dot(a_tile.to(tl.bfloat16), b_tile.to(tl.bfloat16))
        else:
            acc += tl.dot(a_tile, b_tile)

    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def mm_fp8_blockscale_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,
) -> torch.Tensor:
    """fp8 block-scale dense GEMM (Triton, gfx942). See module docstring.

    ``dot_bf16`` (opt-in, default False) casts the block-scaled operands to bf16
    so ``tl.dot`` runs on the CDNA3 bf16 MFMA path (accumulating in fp32) —
    faster but only ~bf16-bit accurate. Default False keeps the exact-fp32 dot.
    """
    a_fp8 = a_fp8.contiguous()
    b_fp8 = b_fp8.contiguous()
    a_scales = a_scales.contiguous().float()
    b_scales = b_scales.contiguous().float()

    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    if b_fp8.shape[1] != K:
        raise ValueError(f"b_fp8 must be [N, K] with K={K}, got {tuple(b_fp8.shape)}")
    kt = (K + block - 1) // block
    nt = (N + block - 1) // block
    if tuple(a_scales.shape) != (M, kt):
        raise ValueError(
            f"a_scales must be [M, kt] = [{M}, {kt}], got {tuple(a_scales.shape)}"
        )
    if tuple(b_scales.shape) != (nt, kt):
        raise ValueError(
            f"b_scales must be [{nt}, {kt}], got {tuple(b_scales.shape)}"
        )

    c = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return c

    # Compute tiles (must divide the quant block on the N and K axes so each tile
    # carries a single B block-scale; small enough that the LDS footprint stays
    # under the 64 KB CDNA3 limit). bf16 operands halve LDS, so the bf16 path can
    # afford the full BLOCK_K = block.
    BLOCK_M = 64
    BLOCK_N = min(64, block)
    BLOCK_K = min(block, 128) if dot_bf16 else min(64, block)
    if block % BLOCK_N or block % BLOCK_K:
        raise ValueError(
            f"block={block} must be a multiple of BLOCK_N={BLOCK_N} and "
            f"BLOCK_K={BLOCK_K}"
        )
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    mm_fp8_blockscale_kernel[grid](
        a_fp8, a_scales, b_fp8, b_scales, c,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK=block,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        DOT_BF16=dot_bf16,
        num_warps=4,
    )
    return c
