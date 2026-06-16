# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Native fp8 MFMA block-scale dense GEMM for AMD MI300A (gfx942, CDNA3), issue #41.

The fast-path counterpart to the portable dequant-then-dot kernel
(``mm_fp8_blockscale_kernel.py``, #40). Computes

    out[M, N] = A_deq @ B_deq.T

via **two-level (block-promoted) accumulation**: per 128-K quant block, a raw
``fp8.fp8`` ``tl.dot`` accumulates into an fp32 block-accumulator (the native
CDNA3 fp8 MFMA), then promotes into the main fp32 accumulator scaled by the
per-row A group-scale and the per-N-block B scale -- because both scales are
constant within a 128-K block:

    out = SUM_kb a_s[m,kb] * b_s[n//128,kb] * SUM_{k in block kb} A_fp8[m,k] B_fp8[n,k]

The operands enter ``tl.dot`` in their fp8 dtype (no pre-dequant) -- that is what
routes to the fp8 matrix path. The kernel is fp8-format-agnostic (``e4m3fn`` or
``e4m3fnuz``); whichever the caller supplies is what the MFMA consumes. The
per-column ``b_s[cols//128, kb]`` load generalizes #40's "BLOCK_N must divide 128
-> single scalar" constraint, so ``BLOCK_N`` is a free tuning knob.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .configs import get_fp8_gemm_config

__all__ = ["mm_fp8_blockscale_mfma_triton", "mm_fp8_blockscale_mfma_kernel"]


@triton.jit
def mm_fp8_blockscale_mfma_kernel(
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
    GROUP_M: tl.constexpr,
    waves_per_eu: tl.constexpr = 0,
    matrix_instr_nonkdim: tl.constexpr = 16,
    kpack: tl.constexpr = 2,
):
    # L2-friendly program swizzle (group along M), like the MoE INT4 kernel.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    nb = cols // BLOCK  # [BLOCK_N] N-quant-block per output column

    kt = tl.cdiv(K, BLOCK)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kb in range(0, kt):
        pacc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_base = kb * BLOCK
        for ki in tl.static_range(0, BLOCK, BLOCK_K):
            ks = k_base + ki + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            a = tl.load(
                a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            )
            b = tl.load(
                b_ptr + cols[None, :] * stride_bn + ks[:, None] * stride_bk,
                mask=col_mask[None, :] & k_mask[:, None], other=0.0,
            )
            pacc += tl.dot(a, b)  # fp8 operands -> native fp8 MFMA, fp32 accumulate
        a_sc = tl.load(as_ptr + rows * stride_asm + kb * stride_ask, mask=row_mask, other=0.0)
        b_sc = tl.load(bs_ptr + nb * stride_bsn + kb * stride_bsk, mask=col_mask, other=0.0)
        acc += pacc * a_sc[:, None] * b_sc[None, :]

    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def mm_fp8_blockscale_mfma_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    config: dict | None = None,
) -> torch.Tensor:
    """Native fp8 MFMA fp8 block-scale GEMM (gfx942). See module docstring.

    ``a_fp8``/``b_fp8`` may be ``float8_e4m3fn`` or ``float8_e4m3fnuz``; the kernel
    dots whatever it is given. ``config`` overrides the baked launch config.
    """
    a_fp8 = a_fp8.contiguous()
    b_fp8 = b_fp8.contiguous()
    a_scales = a_scales.contiguous().float()
    b_scales = b_scales.contiguous().float()

    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    if b_fp8.shape[1] != K:
        raise ValueError(f"b_fp8 must be [N, K] with K={K}, got {tuple(b_fp8.shape)}")
    if block != 128:
        raise ValueError(f"native fp8 MFMA path requires block=128, got {block}")
    kt = (K + block - 1) // block
    nt = (N + block - 1) // block
    if tuple(a_scales.shape) != (M, kt):
        raise ValueError(f"a_scales must be [M, kt] = [{M}, {kt}], got {tuple(a_scales.shape)}")
    if tuple(b_scales.shape) != (nt, kt):
        raise ValueError(f"b_scales must be [{nt}, {kt}], got {tuple(b_scales.shape)}")

    c = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return c

    cfg = config or get_fp8_gemm_config(M, N, K)
    if block % cfg["BLOCK_K"]:
        raise ValueError(f"BLOCK_K={cfg['BLOCK_K']} must divide block={block}")
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    mm_fp8_blockscale_mfma_kernel[grid](
        a_fp8, a_scales, b_fp8, b_scales, c,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK=block,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        kpack=cfg["kpack"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return c
