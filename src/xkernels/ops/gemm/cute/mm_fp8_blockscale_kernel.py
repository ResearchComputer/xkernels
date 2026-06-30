# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) fp32 GEMM kernel for ``mm_fp8_blockscale`` on sm_121.

Host-side dequant is done in torch (bit-identical to ``mm_fp8_blockscale_ref``'s
dequant — both operands -> fp32 with the DeepSeek fp32 block scales broadcast),
so the CUTE device kernel is a plain fp32 GEMM ``out = A @ B.T`` (A [M,K], B [N,K],
out [M,N], all fp32). This isolates the GPU-iterated CUTE work to a correct fp32
matmul; correctness of the dequant is inherited from torch.

Design notes (why no matrix engine on ds5, see the Impl Card):
  * MmaFP8Op (fp8 m16n8k32 -> fp32) requires CTK >= 13.1; ds5 has CTK 13.0 -> gated.
  * MmaSM120BlockScaledOp (sm_121 native) is MX microscaling (e8m0/e4m3), not this
    op's DeepSeek fp32 block=128 scales -> wrong contract.
  * bf16 MMA would fail the op's fp32 sweep point (rtol 1e-3).
So this card matches the op's defined parity target (dequant -> fp32 matmul) and
is honestly non-peak; perf is a documented follow-up gated on CTK 13.1.

Kernel strategy (correctness-first): one CTA per (_TILE_M x _TILE_N) output tile;
each thread owns ONE output element (m, n) and reduces over K with a scalar fp32
FMA loop, loading A[m, k] and B[n, k] via integer tensor indexing (the pattern
proven in ``smoke_vecadd``). No cross-thread reduction, no smem staging -> the
simplest correct fp32 GEMM. Vectorizing the K-reduction / tiling through smem is
the perf follow-up, not the correctness bar.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

from ..._cute_backend.launch import _cached_handle, _require_cuda

_TILE_M = 8
_TILE_N = 16
_BLOCK_THREADS = 128  # = _TILE_M * _TILE_N, exactly one output element per thread

# Compile-once / launch-many handle cache, keyed by the constexpr (M, N, K).
# The load-bearing rationale (why not @cute.jit __call__; why tensors-only
# launch) lives ONCE in ``ops/_cute_backend/launch.py`` — every CUTE card
# shares it. This card is where the ~223x end-to-end speedup was first measured
# (9307 us -> 41.6 us/call at identical numerics).
_COMPILED_HANDLE_CACHE: dict[tuple[int, int, int], object] = {}


@cute.kernel
def _fp32_matmul_kernel(
    gA: Tensor,      # [M, K] row-major fp32 (dequantized A)
    gB: Tensor,      # [K, N] row-major fp32 (dequantized B, host-transposed)
    gOut: Tensor,    # [M, N] row-major fp32
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
) -> None:
    """Each thread owns ONE output (m, n) and reduces over K with scalar fp32 FMA.

    Bounds predication is mandatory: when M or N is not a multiple of the tile,
    the padding threads would otherwise read/write out of bounds and corrupt
    the valid output region (the classic tiled-GEMM tail-CTA bug).
    """
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    bidy = nvvm.read_ptx_sreg_ctaid_y(T.i32())

    thr_m_in_tile = tidx // _TILE_N
    thr_n_in_tile = tidx % _TILE_N
    m = bidy * _TILE_M + thr_m_in_tile
    n = bidx * _TILE_N + thr_n_in_tile

    if m < M and n < N:
        a_row = gA[(m, None)]    # A[m, :], K-row (contiguous in K; broadcast across the warp)
        b_col = gB[(None, n)]    # B[:, n], K-column (coalesced across the warp)

        # Kahan compensated summation: fp8-block-scaled operands have per-block
        # magnitude variation, so a naive sequential sum loses low-order bits
        # vs torch's tree reduction and can exceed the op's fp32 rtol (1e-3) on
        # ill-conditioned elements. The compensation term `c` recovers the lost
        # precision, bringing sequential-sum agreement to ~1e-6.
        #
        # (d) NOTE: a 2-way Kahan (two independent even/odd chains merged at the end)
        # was tried to attack the residual scoreboard stall (11.6 cyc) but REGRESSED
        # per ncu: scoreboard stall 11.6->15.4 cyc, duration 90->102us (occupancy
        # rose 53->68% but didn't translate to speed). The single chain is already at
        # its ILP ceiling for this (8x16, one-output-per-thread) tile; the 2-way adds
        # merge-dependency + register pressure that outweigh the ILP gain. The real
        # perf lever is the matrix engine (gated on CTK 13.1) — not scalar unrolling.
        acc = cutlass.Float32(0.0)
        c = cutlass.Float32(0.0)
        k = cutlass.Int32(0)
        while k < K:
            y = a_row[(k,)] * b_col[(k,)] - c
            t = acc + y
            c = (t - acc) - y
            acc = t
            k = k + 1

        gOut[(m, n)] = acc


@cute.jit
def _fp32_matmul(
    A: Tensor,
    B: Tensor,
    Out: Tensor,
    M: cutlass.Constexpr,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
) -> None:
    """Host JIT: launch one CTA per (_TILE_M x _TILE_N) output tile."""
    _fp32_matmul_kernel(
        A, B, Out, M, N, K,
    ).launch(
        grid=[(N + _TILE_N - 1) // _TILE_N, (M + _TILE_M - 1) // _TILE_M, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def fp32_matmul_cute(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """fp32 GEMM ``out [M,N] = a [M,K] @ b [N,K].T`` via a JIT CUTE DSL kernel.

    Uses the compile-once / launch-many path: the first call per (M,N,K) shape pays
    a ~120 ms one-time JIT to build a reusable ``cute.compile`` handle; every
    subsequent call launches in ~40 us (vs ~9.3 ms for the @cute.jit __call__ path).
    """
    M, K = a.shape
    N = b.shape[0]
    assert b.shape == (N, K), f"b must be [N,K]=[{N},{K}], got {tuple(b.shape)}"
    _require_cuda(a)
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Transpose B to (K,N) row-major so the K-reduction reads a CONTIGUOUS column
    # across the warp (coalesced) instead of a K-strided row. The baseline ncu
    # profile (scripts/ds5_ncu_baseline.py) flagged the LG-throttle stall ("waiting
    # for the L1/LG memory queue to be not full") at 71.6% of cycles, caused by the
    # 16-way strided B[n,k] access of consecutive-n threads. Coalescing B[k,n]
    # removes it (see diagnose-memory-bound).
    bT = b.t().contiguous()     # (K, N) row-major
    gA = from_dlpack(a)         # (M, K)
    gB = from_dlpack(bT)        # (K, N)
    gOut = from_dlpack(out)     # (M, N)

    key = (M, N, K)
    handle = _cached_handle(
        _COMPILED_HANDLE_CACHE, key, _fp32_matmul,
        (gA, gB, gOut), (M, N, K),
    )
    handle(gA, gB, gOut)  # fast launch — tensors only (constexpr baked in)
    return out
