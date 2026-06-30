# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) fp32 kernel for ``hc_prenorm_gemm`` on sm_121.

DeepSeek-V4 MHC hidden-compression prenorm GEMM — a fused GEMM + per-row
squared-sum sharing the SAME K-reduction axis:

    gemm_out_mul[0, t, n]  = sum_k  a[t,k] * fn[n,k]      ( = a @ fn.T )
    gemm_out_sqrsum[0, t]  = sum_k  a[t,k] ** 2            (RMS prenorm partial)

Host-side dtype plumbing matches the reference + the passing triton card: ``a``
is upcast to fp32 on the host (bit-identical to the reference's ``a.float()``);
``fn`` is already fp32. The device kernel is PURE fp32 with Kahan-compensated
K-reductions (both the GEMM accumulation and the squared-sum), marginally MORE
accurate than torch's default reduction. Outputs stay fp32 (the op's contract).

Split-K: the reference uses the *trivial* partition (full result in split 0,
zeros elsewhere), so this card does the same — correct for ANY ``n_splits`` (the
downstream consumer only sums over the split axis; the sum is invariant). The
kernel always writes split 0; the host allocates the [n_splits, ...] outputs as
zeros so splits 1..n_splits-1 are the reference's zero partials.

Design — one CTA per output row ``t`` (grid = T blocks); 128 threads tile the
output columns ``n`` (thread-stride). This is a *skinny* GEMM (sweep: T≤37,
N≤24, K≤256), so the largest axis (K) is the per-thread serial reduction — no
block-wide K-reduction / SMEM tiling is needed at this size. The squared-sum is
per-row, so thread 0 computes it (O(K), cheap); every thread owns ≥0 GEMM
columns and writes independently (distinct ``n`` → no write race).

Coalescing fix (mirrors mm_fp8_blockscale): ``fn`` arrives as ``[N, K]`` row-major,
so ``fn[n, k]`` for varying ``n`` (different threads) at fixed ``k`` is
``K``-strided → uncoalesced. The host transposes to ``fn_t [K, N]`` row-major so
``fn_t[k, n]`` for varying ``n`` at fixed ``k`` is contiguous → coalesced across
the active threads. (Math-preserving: ``fn_t[k,n] == fn[n,k]``.)
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

_BLOCK_THREADS = 128

# Compile-once / launch-many handle cache, keyed by the constexpr (T, K, N).
# See mm_fp8_blockscale_kernel._COMPILED_HANDLE_CACHE for the full rationale.
_COMPILED_HANDLE_CACHE: "dict[tuple[int, int, int], object]" = {}


@cute.kernel
def _prenorm_gemm_kernel(
    gA: Tensor,     # [T, K]      fp32 (host-upcast from a.dtype)
    gFnT: Tensor,   # [K, N]      fp32 (host-transposed fn, for coalesced n-read)
    gMul: Tensor,   # [n_splits, T, N] fp32 (writes split 0 only)
    gSqr: Tensor,   # [n_splits, T]    fp32 (writes split 0 only)
    ROWS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    N: cutlass.Constexpr,
) -> None:
    """One CTA per output row t; threads tile the columns n (thread-stride)."""
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    t = bidx

    # Thread 0: per-row squared-sum of a[t,:] (the fused RMS-prenorm partial).
    # Kahan-compensated, matching the reference's fp32 reduction order.
    if tidx == 0:
        s_acc = cutlass.Float32(0.0)
        s_c = cutlass.Float32(0.0)
        k = cutlass.Int32(0)
        while k < K:
            ak = gA[(t, k)]
            term = ak * ak
            y_ = term - s_c
            t_ = s_acc + y_
            s_c = (t_ - s_acc) - y_
            s_acc = t_
            k = k + 1
        gSqr[(0, t)] = s_acc

    # Every thread: thread-stride over the columns n, full K reduction (Kahan).
    n = tidx
    while n < N:
        g_acc = cutlass.Float32(0.0)
        g_c = cutlass.Float32(0.0)
        k = cutlass.Int32(0)
        while k < K:
            ak = gA[(t, k)]
            fnk = gFnT[(k, n)]
            term = ak * fnk
            y_ = term - g_c
            t_ = g_acc + y_
            g_c = (t_ - g_acc) - y_
            g_acc = t_
            k = k + 1
        gMul[(0, t, n)] = g_acc
        n = n + _BLOCK_THREADS


@cute.jit
def _prenorm_gemm(
    A: Tensor,
    FnT: Tensor,
    Mul: Tensor,
    Sqr: Tensor,
    ROWS: cutlass.Constexpr,
    K: cutlass.Constexpr,
    N: cutlass.Constexpr,
) -> None:
    """Host JIT: one CTA per output row (ROWS blocks of 128 threads)."""
    _prenorm_gemm_kernel(A, FnT, Mul, Sqr, ROWS, K, N).launch(
        grid=[ROWS, 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def hc_prenorm_gemm_cute(
    a: "torch.Tensor", fn: "torch.Tensor", *, n_splits: int  # type: ignore[name-defined]
) -> "tuple[torch.Tensor, torch.Tensor]":  # type: ignore[name-defined]
    """Fused GEMM + per-row squared-sum via a JIT CUTE DSL kernel (pure fp32).

    ``a`` is upcast to fp32 on the host (bit-identical to the reference); the
    kernel runs pure fp32 with Kahan K-reductions; outputs stay fp32 (the op's
    contract). Writes the full result to split 0 and leaves splits 1..n_splits-1
    as zeros — the reference's trivial split-K partition (sum-invariant). Uses
    the compile-once / launch-many path keyed by ``(T, K, N)``.
    """
    import torch

    if not getattr(a, "is_cuda", False):
        # GPU-only: verify_parity() hardcodes device='cpu'; raising here lets the
        # harness record CUDA as a caught backend error instead of segfaulting.
        raise RuntimeError(
            "CUTE DSL kernel requires CUDA tensors; got device='cpu'. "
            "verify_parity() hardcodes device='cpu' and cannot exercise a GPU-only card."
        )
    T, K = a.shape
    N = fn.shape[0]
    af = a.to(torch.float32).contiguous()
    # Transpose fn [N,K] -> [K,N] for coalesced reads across the N-tiled threads.
    fn_t = fn.to(torch.float32).contiguous().t().contiguous()
    # Allocate as zeros: split 0 gets the full result, splits 1..n_splits-1 stay
    # zero (the reference's trivial split-K partition — sum-invariant).
    gemm_out_mul = torch.zeros(n_splits, T, N, device=a.device, dtype=torch.float32)
    gemm_out_sqrsum = torch.zeros(n_splits, T, device=a.device, dtype=torch.float32)
    if T == 0:
        return gemm_out_mul, gemm_out_sqrsum

    gA = from_dlpack(af)
    gFnT = from_dlpack(fn_t)
    gMul = from_dlpack(gemm_out_mul)
    gSqr = from_dlpack(gemm_out_sqrsum)

    key = (T, K, N)
    handle = _COMPILED_HANDLE_CACHE.get(key)
    if handle is None:
        _prenorm_gemm(gA, gFnT, gMul, gSqr, T, K, N)
        torch.cuda.synchronize()
        handle = cute.compile(_prenorm_gemm, gA, gFnT, gMul, gSqr, T, K, N)
        _COMPILED_HANDLE_CACHE[key] = handle

    # Fast launch — tensors only (constexpr baked in at compile; see cache note).
    handle(gA, gFnT, gMul, gSqr)
    return gemm_out_mul, gemm_out_sqrsum
