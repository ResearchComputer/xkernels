# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton split-K MHC prenorm GEMM for AMD MI300A (gfx942, CDNA3), issue #36.

One program per ``(split, row-tile)``. Split ``s`` owns a contiguous K-range (the
``ceil_div(K, BLOCK_K)`` K-blocks partitioned as evenly as possible across
``n_splits``), streams it in ``BLOCK_K`` chunks, and accumulates both
``A[:, krange] @ fn[:, krange].T`` (via ``tl.dot`` with a transposed ``fn`` tile —
``fn`` is stored ``[N, K]``) and the per-row ``Σ A²`` from the same A loads. The
downstream TileLang post-fusion sums the per-split partials, so the disjoint
K-partition reproduces the full ``F.linear``/sqsum exactly. Empty splits (when
``n_splits > num_kblocks``) still run and store zeros, keeping every
``torch.empty`` output slot defined. Compute is fp32 (CDNA3 has no TF32).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from .configs import resolve_mhc_gemm_config

__all__ = ["hc_prenorm_gemm_triton", "hc_prenorm_gemm_kernel"]


@triton.jit
def hc_prenorm_gemm_kernel(
    a_ptr, fn_ptr, mul_ptr, sqr_ptr,
    T, K, N, n_splits, num_kblocks,
    stride_at, stride_ak,
    stride_fn, stride_fk,
    stride_ms, stride_mt, stride_mn,
    stride_ss, stride_st,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    s = tl.program_id(0)
    m = tl.program_id(1)
    rows = m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    ns = tl.arange(0, BLOCK_N)
    n_mask = ns < N

    # Contiguous K-block range owned by this split (even partition).
    kb_lo = s * num_kblocks // n_splits
    kb_hi = (s + 1) * num_kblocks // n_splits
    k_lo = kb_lo * BLOCK_K
    k_hi = tl.minimum(kb_hi * BLOCK_K, K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(k_lo, k_hi, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        a_tile = tl.load(
            a_ptr + rows[:, None] * stride_at + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        # fn is [N, K]; gather a [BLOCK_K, BLOCK_N] tile (K on axis0, N on axis1).
        fn_tile = tl.load(
            fn_ptr + ks[:, None] * stride_fk + ns[None, :] * stride_fn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a_tile, fn_tile)
        sq += tl.sum(a_tile * a_tile, axis=1)

    tl.store(
        mul_ptr + s * stride_ms + rows[:, None] * stride_mt + ns[None, :] * stride_mn,
        acc, mask=row_mask[:, None] & n_mask[None, :],
    )
    tl.store(sqr_ptr + s * stride_ss + rows * stride_st, sq, mask=row_mask)


def hc_prenorm_gemm_triton(a, fn, *, n_splits):
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    a = a.contiguous()
    fn = fn.contiguous()
    T, K = a.shape
    N = fn.shape[0]
    if fn.shape[1] != K:
        raise ValueError(f"fn must be [N, K] with K={K}, got {tuple(fn.shape)}")
    mul = torch.empty(n_splits, T, N, device=a.device, dtype=torch.float32)
    sqr = torch.empty(n_splits, T, device=a.device, dtype=torch.float32)
    if T == 0:
        return mul, sqr  # no rows; nothing to write

    # Perf pass (#39): block sizes + CDNA3 lowering knobs are resolved from a
    # config (env-overridable for the on-device sweep). The default reproduces
    # the #36 launch (BLOCK_M=BLOCK_K=64), so behavior is unchanged by default.
    # The split-K partition is by k-block range, so any BLOCK_K is correct (the
    # downstream only sums over splits — see configs.py / docs/issue-36).
    cfg = resolve_mhc_gemm_config()
    BLOCK_M = int(cfg["BLOCK_M"])
    BLOCK_K = int(cfg["BLOCK_K"])
    BLOCK_N = max(16, triton.next_power_of_2(N))
    num_kblocks = triton.cdiv(K, BLOCK_K)
    grid = (n_splits, triton.cdiv(T, BLOCK_M))
    # AMD-only lowering kwargs: read by the Triton AMD backend, ignored elsewhere
    # (and under TRITON_INTERPRET=1), so the same call stays portable.
    amd_knobs = {
        "waves_per_eu": int(cfg.get("waves_per_eu", 0)),
        "matrix_instr_nonkdim": int(cfg.get("matrix_instr_nonkdim", 16)),
        "kpack": int(cfg.get("kpack", 2)),
    }
    hc_prenorm_gemm_kernel[grid](
        a, fn, mul, sqr,
        T, K, N, n_splits, num_kblocks,
        a.stride(0), a.stride(1),
        fn.stride(0), fn.stride(1),
        mul.stride(0), mul.stride(1), mul.stride(2),
        sqr.stride(0), sqr.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        num_warps=int(cfg.get("num_warps", 4)),
        num_stages=int(cfg.get("num_stages", 2)),
        **amd_knobs,
    )
    return mul, sqr


register("hc_prenorm_gemm", Backend.TRITON)(hc_prenorm_gemm_triton)
