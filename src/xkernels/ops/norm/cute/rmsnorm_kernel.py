# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) fp32 RMSNorm kernel for ``dual_rmsnorm`` on sm_121.

Host-side dtype plumbing mirrors the reference + the passing triton card:
``x`` / ``w`` are upcast to fp32 on the host (bit-identical to the reference's
``x.float()``), the CUTE device kernel is a PURE fp32 row-RMSNorm, and the host
casts the result back to ``x.dtype`` (round-to-nearest — identical to a bf16
store). This isolates the GPU-iterated work to a correct fp32 normalize; the
ordering (fp32-multiply-then-cast) matches the triton card, well inside the op's
calibrated bf16 rtol 0.016.

Design — one CTA per token row, 128 threads (4 warps), thread-stride over the
feature dim ``d`` (coalesced: consecutive lanes read consecutive columns):

  Pass 1 (reduce): each thread walks ``col = tid, tid+128, ...`` loading
    ``x[row, col]`` (fp32) and Kahan-accumulating ``x*x``. A block-wide reduction
    (``warp_reduction_sum`` per warp -> lane-0 writes its warp's partial to SMEM
    -> ``sync_threads`` -> thread 0 folds the 4 partials -> ``math.rsqrt`` ->
    broadcasts the per-row scale via SMEM) yields the rsqrt-of-mean-squares on
    every thread.
  Pass 2 (apply): each thread re-walks its columns, loads ``x`` again and the
    weight ``w[col]``, and stores ``x * scale * w``.

The 2-pass design re-reads ``x`` once from DRAM (no SMEM x-cache). That is honest
for a memory-bound normalize; an SMEM x-cache (read-once) is a documented perf
follow-up, not a correctness bar. The reduction primitives were all confirmed on
sm_121 by ``scripts/archive/ds5-probes/ds5_dsl_rowsum_probe.py`` and
``scripts/archive/ds5-probes/ds5_dsl_math_probe2.py``.
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import math, nvvm
from cutlass.cute.arch import alloc_smem, sync_threads, warp_reduction_sum
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

from ..._cute_backend.launch import _cached_handle, _require_cuda

_BLOCK_THREADS = 128
_NUM_WARPS = _BLOCK_THREADS // 32  # = 4

# Compile-once / launch-many handle cache, keyed by (T, D, eps). The
# load-bearing rationale (why not @cute.jit __call__; why tensors-only launch)
# lives ONCE in ``ops/_cute_backend/launch.py`` — every CUTE card shares it.
# eps is in the key because it is a constexpr too: a key without it would
# silently reuse a handle compiled for a different eps.
_COMPILED_HANDLE_CACHE: dict[tuple[int, int, float], object] = {}


@cute.kernel
def _rmsnorm_kernel(
    gX: Tensor,       # [T, D] row-major fp32 (host-upcast from x.dtype)
    gW: Tensor,       # [D]   row-major fp32 (host-upcast from w.dtype)
    gOut: Tensor,     # [T, D] row-major fp32
    T_rows: cutlass.Constexpr,
    D: cutlass.Constexpr,
    eps: cutlass.Constexpr,
) -> None:
    """One CTA per row: fp32 mean-of-squares -> rsqrt -> scale*x*w."""
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    row = bidx

    # SMEM holds the per-warp partial sums + the broadcast scale (1 slot).
    smem = alloc_smem(cutlass.Float32, _NUM_WARPS + 1)
    warp_id = tidx // 32
    lane = tidx % 32

    # ---- Pass 1: thread-stride fp32 sum-of-squares (Kahan) ----
    acc = cutlass.Float32(0.0)
    c = cutlass.Float32(0.0)
    col = tidx
    while col < D:
        x = gX[(row, col)]
        sq = x * x
        y = sq - c
        t_ = acc + y
        c = (t_ - acc) - y
        acc = t_
        col = col + _BLOCK_THREADS

    # Warp reduce: every lane now holds the warp's sum.
    acc = warp_reduction_sum(acc, threads_in_group=32)
    if lane == 0:
        smem[warp_id] = acc
    sync_threads()

    # Thread 0 folds the 4 warp partials -> mean -> rsqrt -> scale slot.
    if tidx == 0:
        total = cutlass.Float32(0.0)
        w = cutlass.Int32(0)
        while w < _NUM_WARPS:
            total = total + smem[w]
            w = w + 1
        mean = total / cutlass.Float32(D)
        scale = math.rsqrt(mean + cutlass.Float32(eps))
        smem[_NUM_WARPS] = scale
    sync_threads()

    scale = smem[_NUM_WARPS]

    # ---- Pass 2: re-walk columns, apply x*scale*w, store ----
    col = tidx
    while col < D:
        x = gX[(row, col)]
        wv = gW[(col,)]
        gOut[(row, col)] = x * scale * wv
        col = col + _BLOCK_THREADS


@cute.jit
def _rmsnorm(
    X: Tensor,
    W: Tensor,
    Out: Tensor,
    T_rows: cutlass.Constexpr,
    D: cutlass.Constexpr,
    eps: cutlass.Constexpr,
) -> None:
    """Host JIT: one CTA per row (T_rows blocks of 128 threads)."""
    _rmsnorm_kernel(
        X, W, Out, T_rows, D, eps,
    ).launch(
        grid=[T_rows, 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def rmsnorm_cute(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """fp32 RMSNorm ``out = x * rsqrt(mean(x^2) + eps) * w`` via a JIT CUTE kernel.

    Inputs are upcast to fp32 on the host (bit-identical to the reference); the
    kernel runs pure fp32; the result is cast back to ``x.dtype`` by the caller.
    Uses the compile-once / launch-many path keyed by ``(T, D, eps)``.
    """
    _require_cuda(x)
    xf = x.to(torch.float32)
    wf = w.to(torch.float32)
    out = torch.empty_like(xf)
    T_, D = xf.shape

    gX = from_dlpack(xf)
    gW = from_dlpack(wf)
    gOut = from_dlpack(out)

    key = (T_, D, eps)
    handle = _cached_handle(
        _COMPILED_HANDLE_CACHE, key, _rmsnorm,
        (gX, gW, gOut), (T_, D, eps),
    )
    handle(gX, gW, gOut)  # fast launch — tensors only (constexpr baked in)
    return out
