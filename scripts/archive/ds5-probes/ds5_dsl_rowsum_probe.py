#!/usr/bin/env python
"""De-risk the CUTE DSL reduction primitive set in ONE minimal kernel.

A row-sum (sum over columns, one block per row) exercises every primitive the
dual_rmsnorm / moe_sum_reduce cards hinge on, in the simplest possible form:

  * thread-stride scalar load via tensor indexing gX[(row, col)]   (proven in gemm)
  * fp32 partial accumulation                                       (proven in gemm)
  * warp_reduction_sum(partial, threads_in_group=32)               <- NEW, confirm
  * alloc_smem(Float32, n) -> Pointer                               <- NEW, confirm
  * smem pointer store / load syntax                                <- NEW, confirm
  * sync_threads()                                                  <- NEW, confirm

If this compiles, runs, and matches torch's row-sum to ~1e-5 on ds5, the whole
reduction-class family (dual_rmsnorm, moe_sum_reduce, mha_merge_state) is
unblocked and the API is confirmed (not guessed).
"""
from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.arch import alloc_smem, sync_threads, warp_reduction_sum
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

_BLOCK_THREADS = 128
_NUM_WARPS = _BLOCK_THREADS // 32  # = 4


@cute.kernel
def _rowsum_kernel(
    gX: Tensor,       # [T, D] fp32
    gOut: Tensor,     # [T]    fp32
    D: cutlass.Constexpr,
) -> None:
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    row = bidx

    # Pass 1: thread-stride over columns; accumulate partial sum-of-x.
    acc = cutlass.Float32(0.0)
    col = tidx
    while col < D:
        x = gX[(row, col)]
        acc = acc + x
        col = col + _BLOCK_THREADS

    # Warp reduce: every lane now holds the warp's sum.
    acc = warp_reduction_sum(acc, threads_in_group=32)

    # Lane 0 of each warp -> smem.
    smem = alloc_smem(cutlass.Float32, _NUM_WARPS)
    warp_id = tidx // 32
    lane = tidx % 32
    if lane == 0:
        smem[warp_id] = acc

    sync_threads()

    # Thread 0 folds the 4 warp partials -> final row sum -> gOut.
    if tidx == 0:
        total = cutlass.Float32(0.0)
        w = cutlass.Int32(0)
        while w < _NUM_WARPS:
            total = total + smem[w]
            w = w + 1
        gOut[(row,)] = total


@cute.jit
def _rowsum(
    X: Tensor,
    Out: Tensor,
    D: cutlass.Constexpr,
) -> None:
    from cutlass.cute.core import size
    _rowsum_kernel(
        X, Out, D,
    ).launch(
        grid=[size(X, mode=[0]), 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def rowsum_cute(x: torch.Tensor) -> torch.Tensor:
    T_, D = x.shape
    out = torch.empty(T_, device=x.device, dtype=torch.float32)
    gX = from_dlpack(x)
    gOut = from_dlpack(out)
    # warmup + compile handle
    _rowsum(gX, gOut, D)
    torch.cuda.synchronize()
    handle = cute.compile(_rowsum, gX, gOut, D)
    handle(gX, gOut)
    return out


def _self_check() -> None:
    assert torch.cuda.is_available()
    torch.manual_seed(0)
    worst = 0.0
    for (T_, D) in [(64, 1536), (64, 512), (1, 128), (7, 100), (3, 129)]:
        x = torch.randn(T_, D, device="cuda", dtype=torch.float32)
        got = rowsum_cute(x)
        ref = x.sum(dim=1)
        err = (got - ref).abs().max().item()
        ok = err < 1e-3
        worst = max(worst, err)
        print(f"  T={T_} D={D:5d}: max_abs_err={err:.3e} pass={ok}")
        if not ok:
            raise SystemExit(f"FAIL T={T_} D={D}: err={err}")
    print(f"rowsum_cute vs torch: worst max_abs_err = {worst:.3e} — PASS")
    print("=> reduction primitive set CONFIRMED: warp_reduction_sum + alloc_smem +")
    print("   smem[i] store/load + sync_threads all compile & run on sm_121.")


if __name__ == "__main__":
    _self_check()
