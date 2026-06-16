#!/usr/bin/env python3
"""Benchmark ``moe_sum_reduce`` vs the torch oracle.

Run directly on a gfx942 node:
    python benchmarks/bench_moe_sum_reduce.py
"""

from __future__ import annotations

import time

import torch

from xkernels import moe_sum_reduce
from xkernels.ops.moe.sum_reduce import moe_sum_reduce_ref


def bench(
    M: int,
    top_k: int,
    H: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    n_warmup: int = 10,
    n_iter: int = 50,
):
    y = torch.randn(M, top_k, H, dtype=dtype, device=device)
    w = torch.rand(M, top_k, device=device, dtype=torch.float32)
    scaling = 1.0 / top_k

    # Reference
    ref = moe_sum_reduce_ref(y, w, scaling)

    # Triton result + warmup
    for _ in range(n_warmup):
        out = moe_sum_reduce(y, w, routed_scaling_factor=scaling)
    torch.cuda.synchronize() if device == "cuda" else None

    # Time Triton
    start = time.perf_counter()
    for _ in range(n_iter):
        out = moe_sum_reduce(y, w, routed_scaling_factor=scaling)
    if device == "cuda":
        torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / n_iter * 1e3

    # Time torch reference
    start = time.perf_counter()
    for _ in range(n_iter):
        _ = moe_sum_reduce_ref(y, w, scaling)
    if device == "cuda":
        torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) / n_iter * 1e3

    max_err = (out - ref).abs().max().item()
    mean_err = (out - ref).abs().mean().item()

    return {
        "M": M,
        "top_k": top_k,
        "H": H,
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / triton_ms,
        "max_err": max_err,
        "mean_err": mean_err,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires a GPU.")
        return

    print("Benchmarking moe_sum_reduce (Triton vs torch reference)")
    print("-" * 90)
    print(
        f"{'M':>6} {'top_k':>6} {'H':>6} "
        f"{'triton_ms':>12} {'torch_ms':>12} {'speedup':>8} "
        f"{'max_err':>10} {'mean_err':>10}"
    )
    print("-" * 90)

    shapes = [
        (1, 8, 4096),
        (1, 8, 11008),
        (1, 8, 16384),
        (16, 8, 4096),
        (16, 8, 11008),
        (64, 8, 4096),
        (64, 8, 11008),
        (256, 8, 4096),
        (256, 8, 11008),
        (1024, 8, 4096),
        (4096, 8, 4096),
    ]

    for M, top_k, H in shapes:
        try:
            res = bench(M, top_k, H)
            print(
                f"{res['M']:>6} {res['top_k']:>6} {res['H']:>6} "
                f"{res['triton_ms']:>12.4f} {res['torch_ms']:>12.4f} "
                f"{res['speedup']:>8.2f}x {res['max_err']:>10.2e} {res['mean_err']:>10.2e}"
            )
        except Exception as e:
            print(f"{M:>6} {top_k:>6} {H:>6} FAILED: {e}")


if __name__ == "__main__":
    main()
