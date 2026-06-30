#!/usr/bin/env python3
"""Benchmark ``mhc_pre`` / ``mhc_post`` vs the torch oracle.

Run directly on a gfx942 node:
    python meta/benchmarks/bench_mhc_pre_post.py
"""

from __future__ import annotations

import time

import torch

from xkernels import mhc_post, mhc_pre
from xkernels.ops.mhc.pre_post_reference import mhc_post_ref, mhc_pre_ref


def bench_pre(
    T: int,
    hc_mult: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    n_warmup: int = 10,
    n_iter: int = 50,
):
    residual = torch.randn(T, hc_mult, hidden, dtype=dtype, device=device)
    K = hc_mult * hidden
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    fn = torch.randn(hc_mult3, K, dtype=torch.float32, device=device)
    hc_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.zeros(hc_mult3, dtype=torch.float32, device=device)

    ref = mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=3)

    for _ in range(n_warmup):
        out = mhc_pre(residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=3)
    torch.cuda.synchronize() if device == "cuda" else None

    start = time.perf_counter()
    for _ in range(n_iter):
        out = mhc_pre(residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=3)
    if device == "cuda":
        torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / n_iter * 1e3

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = mhc_pre_ref(
            residual, fn, hc_scale, hc_base,
            rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=3,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) / n_iter * 1e3

    li_err = (out[0] - ref[0]).abs().max().item()
    post_err = (out[1] - ref[1]).abs().max().item()
    comb_err = (out[2] - ref[2]).abs().max().item()

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / triton_ms,
        "li_err": li_err,
        "post_err": post_err,
        "comb_err": comb_err,
    }


def bench_post(
    T: int,
    hc_mult: int,
    hidden: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    n_warmup: int = 10,
    n_iter: int = 50,
):
    residual = torch.randn(T, hc_mult, hidden, dtype=dtype, device=device)
    hidden_states = torch.randn(T, hidden, dtype=dtype, device=device)
    post = torch.randn(T, hc_mult, 1, dtype=torch.float32, device=device)
    comb = torch.randn(T, hc_mult, hc_mult, dtype=torch.float32, device=device)

    ref = mhc_post_ref(hidden_states, residual, post, comb)

    for _ in range(n_warmup):
        out = mhc_post(hidden_states, residual, post, comb)
    torch.cuda.synchronize() if device == "cuda" else None

    start = time.perf_counter()
    for _ in range(n_iter):
        out = mhc_post(hidden_states, residual, post, comb)
    if device == "cuda":
        torch.cuda.synchronize()
    triton_ms = (time.perf_counter() - start) / n_iter * 1e3

    start = time.perf_counter()
    for _ in range(n_iter):
        _ = mhc_post_ref(hidden_states, residual, post, comb)
    if device == "cuda":
        torch.cuda.synchronize()
    torch_ms = (time.perf_counter() - start) / n_iter * 1e3

    err = (out - ref).abs().max().item()

    return {
        "triton_ms": triton_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / triton_ms,
        "err": err,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; benchmark requires a GPU.")
        return

    shapes = [
        (1, 4, 1024),
        (1, 4, 4096),
        (16, 4, 4096),
        (64, 4, 4096),
        (256, 4, 4096),
        (1024, 4, 4096),
        (4096, 4, 4096),
    ]

    print("Benchmarking mhc_pre (Triton vs torch reference)")
    print("-" * 100)
    header = (
        f"{'T':>6} {'hc':>4} {'hidden':>6} {'triton_ms':>12} {'torch_ms':>12} "
        f"{'speedup':>8} {'li_err':>10} {'post_err':>10} {'comb_err':>10}"
    )
    print(header)
    print("-" * 100)
    for T, hc_mult, hidden in shapes:
        try:
            r = bench_pre(T, hc_mult, hidden)
            print(
                f"{T:>6} {hc_mult:>4} {hidden:>6} "
                f"{r['triton_ms']:>12.4f} {r['torch_ms']:>12.4f} "
                f"{r['speedup']:>8.2f}x {r['li_err']:>10.2e} "
                f"{r['post_err']:>10.2e} {r['comb_err']:>10.2e}"
            )
        except Exception as e:
            print(f"{T:>6} {hc_mult:>4} {hidden:>6} FAILED: {e}")

    print()
    print("Benchmarking mhc_post (Triton vs torch reference)")
    print("-" * 90)
    print(
        f"{'T':>6} {'hc':>4} {'hidden':>6} "
        f"{'triton_ms':>12} {'torch_ms':>12} {'speedup':>8} {'max_err':>10}"
    )
    print("-" * 90)
    for T, hc_mult, hidden in shapes:
        try:
            r = bench_post(T, hc_mult, hidden)
            print(
                f"{T:>6} {hc_mult:>4} {hidden:>6} "
                f"{r['triton_ms']:>12.4f} {r['torch_ms']:>12.4f} "
                f"{r['speedup']:>8.2f}x {r['err']:>10.2e}"
            )
        except Exception as e:
            print(f"{T:>6} {hc_mult:>4} {hidden:>6} FAILED: {e}")


if __name__ == "__main__":
    main()
