# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Benchmark the MHC prenorm GEMM (#36) vs the naive torch baseline a
practitioner would write: F.linear(A, fn) + a separate per-row sum-of-squares.

Run on one gfx942 GPU (see scripts/archive/issues/test_mhc_prenorm_beverin.sbatch)::

    python meta/benchmarks/bench_mhc_prenorm_gemm.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from xkernels import hc_prenorm_gemm
from xkernels._backends import Backend


def main():
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (or any CUDA/ROCm) GPU.")
        return
    dev = "cuda"
    # V4-Flash MHC: hc_mult=4, hidden=4096 -> K=16384, N=24. Decode T small.
    for T in (1, 8, 64):
        hc_mult, hidden = 4, 4096
        K, N = hc_mult * hidden, 24
        n_splits = 16
        a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        fn = torch.randn(N, K, device=dev, dtype=torch.float32)

        def naive(a=a, fn=fn):
            af = a.float()
            return F.linear(af, fn.float()), (af * af).sum(-1)

        def opt(a=a, fn=fn, n_splits=n_splits):
            return hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)

        t_naive = triton.testing.do_bench(naive)
        t_opt = triton.testing.do_bench(opt)
        print(
            f"| mhc_prenorm_gemm | T={T}, K={K}, N={N}, splits={n_splits} | "
            f"{t_naive:.3f} ms (F.linear+sqsum) | {t_opt:.3f} ms | "
            f"{t_naive / t_opt:.2f}x |"
        )


if __name__ == "__main__":
    main()
