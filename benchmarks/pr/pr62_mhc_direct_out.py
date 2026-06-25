# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""PR #62 benchmark: write MHC prenorm GEMM into caller buffers.

Measures, on one gfx942 (MI300A) GPU, the in-place faithful wrapper
``tf32_hc_prenorm_gemm`` (the tokenspeed binding target), comparing:

  * "main default"  -> old behavior: hc_prenorm_gemm allocates the
    [n_splits,T,N] / [n_splits,T] fp32 buffers internally and the wrapper
    .copy_()s them into the caller buffers.
  * "PR default"    -> new behavior: the wrapper dispatches hc_prenorm_gemm_out,
    which writes directly into the caller buffers (no allocation, no copy-back).

Both paths write the same pre-allocated caller buffers, so the timing captures
exactly what a caller of tf32_hc_prenorm_gemm pays in each revision.

    python benchmarks/pr/pr62_mhc_direct_out.py
"""
from __future__ import annotations

import torch

from xkernels._backends import Backend
from xkernels._dispatch import dispatch

# V4-Flash MHC: hc_mult=4, hidden=4096 -> K=16384, N=24 (hc_mult3 = 2*4 + 4**2 = 24).
HC_MULT, HIDDEN, N_SPLITS = 4, 4096, 16
K, N = HC_MULT * HIDDEN, 24


def _bench(T, *, via_out, dev="cuda"):
    import triton

    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    mul = torch.empty(N_SPLITS, T, N, device=dev, dtype=torch.float32)
    sqr = torch.empty(N_SPLITS, T, device=dev, dtype=torch.float32)

    def run_old():
        # Reproduce the main-revision tf32_hc_prenorm_gemm: allocate + copy back.
        m, s = dispatch("hc_prenorm_gemm", a, fn, n_splits=N_SPLITS, backend=Backend.TRITON)
        mul.copy_(m)
        sqr.copy_(s)

    def run_new():
        # PR revision: dispatch the out-buffer backend directly into caller buffers.
        dispatch(
            "hc_prenorm_gemm_out", a, fn, mul, sqr,
            n_splits=N_SPLITS, backend=Backend.TRITON,
        )

    run = run_new if via_out else run_old
    return triton.testing.do_bench(run, warmup=25, rep=200)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (MI300A) GPU.")
        return

    print(f"geometry: K={K}, N={N}, n_splits={N_SPLITS} (V4-Flash MHC prenorm GEMM)")
    print(f"{'T':>5} | {'alloc+copy_ms':>14} {'direct_out_ms':>14} | {'speedup':>8}")
    for T in (1, 8, 64):
        t_old = _bench(T, via_out=False)
        t_new = _bench(T, via_out=True)
        print(f"{T:>5} | {t_old:>14.4f} {t_new:>14.4f} | {t_old / t_new:>7.3f}x")


if __name__ == "__main__":
    main()
