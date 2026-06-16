# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Bench the fp8 block-scale dense GEMM on gfx942 (issue #41): native fp8 MFMA vs
the #40 portable dequant path vs the torch dequant reference, across the V4 MLA
shapes. The native fp8 MFMA path needs ``float8_e4m3fnuz`` operands (the AMD
CDNA3 fp8 MFMA encoding); ``float8_e4m3fn`` falls back to an f16 MFMA.

    python benchmarks/bench_fp8_blockscale_gemm.py

Run on one gfx942 GPU (see ``slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch``).
"""
from __future__ import annotations

import torch

from xkernels._backends import Backend
from xkernels.ops.gemm import (
    mm_fp8_blockscale,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)
from xkernels.ops.gemm.reference import mm_fp8_blockscale_ref
from xkernels.utils.benchmarking import benchmark

# V4-Flash MLA / gate / shared-expert projection shapes (M = token regime).
SHAPES = [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]
FNUZ = torch.float8_e4m3fnuz  # native fp8 MFMA on gfx942
BLOCK = 128


def main():
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (or any CUDA/ROCm) GPU.")
        return
    dev = "cuda"
    print(
        f"{'M':>5} {'N':>5} {'K':>5} | {'mfma(ms)':>9} {'TFLOP/s':>8} "
        f"{'portable':>9} {'torch_ref':>9} | {'mfma/ref':>8} {'mfma/port':>9}"
    )
    for (M, N, K) in SHAPES:
        a = torch.randn(M, K, device=dev)
        w = torch.randn(N, K, device=dev)
        a8, as_ = per_token_group_quant_fp8(a, block=BLOCK, fp8_dtype=FNUZ)
        w8, ws_ = per_block_quant_fp8(w, block=BLOCK, fp8_dtype=FNUZ)

        def _gemm(path, *, a8=a8, as_=as_, w8=w8, ws_=ws_):
            return mm_fp8_blockscale(
                a8, as_, w8, ws_, block=BLOCK, out_dtype=torch.bfloat16,
                path=path, backend=Backend.TRITON,
            )

        def _ref(*, a8=a8, as_=as_, w8=w8, ws_=ws_):
            return mm_fp8_blockscale_ref(a8, as_, w8, ws_, block=BLOCK, out_dtype=torch.bfloat16)

        t_mfma = benchmark(lambda: _gemm("mfma"))
        t_port = benchmark(lambda: _gemm("portable"))
        t_ref = benchmark(_ref)
        tf = 2 * M * N * K / t_mfma / 1e9
        print(
            f"{M:>5} {N:>5} {K:>5} | {t_mfma:>9.3f} {tf:>8.1f} "
            f"{t_port:>9.3f} {t_ref:>9.3f} | {t_ref / t_mfma:>7.2f}x {t_port / t_mfma:>8.2f}x"
        )


if __name__ == "__main__":
    main()
