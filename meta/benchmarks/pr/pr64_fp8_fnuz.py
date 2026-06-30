# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""PR #64 benchmark: prefer AMD fnuz fp8 for blockscale quantization.

Measures, on one gfx942 (MI300A) GPU, the fp8 block-scale dense GEMM across the
V4 MLA/gate/shared-expert projection shapes, comparing:

  * "main default"  -> operands quantized with float8_e4m3fn; path="auto" keeps
    them on the portable dequant-then-dot kernel (e4m3fn upcasts to a slower f16
    MFMA). This is what main resolves because per_*_quant_fp8 defaults to e4m3fn.
  * "PR default"    -> operands quantized with the new auto default
    (preferred_fp8_dtype -> float8_e4m3fnuz on AMD); path="auto" reaches the
    native fp8 MFMA fast path (#41).

Also reports the achieved native-MFMA TFLOP/s. The one-time "portable on AMD"
warning emitted by the e4m3fn path is captured/suppressed.

    python meta/benchmarks/pr/pr64_fp8_fnuz.py
"""
from __future__ import annotations

import warnings

import torch

from xkernels._backends import Backend
from xkernels.ops.gemm import (
    mm_fp8_blockscale,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    preferred_fp8_dtype,
)

# V4-Flash MLA / gate / shared-expert projection shapes (M = token regime).
SHAPES = [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]
BLOCK = 128


def _bench(a8, as_, w8, ws_, *, path):
    import triton

    def run():
        mm_fp8_blockscale(
            a8, as_, w8, ws_, block=BLOCK, out_dtype=torch.bfloat16,
            path=path, backend=Backend.TRITON,
        )

    return triton.testing.do_bench(run, warmup=25, rep=100)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (MI300A) GPU.")
        return
    dev = "cuda"
    pref = preferred_fp8_dtype(dev)
    print(f"preferred_fp8_dtype(cuda) = {pref}  (PR default quantization dtype)")
    assert pref == torch.float8_e4m3fnuz, "expected fnuz on AMD"

    print(
        f"{'M':>5} {'N':>5} {'K':>5} | {'e4m3fn_ms':>10} {'fnuz_ms':>9} "
        f"{'speedup':>8} {'TFLOP/s':>8}"
    )
    for (M, N, K) in SHAPES:
        a = torch.randn(M, K, device=dev)
        w = torch.randn(N, K, device=dev)

        # main default operands: explicit e4m3fn
        a_fn, as_fn = per_token_group_quant_fp8(a, block=BLOCK, fp8_dtype=torch.float8_e4m3fn)
        w_fn, ws_fn = per_block_quant_fp8(w, block=BLOCK, fp8_dtype=torch.float8_e4m3fn)
        # PR default operands: auto -> fnuz on AMD
        a_fz, as_fz = per_token_group_quant_fp8(a, block=BLOCK)  # auto
        w_fz, ws_fz = per_block_quant_fp8(w, block=BLOCK)        # auto

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the one-time portable-on-AMD notice
            t_e4m3 = _bench(a_fn, as_fn, w_fn, ws_fn, path="auto")   # -> portable
            t_fnuz = _bench(a_fz, as_fz, w_fz, ws_fz, path="auto")   # -> native MFMA

        tflops = 2 * M * N * K / t_fnuz / 1e9
        print(
            f"{M:>5} {N:>5} {K:>5} | {t_e4m3:>10.3f} {t_fnuz:>9.3f} "
            f"{t_e4m3 / t_fnuz:>7.2f}x {tflops:>8.1f}"
        )


if __name__ == "__main__":
    main()
