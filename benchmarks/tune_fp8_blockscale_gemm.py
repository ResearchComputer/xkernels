# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Sweep the CDNA3 autotune config space for the native fp8 MFMA fp8 block-scale
GEMM (issue #41) across the V4 MLA shapes and report the best config per shape.

The winners are baked into ``configs.get_fp8_gemm_config``. Run on one gfx942 GPU:

    python benchmarks/tune_fp8_blockscale_gemm.py
"""
from __future__ import annotations

import torch
from triton.testing import do_bench

from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.ops.gemm.triton.configs import get_autotune_configs
from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton

SHAPES = [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]
FNUZ = torch.float8_e4m3fnuz


def _cfg_to_dict(c):
    d = dict(c.kwargs)
    d["num_warps"] = c.num_warps
    d["num_stages"] = c.num_stages
    return d


def main():
    dev = "cuda"
    block = 128
    cfgs = get_autotune_configs()
    for (M, N, K) in SHAPES:
        a = torch.randn(M, K, device=dev)
        w = torch.randn(N, K, device=dev)
        a8, as_ = per_token_group_quant_fp8(a, block=block, fp8_dtype=FNUZ)
        w8, ws_ = per_block_quant_fp8(w, block=block, fp8_dtype=FNUZ)
        best = None
        for c in cfgs:
            d = _cfg_to_dict(c)
            try:
                t = do_bench(
                    lambda d=d: mm_fp8_blockscale_mfma_triton(
                        a8, as_, w8, ws_, block=block, out_dtype=torch.bfloat16, config=d
                    )
                )
            except Exception as e:  # noqa: BLE001 - skip configs that OOM-LDS / fail
                print(f"    skip {d['BLOCK_M']}x{d['BLOCK_N']}x{d['BLOCK_K']} "
                      f"w{d['num_warps']}s{d['num_stages']}nk{d['matrix_instr_nonkdim']}: {repr(e)[:60]}")
                continue
            tf = 2 * M * N * K / t / 1e9
            tag = (f"BM{d['BLOCK_M']} BN{d['BLOCK_N']} BK{d['BLOCK_K']} G{d['GROUP_M']} "
                   f"w{d['num_warps']} s{d['num_stages']} nk{d['matrix_instr_nonkdim']} we{d['waves_per_eu']}")
            print(f"[M={M} N={N} K={K}] {t:.4f}ms {tf:6.1f}TF  {tag}")
            if best is None or t < best[0]:
                best = (t, tag, d)
        print(f"==> BEST M={M} N={N} K={K}: {best[0]:.4f}ms  {best[1]}\n")


if __name__ == "__main__":
    main()
