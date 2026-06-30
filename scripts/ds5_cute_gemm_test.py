#!/usr/bin/env python
"""Standalone seeded test for the CUTE DSL fp32 GEMM + the full mm_fp8_blockscale
CUTE path, isolated from the verify harness (diagnose-wrong-results §1:
reproduce in a standalone seeded script, NOT pytest).

Stage 1: fp32 GEMM correctness vs torch (isolates the CUTE kernel compile/run
         from the dequant).
Stage 2: full mm_fp8_blockscale_cute vs the torch reference (dequant + GEMM +
         cast), at every shape-sweep point, printing max_abs/rel err so the
         tolerance verdict is explicit.

Run on ds5:
  rcc --profile ds5 run -s 'cd /local/home/xiayao/xkernels && \
    export CUDA_HOME=/usr/local/cuda-13.0 && . .venv/bin/activate && \
    python scripts/ds5_cute_gemm_test.py'
"""
from __future__ import annotations

import torch

from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute
from xkernels.ops.gemm.cute.entry import mm_fp8_blockscale_cute
from xkernels.ops.gemm.reference import (
    mm_fp8_blockscale_ref,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)


def stage1_fp32_gemm() -> None:
    print("=== Stage 1: CUTE fp32 GEMM vs torch (kernel isolated from dequant) ===")
    torch.manual_seed(0)
    for M, N, K in [(16, 16, 16), (32, 32, 32), (4, 64, 64), (16, 256, 256)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(N, K, device="cuda", dtype=torch.float32)
        got = fp32_matmul_cute(a, b)
        ref = a @ b.t()
        err = (got - ref).abs().max().item()
        rel = err / ref.abs().clamp_min(1e-6).max().item()
        ok = torch.allclose(got, ref, rtol=1e-4, atol=1e-3)
        print(f"  M={M:4d} N={N:4d} K={K:4d}: max_abs={err:.3e} max_rel={rel:.3e} "
              f"allclose(1e-4,1e-3)={ok}")
        if not ok:
            print(f"    !! first got={got.flatten()[:6].tolist()} ref={ref.flatten()[:6].tolist()}")


def stage2_full_op() -> None:
    print("\n=== Stage 2: mm_fp8_blockscale_cute vs reference (full path) ===")
    torch.manual_seed(1)
    # Mirror the op's shape sweep.
    points = [
        ("bf16", 1, 256, 256), ("bf16", 8, 256, 512), ("bf16", 37, 256, 256),
        ("bf16", 128, 512, 512), ("fp32", 16, 256, 256),
    ]
    for dtype, M, N, K in points:
        out_dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        a = torch.randn(M, K, device="cuda", dtype=torch.float32)
        b = torch.randn(N, K, device="cuda", dtype=torch.float32)
        a_fp8, a_scales = per_token_group_quant_fp8(a, block=128)
        b_fp8, b_scales = per_block_quant_fp8(b, block=128)

        got = mm_fp8_blockscale_cute(a_fp8, a_scales, b_fp8, b_scales,
                                     block=128, out_dtype=out_dt)
        ref = mm_fp8_blockscale_ref(a_fp8, a_scales, b_fp8, b_scales,
                                    block=128, out_dtype=out_dt)
        rtol = 1e-2 if dtype == "bf16" else 1e-3
        atol = 1e-1 if dtype == "bf16" else 1e-3
        err = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        denom = ref.to(torch.float32).abs().clamp_min(1e-6)
        rel = (got.to(torch.float32) - ref.to(torch.float32)).abs().div(denom).max().item()
        ok = err <= atol and rel <= rtol
        print(f"  {dtype} M={M:4d} N={N:4d} K={K:4d}: max_abs={err:.3e} max_rel={rel:.3e} "
              f"tol(rtol={rtol},atol={atol}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    !! got={got.flatten()[:6].tolist()} ref={ref.flatten()[:6].tolist()}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA device"
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"(cc {''.join(map(str, torch.cuda.get_device_capability(0)))})\n")
    stage1_fp32_gemm()
    stage2_full_op()
