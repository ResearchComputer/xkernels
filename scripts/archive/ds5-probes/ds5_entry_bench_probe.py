#!/usr/bin/env python
"""Isolate the verify segfault: run mm_fp8_blockscale_cute (entry, with dequant)
directly, then via triton do_bench (the timer verify uses)."""
from __future__ import annotations
import sys, torch
from xkernels.ops.gemm.cute.entry import mm_fp8_blockscale_cute
from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8

def log(*a): print(*a); sys.stdout.flush()

torch.manual_seed(1729)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
a8, as_ = per_token_group_quant_fp8(a, block=128); b8, bs = per_block_quant_fp8(b, block=128)

log("[1] direct call (entry + dequant + cached handle):")
o = mm_fp8_blockscale_cute(a8, as_, b8, bs, block=128, out_dtype=torch.bfloat16); torch.cuda.synchronize()
log(f"    ok shape={o.shape} dtype={o.dtype}")

log("[2] do_bench timing (what verify uses):")
try:
    from triton.testing import do_bench
    ms = do_bench(lambda: mm_fp8_blockscale_cute(a8, as_, b8, bs, block=128, out_dtype=torch.bfloat16), warmup=10, rep=50)
    log(f"    do_bench: {ms:.4f} ms/call")
except Exception as e:
    log(f"    do_bench EXC: {repr(e)[:160]}")
log("DONE")
