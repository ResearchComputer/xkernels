#!/usr/bin/env python
"""Isolate the verify segfault: does compiling a 2nd cute.compile handle for a
DIFFERENT shape corrupt the first (shared CuTeDSL singleton state)?"""
from __future__ import annotations
import sys, torch
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute

def log(*a): print(*a); sys.stdout.flush()

SHAPES = [(128, 512, 512), (16, 256, 256), (64, 128, 128), (32, 64, 256)]
for i, (M, N, K) in enumerate(SHAPES):
    log(f"[{i}] shape M={M} N={N} K={K}")
    a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
    try:
        o = fp32_matmul_cute(a, b); torch.cuda.synchronize()
        ref = a @ b.t()
        log(f"    ok, max_abs={ (o-ref).abs().max().item():.3e}")
    except Exception as e:
        log(f"    EXC: {repr(e)[:140]}")
log("now RE-RUN shape[0] (is its cached handle still valid after shape 1,2,3 compiled?):")
M, N, K = SHAPES[0]
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
try:
    o = fp32_matmul_cute(a, b); torch.cuda.synchronize()
    log(f"    re-run ok, max_abs={ (o-(a@b.t())).abs().max().item():.3e}")
except Exception as e:
    log(f"    EXC: {repr(e)[:140]}")
log("DONE")
