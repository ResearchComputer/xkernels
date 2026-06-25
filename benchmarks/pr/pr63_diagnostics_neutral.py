# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""PR #63 benchmark: backend-failure-diagnostics is an observability change, not
a kernel hot-path change, so it should be performance-neutral. This script times
two representative optimized kernels that go through dispatch() (the only code
path PR #63 touches) and prints the timings; compare against the same script run
on main to confirm no regression. It also prints backend_diagnostics() to
demonstrate the new observability surface.

    python benchmarks/pr/pr63_diagnostics_neutral.py
"""
from __future__ import annotations

import torch

from xkernels import sparse_mla_attention
from xkernels._backends import Backend

try:  # only present on PR #63+
    from xkernels import backend_diagnostics
except ImportError:
    backend_diagnostics = None
from xkernels.ops.gemm import (
    mm_fp8_blockscale,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)
from xkernels.ops.moe import make_w4a16_weights
from xkernels.ops.moe.interface import fused_moe_int4_w4a16

BLOCK = 128


def _time(fn):
    import triton

    return triton.testing.do_bench(fn, warmup=25, rep=100)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (MI300A) GPU.")
        return
    dev = "cuda"

    # (1) sparse-MLA multi-token (the documented bench workload), dispatch path.
    T, H, D, D_V, Kv, topk = 8, 128, 512, 448, 8192, 512
    sm_scale = 1.0 / (D**0.5)
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)

    def run_mla():
        sparse_mla_attention(
            q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
            backend=Backend.TRITON,
        )

    t_mla = _time(run_mla)

    # (2) INT4 W4A16 fused-MoE GEMM decode bucket, dispatch path.
    E, N, K, top_k, gs = 48, 4096, 7168, 8, 32
    M = 16
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=1)
    a = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M * top_k, device=dev, dtype=torch.float32)

    def run_moe():
        fused_moe_int4_w4a16(
            a, packed, scale, topk_ids, topk_w, group_size=gs,
            mul_routed_weight=True, fused_combine=True, backend=Backend.TRITON,
        )

    t_moe = _time(run_moe)

    # (3) fp8 blockscale native MFMA, dispatch path.
    Mf, Nf, Kf = 4096, 7168, 2048
    af = torch.randn(Mf, Kf, device=dev)
    wf = torch.randn(Nf, Kf, device=dev)
    a8, as_ = per_token_group_quant_fp8(af, block=BLOCK)
    w8, ws_ = per_block_quant_fp8(wf, block=BLOCK)

    def run_fp8():
        mm_fp8_blockscale(a8, as_, w8, ws_, block=BLOCK, out_dtype=torch.bfloat16,
                          path="auto", backend=Backend.TRITON)

    t_fp8 = _time(run_fp8)

    print("=== representative kernel timings through dispatch() (ms) ===")
    print(f"sparse_mla (T=8, topk=512):        {t_mla:.4f}")
    print(f"moe_int4_w4a16 (M=16, gate_up):    {t_moe:.4f}")
    print(f"mm_fp8_blockscale (4096x7168x2048):{t_fp8:.4f}")

    if backend_diagnostics is None:
        print("\n(backend_diagnostics() not available on this revision)")
        return
    print("\n=== backend_diagnostics() (PR #63 surface) ===")
    diag = backend_diagnostics()
    sample = ["sparse_mla_attention", "moe_int4_w4a16", "mm_fp8_blockscale", "residual_rmsnorm"]
    for name in sample:
        if name in diag:
            print(f"  {name}: registered={diag[name]['registered']}")


if __name__ == "__main__":
    main()
