# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Tiny single-kernel workload for profiling on NVIDIA under Nsight Compute /
Nsight Systems.

`ncu --kernel-name regex:<frag> -- python3 meta/benchmarks/probe_ncu.py <kernel>` (or
the nsys equivalent) runs ONE kernel a fixed number of times after a warm-up,
producing a clean, repeatable dispatch to profile. This is the NVIDIA twin of
`probe_omniperf.py` (the AMD/ROCm-Compute-Profiler workload) — same kernel set,
same deterministic (seeded) shapes, so a profile captured on bristen is directly
comparable to one captured on beverin. Keep it dependency-light.

Supported kernels (mirror meta/benchmarks/bench_all.py shapes):
  dual_rmsnorm, moe_sum_reduce, fused_ffn, mha_merge_state,
  hc_prenorm_gemm, mhc_pre, sparse_mla_attention,
  moe_align_block_size, moe_int4_w4a16, mm_fp8_blockscale.

Note: mm_fp8_blockscale uses the gfx942-only e4m3fnuz fp8 MFMA path; it is not
profiled on bristen (sm_80 has no hardware fp8). Each builder returns a thunk +
the dominant Triton kernel name fragment so `ncu -k regex:<frag>` can isolate the
dispatch from the autotune/launch helpers.
"""
from __future__ import annotations

import sys

import torch

from xkernels._backends import Backend

DT = torch.bfloat16
WARMUP = 5
ITERS = 10


def _dual_rmsnorm(dev: str):
    from xkernels import dual_rmsnorm

    T, D1, D2 = 8192, 1536, 512
    x1 = torch.randn(T, D1, device=dev, dtype=DT)
    x2 = torch.randn(T, D2, device=dev, dtype=DT)
    w1 = torch.randn(D1, device=dev, dtype=DT)
    w2 = torch.randn(D2, device=dev, dtype=DT)
    return lambda: dual_rmsnorm(x1, w1, x2, w2), "rmsnorm"


def _moe_sum_reduce(dev: str):
    from xkernels import moe_sum_reduce

    M, TOP_K, H = 8192, 8, 7168
    y = torch.randn(M, TOP_K, H, device=dev, dtype=DT)
    w = torch.rand(M, TOP_K, device=dev, dtype=torch.float32)
    return lambda: moe_sum_reduce(y, w), "sum_reduce"


def _fused_ffn(dev: str):
    from xkernels import fused_ffn

    M, d_model, d_ff = 4096, 4096, 11008
    ft = torch.float16
    x = torch.randn(M, d_model, device=dev, dtype=ft)
    wg = torch.randn(d_model, d_ff, device=dev, dtype=ft)
    wu = torch.randn(d_model, d_ff, device=dev, dtype=ft)
    wd = torch.randn(d_ff, d_model, device=dev, dtype=ft)
    return lambda: fused_ffn(x, wg, wu, wd, backend=Backend.TRITON), "swiglu"


def _mha_merge_state(dev: str):
    from xkernels import mha_merge_state

    T, H, D = 8192, 128, 128
    oa = torch.randn(T, H, D, device=dev, dtype=DT)
    ob = torch.randn(T, H, D, device=dev, dtype=DT)
    la = torch.randn(T, H, device=dev)
    lb = torch.randn(T, H, device=dev)
    return lambda: mha_merge_state(oa, la, ob, lb), "merge_state"


def _hc_prenorm_gemm(dev: str):
    from xkernels import hc_prenorm_gemm

    T, hc_mult, hidden = 8, 4, 4096
    K, N, n_splits = hc_mult * hidden, 24, 16
    a = torch.randn(T, K, device=dev, dtype=DT)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    return (
        lambda: hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON),
        "prenorm_gemm",
    )


def _mhc_pre(dev: str):
    from xkernels import mhc_post, mhc_pre

    T, hc_mult, hidden = 8, 4, 4096
    K = hc_mult * hidden
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    residual = torch.randn(T, hc_mult, hidden, device=dev, dtype=DT)
    fn = torch.randn(hc_mult3, K, device=dev, dtype=torch.float32)
    hc_scale = torch.tensor([1.0, 1.0, 1.0], device=dev, dtype=torch.float32)
    hc_base = torch.zeros(hc_mult3, device=dev, dtype=torch.float32)

    def _run():
        li, post, comb = mhc_pre(
            residual, fn, hc_scale, hc_base, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=3
        )
        return mhc_post(li, residual, post, comb)

    return _run, "prenorm_gemm"


def _sparse_mla(dev: str):
    from xkernels import sparse_mla_attention

    T, H, D, D_V, Kv, topk = 8, 128, 512, 448, 8192, 512
    sm_scale = 1.0 / (D**0.5)
    q = torch.randn(T, H, D, device=dev, dtype=DT)
    kv = torch.randn(Kv, D, device=dev, dtype=DT)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)
    return (
        lambda: sparse_mla_attention(
            q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V, backend=Backend.TRITON
        ),
        "sparse_mla",
    )


def _moe_align(dev: str):
    from xkernels import moe_align_block_size

    M, top_k, E, block = 16384, 8, 48, 16
    g = torch.Generator(device=dev).manual_seed(0)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)
    return (
        lambda: moe_align_block_size(topk_ids, block, E, backend="triton"),
        "align",
    )


def _moe_int4(dev: str):
    import triton.language as tl

    from xkernels.ops.moe import make_w4a16_weights, moe_align_block_size_ref
    from xkernels.ops.moe.triton.configs import align_block_m, get_moe_int4_config
    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    M, E, N, K, top_k, gs = 64, 48, 4096, 7168, 8, 32
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=1)
    A = (torch.randn(M, K, device=dev) * 0.1).to(DT)
    topk_ids = (
        torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)])
        .to(torch.int32)
    )
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    config = get_moe_int4_config(E, N, K, M)
    block_m = config["BLOCK_SIZE_M"] if config is not None else align_block_m(M)
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    topk_w_flat = topk_w.reshape(-1).float()
    c = torch.zeros((M * top_k, N), dtype=DT, device=dev)

    def _run():
        int4_w4a16_moe_gemm(
            A, packed, scale, c, topk_w_flat, sorted_ids, expert_ids, num_post,
            top_k=top_k, group_size=gs, mul_routed_weight=True,
            compute_type=tl.bfloat16, filter_expert=False, config=config,
        )
        return c.view(M, top_k, N).sum(dim=1)

    return _run, "fused_moe_int4"


def _mm_fp8_blockscale(dev: str):
    # gfx942-only: native fp8 MFMA needs float8_e4m3fnuz operands.
    from xkernels.ops.gemm import (
        mm_fp8_blockscale,
        per_block_quant_fp8,
        per_token_group_quant_fp8,
    )

    M, N, K = 2048, 512, 7168
    block = 128
    FNUZ = torch.float8_e4m3fnuz
    a = torch.randn(M, K, device=dev)
    w = torch.randn(N, K, device=dev)
    a8, as_ = per_token_group_quant_fp8(a, block=block, fp8_dtype=FNUZ)
    w8, ws_ = per_block_quant_fp8(w, block=block, fp8_dtype=FNUZ)
    return (
        lambda: mm_fp8_blockscale(
            a8, as_, w8, ws_, block=block, out_dtype=torch.bfloat16,
            path="mfma", backend=Backend.TRITON,
        ),
        "blockscale_mfma",
    )


KERNELS = {
    "dual_rmsnorm": _dual_rmsnorm,
    "moe_sum_reduce": _moe_sum_reduce,
    "fused_ffn": _fused_ffn,
    "mha_merge_state": _mha_merge_state,
    "hc_prenorm_gemm": _hc_prenorm_gemm,
    "mhc_pre": _mhc_pre,
    "sparse_mla_attention": _sparse_mla,
    "moe_align_block_size": _moe_align,
    "moe_int4_w4a16": _moe_int4,
    "mm_fp8_blockscale": _mm_fp8_blockscale,
}


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU — run this on a bristen A100 compute node.")
    name = sys.argv[1] if len(sys.argv) > 1 else "dual_rmsnorm"
    if name not in KERNELS:
        raise SystemExit(f"unknown kernel {name!r}; choose from {sorted(KERNELS)}")
    dev = "cuda"
    fn, namefrag = KERNELS[name](dev)
    for _ in range(WARMUP):  # compile + fill caches so the profiled run is steady-state
        fn()
    torch.cuda.synchronize()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    print(f"[probe_ncu] ran {name} {ITERS}x; kernel-name fragment for ncu -k: {namefrag}")


if __name__ == "__main__":
    main()
