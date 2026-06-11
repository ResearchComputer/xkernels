# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Consolidated kernel benchmark: optimized backend vs naive PyTorch, one table.

Times each single-GPU xkernels op against the naive PyTorch implementation a
practitioner would write without the kernel, at a representative serving-regime
shape, and prints a markdown speedup table for the README.

Naive baselines (what "speedup vs naive PyTorch" measures here):
  * mha_merge_state  -> torch online-softmax merge (`merge_state_ref`)
  * dual_rmsnorm     -> two sequential `rmsnorm` launches
  * moe_sum_reduce   -> torch weighted top-k reduce (`moe_sum_reduce_ref`)
  * fused_ffn        -> the `reference` backend (unfused torch matmuls + SiLU)
  * moe_int4_w4a16   -> per-expert dequant(int4->bf16) + grouped matmul
  * moe_align_block_size -> torch argsort + per-expert padding loop (the reference)

The distributed `hierarchical_all_reduce` (see docs/issue-12-*) is not a
single-GPU speedup-vs-torch story and is reported separately in the README.
`moe_align_block_size` is additionally swept across token counts standalone in
`bench_moe_align_block_size.py`.

Run on one gfx942 GPU (see slurm/bench_all_beverin.sbatch). Timing uses
`xkernels.utils.benchmarking.benchmark` (Triton `do_bench` when available).

Usage::

    python benchmarks/bench_all.py
"""
from __future__ import annotations

import torch

from xkernels import (
    dual_rmsnorm,
    fused_ffn,
    mha_merge_state,
    moe_align_block_size,
    moe_sum_reduce,
)
from xkernels.ops.attention.reference import merge_state_ref
from xkernels.ops.moe import dequant_w4a16, make_w4a16_weights
from xkernels.ops.moe.reference import moe_w4a16_ref
from xkernels.ops.moe.sum_reduce import moe_sum_reduce_ref
from xkernels.ops.norm.reference import rmsnorm
from xkernels.utils.benchmarking import benchmark

DT = torch.bfloat16
# Collected rows: (kernel, shape, naive_label, naive_ms, opt_ms).
RESULTS: list[tuple[str, str, str, float, float]] = []


def _record(kernel, shape, naive_label, naive_fn, opt_fn):
    naive_ms = benchmark(naive_fn)
    opt_ms = benchmark(opt_fn)
    RESULTS.append((kernel, shape, naive_label, naive_ms, opt_ms))


def bench_merge_state(dev):
    T, H, D = 8192, 128, 128
    oa = torch.randn(T, H, D, device=dev, dtype=DT)
    ob = torch.randn(T, H, D, device=dev, dtype=DT)
    la = torch.randn(T, H, device=dev)
    lb = torch.randn(T, H, device=dev)
    _record(
        "mha_merge_state", f"T={T}, H={H}, D={D}", "torch merge",
        lambda: merge_state_ref(oa, la, ob, lb),
        lambda: mha_merge_state(oa, la, ob, lb),
    )


def bench_sparse_mla(dev):
    from xkernels import sparse_mla_attention
    from xkernels._backends import Backend
    from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref

    T, H, D, D_V, Kv, topk = 8, 128, 512, 448, 8192, 512
    sm_scale = 1.0 / (D**0.5)
    q = torch.randn(T, H, D, device=dev, dtype=DT)
    kv = torch.randn(Kv, D, device=dev, dtype=DT)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)
    _record(
        "sparse_mla", f"T={T}, H={H}, D={D}, topk={topk}", "torch gather+softmax",
        lambda: sparse_mla_attention_ref(
            q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V
        ),
        lambda: sparse_mla_attention(
            q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
            backend=Backend.TRITON,
        ),
    )


def bench_dual_rmsnorm(dev):
    T, D1, D2 = 8192, 1536, 512
    x1 = torch.randn(T, D1, device=dev, dtype=DT)
    x2 = torch.randn(T, D2, device=dev, dtype=DT)
    w1 = torch.randn(D1, device=dev, dtype=DT)
    w2 = torch.randn(D2, device=dev, dtype=DT)
    _record(
        "dual_rmsnorm", f"T={T}, d=({D1},{D2})", "2x sequential rmsnorm",
        lambda: (rmsnorm(x1, w1), rmsnorm(x2, w2)),
        lambda: dual_rmsnorm(x1, w1, x2, w2),
    )


def bench_moe_sum_reduce(dev):
    M, TOP_K, H = 8192, 8, 7168
    y = torch.randn(M, TOP_K, H, device=dev, dtype=DT)
    w = torch.rand(M, TOP_K, device=dev, dtype=torch.float32)
    _record(
        "moe_sum_reduce", f"M={M}, top_k={TOP_K}, H={H}", "torch reduce",
        lambda: moe_sum_reduce_ref(y, w),
        lambda: moe_sum_reduce(y, w),
    )


def bench_moe_align(dev):
    M, top_k, E, block = 16384, 8, 48, 16  # Kimi-K2.6 top_k/experts; decode block
    g = torch.Generator(device=dev).manual_seed(0)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)
    _record(
        "moe_align_block_size", f"M={M}, top_k={top_k}, E={E}, block={block}", "torch argsort+pad",
        lambda: moe_align_block_size(topk_ids, block, E, backend="reference"),
        lambda: moe_align_block_size(topk_ids, block, E, backend="triton"),
    )


def bench_ffn(dev):
    # fp16, not bf16: on this torch 2.11+rocm7.2 build the bf16 GEMM misses the
    # MFMA/hipBLASLt path and runs ~470x slower than fp16 (0.8 vs 358 TFLOP/s at
    # this shape, see benchmarks/probe_ffn.py). FFN is the only GEMM-bound op, so
    # fp16 gives the representative number; the bf16 pathology would swamp it.
    M, d_model, d_ff = 4096, 4096, 11008
    ft = torch.float16
    x = torch.randn(M, d_model, device=dev, dtype=ft)
    wg = torch.randn(d_model, d_ff, device=dev, dtype=ft)
    wu = torch.randn(d_model, d_ff, device=dev, dtype=ft)
    wd = torch.randn(d_ff, d_model, device=dev, dtype=ft)
    _record(
        "fused_ffn", f"M={M}, d_model={d_model}, d_ff={d_ff} (fp16)", "unfused torch",
        lambda: fused_ffn(x, wg, wu, wd, backend="reference"),
        lambda: fused_ffn(x, wg, wu, wd, backend="triton"),
    )


def _naive_moe_int4(A, packed, scale, topk_ids, topk_w, group_size):
    """Naive W4A16 grouped MoE: per active expert, dequant(int4->bf16) + matmul.

    Mirrors `moe_w4a16_ref` semantics (mul_routed_weight=True) but vectorized
    within each expert instead of a per-token python loop. Dequant is inside the
    timed region — a naive forward reads int4 and dequants every call; the fused
    kernel's win is reading int4 directly into the GEMM.
    """
    M, top_k = topk_ids.shape
    N = packed.shape[1]
    out = torch.zeros(M, N, device=A.device, dtype=torch.float32)
    Af = A.float()
    for e in torch.unique(topk_ids.reshape(-1)).tolist():
        w_e = dequant_w4a16(packed[e : e + 1], scale[e : e + 1], group_size)[0]  # [N,K]
        m_idx, j_idx = (topk_ids == e).nonzero(as_tuple=True)
        contrib = (Af[m_idx] @ w_e.float().T) * topk_w[m_idx, j_idx].unsqueeze(1).float()
        out.index_add_(0, m_idx, contrib)
    return out.to(A.dtype)


def bench_moe_int4(dev):
    import triton.language as tl

    from xkernels.ops.moe import moe_align_block_size_ref
    from xkernels.ops.moe.triton.configs import align_block_m, get_moe_int4_config
    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    M, E, N, K, top_k, gs = 64, 48, 4096, 7168, 8, 32  # Kimi-K2.6 gate_up, decode
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=1)
    A = (torch.randn(M, K, device=dev) * 0.1).to(DT)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)

    # Correctness guard at tiny M: our vectorized naive == the python-loop oracle.
    sm = 4
    ref = moe_w4a16_ref(A[:sm], packed, scale, topk_ids[:sm], topk_w[:sm], gs, True)
    got = _naive_moe_int4(A[:sm], packed, scale, topk_ids[:sm], topk_w[:sm], gs)
    assert torch.allclose(ref.float(), got.float(), atol=2e-2, rtol=2e-2), "naive mismatch"

    # Optimized: the tuned (issue #16) INT4 grouped GEMM + top-k reduce. The
    # block-align dispatch is the *separate* moe_align_block_size kernel (its own
    # row, 32.9x); build it once here so this row isolates the GEMM, matching the
    # issue-#16 tuner's do_bench methodology rather than timing a python-loop align
    # in the hot path. Resolves the checked-in tuned config for this shape/M.
    config = get_moe_int4_config(E, N, K, M)
    block_m = config["BLOCK_SIZE_M"] if config is not None else align_block_m(M)
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    topk_w_flat = topk_w.reshape(-1).float()
    c = torch.zeros((M * top_k, N), dtype=DT, device=dev)

    def _opt():
        int4_w4a16_moe_gemm(
            A, packed, scale, c, topk_w_flat, sorted_ids, expert_ids, num_post,
            top_k=top_k, group_size=gs, mul_routed_weight=True,
            compute_type=tl.bfloat16, filter_expert=False, config=config,
        )
        return c.view(M, top_k, N).sum(dim=1)

    _record(
        "moe_int4_w4a16", f"M={M}, E={E}, N={N}, K={K}, top_k={top_k}", "dequant+matmul",
        lambda: _naive_moe_int4(A, packed, scale, topk_ids, topk_w, gs),
        _opt,
    )


def main():
    if not torch.cuda.is_available():
        print("No GPU available; this benchmark needs a gfx942 (or any CUDA/ROCm) GPU.")
        return
    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"device: {name}  |  dtype: {DT}\n")

    for fn in (
        bench_merge_state,
        bench_sparse_mla,
        bench_dual_rmsnorm,
        bench_moe_sum_reduce,
        bench_moe_align,
        bench_ffn,
        bench_moe_int4,
    ):
        try:
            fn(dev)
        except Exception as exc:  # noqa: BLE001
            RESULTS.append((fn.__name__.replace("bench_", ""), "—", str(exc)[:48], 0.0, 0.0))

    print("| Kernel | Shape | naive PyTorch | optimized | speedup |")
    print("|--------|-------|--------------:|----------:|--------:|")
    for kernel, shape, label, naive_ms, opt_ms in RESULTS:
        if opt_ms <= 0:
            print(f"| `{kernel}` | {shape} | n/a ({label}) | n/a | — |")
            continue
        print(
            f"| `{kernel}` | {shape} | {naive_ms:.3f} ms ({label}) "
            f"| {opt_ms:.3f} ms | **{naive_ms / opt_ms:.2f}×** |"
        )


if __name__ == "__main__":
    main()
