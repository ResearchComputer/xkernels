# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark for the INT4 W4A16 fused-MoE GEMM over Kimi-K2.6 shapes.

Shapes use the Kimi-K2.6 per-rank EP=8 expert geometry (48 experts/rank,
hidden=7168, moe_intermediate=2048, top_k=8):

* gate_up GEMM: N = 2 * moe_intermediate = 4096, K = hidden = 7168
* down    GEMM: N = hidden = 7168,            K = moe_intermediate = 2048

(Packed weights are then ``w13 = [48, 4096, 896]`` and ``w2 = [48, 7168, 256]``
int32, matching the deployment.)

Decode sweeps M in {1, 2, 4, 8, 16, 32} tokens (x top_k experts -> the grouped M
the GEMM actually sees); a couple of prefill points (M in {512, 4096}) are
included for the large-tile configs.

Run on real gfx942 hardware (autotune needs a GPU). It also reports the achieved
effective weight-read bandwidth, which is the figure of merit in the decode
regime. This script intentionally does NOT submit any cluster job; run it
directly on a node you already hold.

Usage::

    python meta/benchmarks/bench_moe_int4_w4a16.py
    python meta/benchmarks/bench_moe_int4_w4a16.py --regime decode
"""

from __future__ import annotations

import argparse

import torch

from xkernels.ops.moe import make_w4a16_weights, moe_align_block_size_ref

# Kimi-K2.6 per-rank EP=8 geometry.
KIMI = dict(E=48, HIDDEN=7168, INTER=2048, TOP_K=8)


def _bench_one(M, E, N, K, top_k, group_size=32, block_m=16, iters=50, warmup=10):
    import triton
    import triton.language as tl

    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    dev = "cuda"
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    a = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M * top_k, device=dev, dtype=torch.float32)
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    # Token-indexed output [M*top_k, N] (see kernel docstring).
    c = torch.zeros((M * top_k, N), dtype=torch.bfloat16, device=dev)

    def run():
        int4_w4a16_moe_gemm(
            a, packed, scale, c, topk_w, sorted_ids, expert_ids, num_post,
            top_k=top_k, group_size=group_size, mul_routed_weight=False,
            compute_type=tl.bfloat16, filter_expert=False,
        )

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(run, rep=iters)

    # Effective bytes: each active expert reads its [N, K] packed weight once.
    active_experts = min(M * top_k, E)
    wbytes = active_experts * N * (K // 8) * 4  # int32 packed
    sbytes = active_experts * N * (K // group_size) * 2  # bf16 scales
    gbps = (wbytes + sbytes) / (ms * 1e-3) / 1e9
    return ms, gbps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["decode", "prefill", "all"], default="all")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU available; benchmark requires gfx942 (or any CUDA/ROCm GPU).")
        print("Run the correctness test under TRITON_INTERPRET=1 instead.")
        return

    decode_M = [1, 2, 4, 8, 16, 32]
    prefill_M = [512, 4096]
    gate_up = dict(N=2 * KIMI["INTER"], K=KIMI["HIDDEN"], tag="gate_up")
    down = dict(N=KIMI["HIDDEN"], K=KIMI["INTER"], tag="down")

    regimes = []
    if args.regime in ("decode", "all"):
        regimes += [("decode", m) for m in decode_M]
    if args.regime in ("prefill", "all"):
        regimes += [("prefill", m) for m in prefill_M]

    print(f"{'regime':8} {'gemm':8} {'M':>5} {'N':>6} {'K':>6} {'ms':>9} {'GB/s':>8}")
    for gemm in (gate_up, down):
        for tag, M in regimes:
            block_m = 16 if M <= 32 else 64
            ms, gbps = _bench_one(
                M, KIMI["E"], gemm["N"], gemm["K"], KIMI["TOP_K"], block_m=block_m
            )
            print(
                f"{tag:8} {gemm['tag']:8} {M:5d} {gemm['N']:6d} "
                f"{gemm['K']:6d} {ms:9.4f} {gbps:8.1f}"
            )


if __name__ == "__main__":
    main()
