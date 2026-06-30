# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Compare the fused top-k combine epilogue vs the unfused (GEMM + moe_sum_reduce
kernel) path for the INT4 W4A16 MoE down GEMM (issue #20).

The block-align dispatch is built once outside the timed region (it is a separate
kernel), so this isolates the combine: the unfused path is the grouped GEMM into a
[M*top_k, hidden] buffer followed by the standalone ``moe_sum_reduce`` kernel
(weights applied in the reduce); the fused path is one GEMM that atomic-
accumulates the weighted result into [M, hidden] directly (and must pre-zero the
fp32 buffer). Reports per-path latency across decode M.

Run on gfx942 (scripts/archive/issues/bench_moe_combine_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels.ops.moe import make_w4a16_weights, moe_sum_reduce

# Kimi-K2.6 per-rank down GEMM (the combine target): N = hidden, K = moe_inter.
KIMI = dict(E=48, N=7168, K=2048, TOP_K=8, GS=32)
DECODE_M = [1, 2, 4, 8, 16]


def _inputs(M, dev):
    packed, scale, _ = make_w4a16_weights(
        KIMI["E"], KIMI["N"], KIMI["K"], KIMI["GS"], device=dev, seed=1
    )
    A = (torch.randn(M, KIMI["K"], device=dev) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(KIMI["E"], device=dev)[: KIMI["TOP_K"]] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, KIMI["TOP_K"], device=dev, dtype=torch.float32)
    return packed, scale, A, topk_ids, topk_w


def main():
    if not torch.cuda.is_available():
        print("No GPU; this benchmark needs gfx942 (or any CUDA/ROCm GPU).")
        return
    import triton
    import triton.language as tl

    from xkernels.ops.moe import moe_align_block_size_ref
    from xkernels.ops.moe.triton.configs import align_block_m, get_moe_int4_config
    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    E, N, K, top_k, gs = KIMI["E"], KIMI["N"], KIMI["K"], KIMI["TOP_K"], KIMI["GS"]
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"down GEMM: E={E} N={N} K={K} top_k={top_k}  (align built once, outside timing)")
    print(f"{'M':>4} {'gemm+reduce_ms':>15} {'fused_ms':>9} {'speedup':>8}  max|err|")
    for M in DECODE_M:
        packed, scale, A, topk_ids, topk_w = _inputs(M, "cuda")
        config = get_moe_int4_config(E, N, K, M)
        block_m = config["BLOCK_SIZE_M"] if config is not None else align_block_m(M)
        sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
        topk_w_flat = topk_w.reshape(-1).float()
        c = torch.zeros((M * top_k, N), dtype=torch.bfloat16, device="cuda")
        out = torch.zeros((M, N), dtype=torch.float32, device="cuda")

        def unfused(c=c, sorted_ids=sorted_ids, expert_ids=expert_ids, num_post=num_post,
                    packed=packed, scale=scale, A=A, topk_w=topk_w, topk_w_flat=topk_w_flat,
                    config=config, M=M):
            # GEMM (un-weighted per-expert) -> [M*top_k, N], then the separate
            # moe_sum_reduce kernel applies the routing weights and sums over top_k.
            int4_w4a16_moe_gemm(
                A, packed, scale, c, topk_w_flat, sorted_ids, expert_ids, num_post,
                top_k=top_k, group_size=gs, mul_routed_weight=False,
                compute_type=tl.bfloat16, filter_expert=False, config=config,
            )
            return moe_sum_reduce(c.view(M, top_k, N), topk_w)

        def fused(out=out, sorted_ids=sorted_ids, expert_ids=expert_ids, num_post=num_post,
                  packed=packed, scale=scale, A=A, topk_w_flat=topk_w_flat, config=config):
            # One kernel: atomic-accumulate the weighted result into [M, N] fp32.
            out.zero_()  # the fp32 combine buffer must start zeroed
            int4_w4a16_moe_gemm(
                A, packed, scale, out, topk_w_flat, sorted_ids, expert_ids, num_post,
                top_k=top_k, group_size=gs, mul_routed_weight=True,
                compute_type=tl.float32, filter_expert=False, config=config, combine=True,
            )
            return out

        d = (unfused().float() - fused().float()).abs().max().item()
        u = triton.testing.do_bench(unfused, warmup=10, rep=50)
        f = triton.testing.do_bench(fused, warmup=10, rep=50)
        print(f"{M:4d} {u:15.4f} {f:9.4f} {u / f:8.2f}  {d:.4f}")


if __name__ == "__main__":
    main()
