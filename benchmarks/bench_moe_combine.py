# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Compare the fused top-k combine epilogue vs the unfused (GEMM + moe_sum_reduce)
path for the INT4 W4A16 MoE down GEMM (issue #20).

Both produce [M, hidden]; the fused path is one kernel and skips the
[M*top_k, hidden] intermediate. Reports per-path latency across decode M and the
kernel-count drop (2 -> 1). Run on gfx942 (slurm/bench_moe_combine_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels.ops.moe import fused_moe_int4_w4a16, make_w4a16_weights

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

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"down GEMM: E={KIMI['E']} N={KIMI['N']} K={KIMI['K']} top_k={KIMI['TOP_K']}")
    print(f"{'M':>4} {'unfused_ms':>11} {'fused_ms':>9} {'speedup':>8}  (2 kernels -> 1)")
    for M in DECODE_M:
        packed, scale, A, topk_ids, topk_w = _inputs(M, "cuda")

        def unfused(A=A, packed=packed, scale=scale, topk_ids=topk_ids, topk_w=topk_w):
            return fused_moe_int4_w4a16(
                A, packed, scale, topk_ids, topk_w, group_size=KIMI["GS"],
                mul_routed_weight=True, backend="triton", fused_combine=False,
            )

        def fused(A=A, packed=packed, scale=scale, topk_ids=topk_ids, topk_w=topk_w):
            return fused_moe_int4_w4a16(
                A, packed, scale, topk_ids, topk_w, group_size=KIMI["GS"],
                mul_routed_weight=True, backend="triton", fused_combine=True,
            )

        # Correctness guard: the two paths must agree before we trust the timing.
        d = (unfused().float() - fused().float()).abs().max().item()
        u = triton.testing.do_bench(unfused, warmup=10, rep=50)
        f = triton.testing.do_bench(fused, warmup=10, rep=50)
        print(f"{M:4d} {u:11.4f} {f:9.4f} {u / f:8.2f}  (max|err|={d:.4f})")


if __name__ == "__main__":
    main()
