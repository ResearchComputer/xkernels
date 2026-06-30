# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark: moe_align_block_size Triton kernel vs the torch reference.

Sorts/pads routed token-slots into per-expert blocks (issue #4) over a sweep of
token counts at Kimi-K2.6 MoE geometry (top_k=8, num_experts=48, block_size=16,
the decode BLOCK_SIZE_M). Reports both backends honestly: the win is
shape-dependent — the Triton kernel's scalar per-token chunk loops (the vLLM
Triton-fallback shape) trade off against the reference's vectorized
``argsort + bincount`` plus its per-expert python padding loop and host syncs.

Needs a GPU you already hold; does not submit a cluster job.

Usage::

    python meta/benchmarks/bench_moe_align_block_size.py
"""

from __future__ import annotations

import torch

from xkernels import moe_align_block_size

TOP_K, NUM_EXPERTS, BLOCK = 8, 48, 16  # Kimi-K2.6 top_k / experts; decode block


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    dev = "cuda"
    print(f"top_k={TOP_K} num_experts={NUM_EXPERTS} block_size={BLOCK}")
    print(f"{'M':>6} {'triton_ms':>10} {'torch_ms':>10} {'speedup':>8}")
    for M in [16, 64, 256, 1024, 4096, 16384]:
        g = torch.Generator(device=dev).manual_seed(0)
        topk_ids = torch.randint(
            0, NUM_EXPERTS, (M, TOP_K), generator=g, dtype=torch.int32, device=dev
        )
        tri = triton.testing.do_bench(
            lambda t=topk_ids: moe_align_block_size(t, BLOCK, NUM_EXPERTS, backend="triton")
        )
        ref = triton.testing.do_bench(
            lambda t=topk_ids: moe_align_block_size(t, BLOCK, NUM_EXPERTS, backend="reference")
        )
        print(f"{M:6d} {tri:10.4f} {ref:10.4f} {ref / tri:7.2f}x")


if __name__ == "__main__":
    main()
