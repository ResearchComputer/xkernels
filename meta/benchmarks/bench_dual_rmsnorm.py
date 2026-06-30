# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark: fused dual RMSNorm vs two sequential RMSNorm launches.

MLA q_a (q_lora_rank=1536) / kv_a (kv_lora_rank=512) latents over a sweep of
token counts. Needs a GPU you already hold; does not submit a cluster job.

Usage::

    python meta/benchmarks/bench_dual_rmsnorm.py
"""

from __future__ import annotations

import torch

from xkernels import dual_rmsnorm
from xkernels.ops.norm.reference import rmsnorm

D1, D2 = 1536, 512  # DeepSeek-V3 / Kimi q_lora_rank, kv_lora_rank


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    dev = "cuda"
    print(f"{'T':>6} {'fused_ms':>10} {'2x_seq_ms':>10} {'speedup':>8}")
    for T in [256, 1024, 4096, 16384]:
        x1 = torch.randn(T, D1, device=dev, dtype=torch.bfloat16)
        x2 = torch.randn(T, D2, device=dev, dtype=torch.bfloat16)
        w1 = torch.randn(D1, device=dev, dtype=torch.bfloat16)
        w2 = torch.randn(D2, device=dev, dtype=torch.bfloat16)

        fused = triton.testing.do_bench(
            lambda x1=x1, w1=w1, x2=x2, w2=w2: dual_rmsnorm(x1, w1, x2, w2)
        )
        seq = triton.testing.do_bench(
            lambda x1=x1, w1=w1, x2=x2, w2=w2: (rmsnorm(x1, w1), rmsnorm(x2, w2))
        )
        print(f"{T:6d} {fused:10.4f} {seq:10.4f} {seq / fused:7.2f}x")


if __name__ == "__main__":
    main()
