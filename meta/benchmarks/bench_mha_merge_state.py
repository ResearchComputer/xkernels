# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark: mha_merge_state Triton kernel vs the torch oracle.

MLA-ish geometry (H heads x D head-dim) over a sweep of token counts. Needs a
GPU you already hold; does not submit a cluster job.

Usage::

    python meta/benchmarks/bench_mha_merge_state.py
"""

from __future__ import annotations

import torch

from xkernels import mha_merge_state
from xkernels.ops.attention.reference import merge_state_ref

H, D = 128, 128  # heads, head dim


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    dev = "cuda"
    print(f"{'T':>6} {'triton_ms':>10} {'torch_ms':>10} {'speedup':>8}")
    for T in [256, 1024, 4096, 16384]:
        out_a = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        out_b = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        lse_a = torch.randn(T, H, device=dev)
        lse_b = torch.randn(T, H, device=dev)

        tri = triton.testing.do_bench(
            lambda oa=out_a, la=lse_a, ob=out_b, lb=lse_b: mha_merge_state(oa, la, ob, lb)
        )
        ref = triton.testing.do_bench(
            lambda oa=out_a, la=lse_a, ob=out_b, lb=lse_b: merge_state_ref(oa, la, ob, lb)
        )
        print(f"{T:6d} {tri:10.4f} {ref:10.4f} {ref / tri:7.2f}x")


if __name__ == "__main__":
    main()
