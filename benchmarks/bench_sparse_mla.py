# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark: sparse-MLA attention compute (issue #32) Triton kernel vs a
naive gather+dense-softmax torch baseline a practitioner would write without it.

DeepSeek-V4 latent-MLA geometry: H heads x D=512 latent (d_v=448 value), MQA,
over a sweep of top-k. Needs a GPU you already hold; does not submit a cluster
job.

Usage::

    python benchmarks/bench_sparse_mla.py
"""

from __future__ import annotations

import torch

from xkernels import sparse_mla_attention
from xkernels._backends import Backend

T, H, D, D_V, Kv = 8, 128, 512, 448, 8192  # V4 prefill-ish tile


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    dev = "cuda"
    sm_scale = 1.0 / (D**0.5)
    print(f"{'topk':>6} {'triton_ms':>10} {'naive_ms':>10} {'speedup':>8}")
    for topk in [256, 512, 1024, 2048]:
        q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
        kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
        idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
        sink = torch.randn(H, device=dev)

        def naive(q=q, kv=kv, idx=idx, sink=sink, topk=topk):
            ks = kv[idx.long()]  # [T, topk, D]
            s = torch.einsum("thd,tkd->thk", q.float(), ks.float()) * sm_scale
            s = torch.cat([s, sink.float().view(1, H, 1).expand(T, H, 1)], dim=-1)
            p = s.softmax(-1)[..., :topk]
            return torch.einsum("thk,tkd->thd", p, ks[..., :D_V].float())

        def opt(q=q, kv=kv, idx=idx, sink=sink):
            return sparse_mla_attention(
                q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
                backend=Backend.TRITON,
            )

        tri = triton.testing.do_bench(opt)
        ref = triton.testing.do_bench(naive)
        print(f"{topk:6d} {tri:10.4f} {ref:10.4f} {ref / tri:7.2f}x")


if __name__ == "__main__":
    main()
