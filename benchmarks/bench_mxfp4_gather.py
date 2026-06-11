# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Time the Triton mxfp4_paged_gather (DeepSeek-V4 DSA indexer, issue #27) at
indexer decode shapes vs the torch oracle, and report max abs error.

The DSA indexer selects top-512 (Flash) / top-1024 (Pro) KV per query. Here we
gather num_seqs queries x topk positions of head_dim=128 mxfp4 KV from a paged
cache. Run on gfx942 (slurm/probe_mxfp4_gather_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels import mxfp4_paged_gather
from xkernels._backends import Backend
from xkernels.ops.gather.mxfp4 import make_mxfp4_kv
from xkernels.ops.gather.reference import mxfp4_paged_gather_ref

GROUP = 32
HEAD_DIM = 128
BLOCK_SIZE = 64
# (num_seqs, topk) — V4 Flash top-512 and Pro top-1024.
SHAPES = [(16, 512), (32, 512), (64, 512), (64, 1024)]


def _bench(fn, *args, iters=50, warmup=10, **kw):
    for _ in range(warmup):
        fn(*args, **kw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args, **kw)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        print("No GPU; this benchmark needs gfx942 (or any CUDA/ROCm GPU).")
        return
    dev = "cuda"
    print(f"{'num_seqs':>8} {'topk':>6} {'triton(ms)':>11} {'max|err|':>10}")
    for num_seqs, topk in SHAPES:
        max_pos = 2048
        num_blocks = (max_pos + BLOCK_SIZE - 1) // BLOCK_SIZE
        packed, scale, _ = make_mxfp4_kv(
            num_blocks, BLOCK_SIZE, HEAD_DIM, group_size=GROUP, device=dev, seed=2
        )
        g = torch.Generator(device=dev).manual_seed(7)
        max_blocks = num_blocks
        block_table = torch.arange(
            num_blocks, device=dev, dtype=torch.int32
        ).repeat(num_seqs, 1)[:, :max_blocks]
        sel_pos = torch.randint(
            0, max_pos, (num_seqs, topk), generator=g, device=dev, dtype=torch.int32
        )
        out = mxfp4_paged_gather(
            packed, scale, block_table, sel_pos, block_size=BLOCK_SIZE,
            group_size=GROUP, out_dtype=torch.bfloat16, backend=Backend.TRITON,
        )
        ref = mxfp4_paged_gather_ref(
            packed, scale, block_table, sel_pos, block_size=BLOCK_SIZE,
            group_size=GROUP, out_dtype=torch.bfloat16,
        )
        err = (out.float() - ref.float()).abs().max().item()
        t = _bench(
            mxfp4_paged_gather, packed, scale, block_table, sel_pos,
            block_size=BLOCK_SIZE, group_size=GROUP, out_dtype=torch.bfloat16,
            backend=Backend.TRITON,
        )
        print(f"{num_seqs:>8} {topk:>6} {t:>11.4f} {err:>10.4f}")


if __name__ == "__main__":
    main()
