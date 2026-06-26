# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""End-to-end fused-MoE timing, routing *included* (issue #50).

The public fused INT4 / MXFP4 MoE launchers now build the sort/pad dispatch
through the sync-free Triton align backend (``truncate=False``) instead of the
torch argsort + Python-padding reference, and under expert parallelism remap
global -> local expert ids device-side (the "ghost expert" path) with no
``e_local = .sum().item()`` host sync. So an end-to-end ``fused_moe_*`` call now
*includes* that fast routing — this benchmark times the whole call (alignment +
GEMM), not the GEMM in isolation.

To make the win legible it also reports the **reference** routing time alone
(``moe_align_block_size_ref`` / ``moe_align_block_size_ep``), i.e. the per-call
host-side tax the launchers used to pay before every GEMM. That tax is now gone
from the GPU path (the reference helpers remain the CPU / Triton-interpreter
fallback).

Sweeps decode buckets (``M = 1, 2, 4, 8, 16``) and a prefill bucket, both with
and without ``expert_map`` (expert parallelism). Needs a GPU you already hold;
does not submit a cluster job.

Usage::

    python benchmarks/bench_moe_e2e_routing.py
    python benchmarks/bench_moe_e2e_routing.py --op mxfp4
"""

from __future__ import annotations

import argparse

import torch

from xkernels import fused_moe_int4_w4a16, fused_moe_mxfp4
from xkernels._backends import Backend
from xkernels.ops.moe import make_w4a16_weights
from xkernels.ops.moe.mxfp4 import make_mxfp4_moe_weights
from xkernels.ops.moe.triton.align_kernel import (
    moe_align_block_size_ep_triton,
    moe_align_block_size_triton,
)
from xkernels.ops.moe.w4a16 import moe_align_block_size_ep, moe_align_block_size_ref

DECODE_M = [1, 2, 4, 8, 16]
PREFILL_M = [512]
GROUP = 32
BLOCK_M = 16


def _ep_map(num_experts, ep_size, rank, dev):
    """Contiguous EP slice for ``rank``: owns experts [rank*per, (rank+1)*per)."""
    per = num_experts // ep_size
    lo, hi = rank * per, (rank + 1) * per
    emap = torch.full((num_experts,), -1, dtype=torch.int32, device=dev)
    emap[lo:hi] = torch.arange(per, dtype=torch.int32, device=dev)
    return emap


def _bench(fn):
    import triton

    return triton.testing.do_bench(fn)


def _route_times(topk_ids, num_experts, emap):
    """Reference (old per-call tax) vs Triton (new) routing time for one bucket."""
    if emap is not None:
        ref = _bench(lambda: moe_align_block_size_ep(topk_ids, BLOCK_M, num_experts, emap))
        tri = _bench(
            lambda: moe_align_block_size_ep_triton(
                topk_ids, BLOCK_M, num_experts, emap, truncate=False
            )
        )
    else:
        ref = _bench(lambda: moe_align_block_size_ref(topk_ids, BLOCK_M, num_experts))
        tri = _bench(
            lambda: moe_align_block_size_triton(topk_ids, BLOCK_M, num_experts, truncate=False)
        )
    return ref, tri


def _row(M, e2e, ref_route, tri_route):
    print(f"{M:6d} {e2e:9.4f} {ref_route:13.4f} {tri_route:13.4f}")


def bench_int4_bucket(dev, M, packed, scale, topk_ids, topk_w, emap, num_experts):
    A = (torch.randn(M, packed.shape[2] * 8, device=dev) * 0.1).to(torch.bfloat16)
    e2e = _bench(lambda: fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w, group_size=GROUP,
        backend=Backend.TRITON, expert_map=emap,
    ))
    return e2e


def bench_int4(dev, with_ep):
    # Kimi-K2.6 per-rank EP geometry (E=48 local, hidden=7168 -> K, inter=2048 -> N).
    E, K, N, top_k = 48, 7168, 4096, 8
    ep_size = 4 if with_ep else 1
    packed, scale, _ = make_w4a16_weights(E, N, K, GROUP, device=dev, seed=0)
    tag = f"on, ep_size={ep_size}" if with_ep else "off"
    print(f"\n=== INT4 W4A16 fused MoE (E={E} K={K} N={N} top_k={top_k} ep={tag}) ===")
    if with_ep:
        # INT4 EP keeps the reference align (issue #50 fallback): the device-side
        # ghost routing is logic-correct but the INT4 GEMM's autotune wrapper
        # corrupts for it (see launcher comment). So e2e_ms below INCLUDES the
        # reference routing tax; tri_route_ms is the untapped device-side speedup.
        print("    (e2e uses REFERENCE routing; tri_route = available device-side align)")
    print(f"{'M':>6} {'e2e_ms':>9} {'ref_route_ms':>13} {'tri_route_ms':>13}")
    for M in DECODE_M + PREFILL_M:
        g = torch.Generator(device=dev).manual_seed(M)
        topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)
        topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
        emap = _ep_map(E, ep_size, 0, dev) if with_ep else None
        e2e = bench_int4_bucket(dev, M, packed, scale, topk_ids, topk_w, emap, E)
        ref_route, tri_route = _route_times(topk_ids, E, emap)
        _row(M, e2e, ref_route, tri_route)


def bench_mxfp4_bucket(dev, M, w, topk_ids, topk_w, emap, num_experts, hidden):
    A = (torch.randn(M, hidden, device=dev) * 0.1).to(torch.bfloat16)
    e2e = _bench(lambda: fused_moe_mxfp4(
        A, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], topk_ids, topk_w,
        swiglu_limit=10.0, group_size=GROUP, backend=Backend.TRITON, expert_map=emap,
    ))
    return e2e


def bench_mxfp4(dev, with_ep):
    # DeepSeek-V4-Flash per-rank (TP=4) geometry.
    E, hidden, ispp, top_k = 256, 4096, 512, 6
    ep_size = 8 if with_ep else 1
    w = make_mxfp4_moe_weights(E, hidden, ispp, group_size=GROUP, device=dev, seed=0)
    tag = f"on, ep_size={ep_size}" if with_ep else "off"
    print(f"\n=== MXFP4 fused MoE (E={E} hidden={hidden} ispp={ispp} top_k={top_k} ep={tag}) ===")
    print(f"{'M':>6} {'e2e_ms':>9} {'ref_route_ms':>13} {'tri_route_ms':>13}")
    for M in DECODE_M + PREFILL_M:
        g = torch.Generator(device=dev).manual_seed(M)
        topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)
        topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
        emap = _ep_map(E, ep_size, 0, dev) if with_ep else None
        e2e = bench_mxfp4_bucket(dev, M, w, topk_ids, topk_w, emap, E, hidden)
        ref_route, tri_route = _route_times(topk_ids, E, emap)
        _row(M, e2e, ref_route, tri_route)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness tests under TRITON_INTERPRET=1.")
        return
    dev = "cuda"
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", choices=["int4", "mxfp4", "both"], default="both")
    args = ap.parse_args()
    if args.op in ("int4", "both"):
        bench_int4(dev, with_ep=False)
        bench_int4(dev, with_ep=True)
    if args.op in ("mxfp4", "both"):
        bench_mxfp4(dev, with_ep=False)
        bench_mxfp4(dev, with_ep=True)


if __name__ == "__main__":
    main()
