# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Microbenchmark for the MXFP4 fused-MoE GEMM over DeepSeek-V4-Flash shapes.

Compares the xkernels Triton grouped GEMM (inline E2M1 + ue8m0 dequant, fused
clamped-SwiGLU + routed combine) against the correctness-first per-expert torch
dequant loop it replaces (tokenspeed's ``Mxfp4DequantBackend``).

V4-Flash per-rank (TP=4) geometry: 256 experts, top-6, hidden=4096,
moe_intermediate=2048 -> ispp=512. Packed weights are then
``w13 = [256, 1024, 2048]`` and ``w2 = [256, 4096, 256]`` uint8.

Decode sweeps M in {1, 2, 4, 8, 16, 32, 48} tokens; a couple of prefill points
(M in {256, 1024}) are included. Run on real gfx942 hardware (autotune needs a
GPU). This script does NOT submit any cluster job; run it on a node you hold.

Usage::

    python benchmarks/bench_moe_mxfp4.py
    python benchmarks/bench_moe_mxfp4.py --regime decode
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from xkernels import fused_moe_mxfp4
from xkernels._backends import Backend
from xkernels.ops.moe.mxfp4 import dequant_mxfp4_weight, make_mxfp4_moe_weights

# V4-Flash per-rank (TP=4) geometry.
V4 = dict(E=256, HIDDEN=4096, ISPP=512, TOP_K=6, LIMIT=10.0)


def _torch_loop(A, w, topk_ids, topk_w, L):
    M, hidden = A.shape
    top_k = topk_ids.shape[1]
    out = torch.zeros(M, hidden, dtype=torch.float32, device=A.device)
    flat = topk_ids.reshape(-1)
    tok = torch.arange(M, device=A.device).repeat_interleave(top_k)
    fw = topk_w.reshape(-1).float()
    for e in torch.unique(flat).tolist():
        sel = flat == e
        t = tok[sel]
        wt = fw[sel]
        w13e = dequant_mxfp4_weight(w["w13"][e], w["w13_scale"][e], 32)
        gu = A[t] @ w13e.T + w["b13"][e]
        g, u = gu.float().chunk(2, -1)
        g = torch.clamp(g, max=L)
        u = torch.clamp(u, -L, L)
        act = (F.silu(g) * u).to(torch.bfloat16)
        w2e = dequant_mxfp4_weight(w["w2"][e], w["w2_scale"][e], 32)
        d = (act @ w2e.T).float() + w["b2"][e].float()
        out.index_add_(0, t, d * wt.unsqueeze(-1))
    return out


def _bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["decode", "prefill", "all"], default="all")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU (autotune + bf16)")
    dev = "cuda"
    E, hidden, ispp, top_k, L = V4["E"], V4["HIDDEN"], V4["ISPP"], V4["TOP_K"], V4["LIMIT"]
    w = make_mxfp4_moe_weights(E, hidden, ispp, group_size=32, with_bias=True, device=dev, seed=1)
    ms = []
    if args.regime in ("decode", "all"):
        ms += [1, 2, 4, 8, 16, 32, 48]
    if args.regime in ("prefill", "all"):
        ms += [256, 1024]

    print(f"V4-Flash MXFP4 MoE  E={E} hidden={hidden} ispp={ispp} top_k={top_k}")
    print(f"{'M':>6} {'torch-loop(ms)':>16} {'triton(ms)':>12} {'speedup':>9} {'max|err|':>10}")
    for M in ms:
        A = (torch.randn(M, hidden, device=dev) * 0.1).to(torch.bfloat16)
        topk_ids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
        ).to(torch.int32)
        topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)

        def triton_path(A=A, topk_ids=topk_ids, topk_w=topk_w):
            return fused_moe_mxfp4(
                A, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], topk_ids, topk_w,
                b13=w["b13"], b2=w["b2"], swiglu_limit=L, group_size=32,
                backend=Backend.TRITON,
            ).float()

        def loop(A=A, topk_ids=topk_ids, topk_w=topk_w):
            return _torch_loop(A, w, topk_ids, topk_w, L)

        got = triton_path()
        ref = loop()
        err = (got - ref).abs().max().item()
        t_loop = _bench(loop)
        t_trit = _bench(triton_path)
        print(f"{M:>6} {t_loop:>16.3f} {t_trit:>12.3f} {t_loop / t_trit:>8.2f}x {err:>10.3e}")


if __name__ == "__main__":
    main()
