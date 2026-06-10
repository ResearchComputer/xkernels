# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Sweep the INT4 W4A16 fused-MoE autotune space on-device and persist winners.

For each Kimi-K2.6 production shape (gate_up: E=48,N=4096,K=7168; down:
E=48,N=7168,K=2048) and each token-batch M-bucket, time every valid candidate
config via the kernel's *direct* launch path (each candidate aligned to its own
BLOCK_SIZE_M) and keep the fastest. Writes one JSON per (E,N,K,device,dtype)
into the kernel's tuned_configs/ dir, mapping str(M) -> winning config.

Run on real gfx942 (needs a GPU); does NOT submit a cluster job::

    python benchmarks/tune_moe_int4_w4a16.py
    python benchmarks/tune_moe_int4_w4a16.py --M 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import torch

from xkernels.ops.moe import make_w4a16_weights, moe_align_block_size_ref
from xkernels.ops.moe.triton.configs import (
    _config_dir,
    _config_filename,
    _device_name,
    get_autotune_configs,
    prune_configs,
)

KIMI = dict(E=48, HIDDEN=7168, INTER=2048, TOP_K=8)


def _candidate_configs(N, K, group_size):
    # No num_valid_tokens -> prune does not apply the BLOCK_SIZE_M filter, so all
    # valid BMs are explored; each is benchmarked with matching alignment below.
    return prune_configs(
        get_autotune_configs(), {"group_k": group_size, "N": N, "K": K}
    )


def _bench_config(cfg, a, packed, scale, c, topk_ids, topk_w, top_k, group_size):
    import triton
    import triton.language as tl

    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    E = packed.shape[0]
    block_m = cfg.kwargs["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    launch_cfg = dict(cfg.kwargs)
    launch_cfg["num_warps"] = cfg.num_warps
    launch_cfg["num_stages"] = cfg.num_stages

    def run():
        int4_w4a16_moe_gemm(
            a, packed, scale, c, topk_w, sorted_ids, expert_ids, num_post,
            top_k=top_k, group_size=group_size, mul_routed_weight=False,
            compute_type=tl.bfloat16, filter_expert=False, config=launch_cfg,
        )

    try:
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run, rep=50)
    except Exception as exc:
        print(f"    skip {launch_cfg}: {str(exc)[:90]}")
        return float("inf")


def tune_shape(tag, E, N, K, top_k, group_size, Ms):
    dev = "cuda"
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    cands = _candidate_configs(N, K, group_size)
    table = {}
    for M in Ms:
        a = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        topk_ids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
        ).to(torch.int32)
        topk_w = torch.rand(M * top_k, device=dev, dtype=torch.float32)
        c = torch.zeros((M * top_k, N), dtype=torch.bfloat16, device=dev)
        best_ms, best = float("inf"), None
        for cfg in cands:
            ms = _bench_config(
                cfg, a, packed, scale, c, topk_ids, topk_w, top_k, group_size
            )
            if ms < best_ms:
                best_ms, best = ms, cfg
        if best is None:
            print(f"  [{tag}] M={M:5d} -> NO VALID CONFIG")
            continue
        entry = dict(best.kwargs)
        entry["num_warps"] = best.num_warps
        entry["num_stages"] = best.num_stages
        entry["_ms"] = round(best_ms, 5)
        table[str(M)] = entry
        print(f"  [{tag}] M={M:5d} -> {best_ms:.5f} ms  {entry}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--M",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 4096],
    )
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU; tuning requires gfx942 (or any CUDA/ROCm GPU).")
        return

    import triton

    device = _device_name()
    date = datetime.date.today().isoformat()
    os.makedirs(_config_dir(), exist_ok=True)
    shapes = [
        ("gate_up", 2 * KIMI["INTER"], KIMI["HIDDEN"]),
        ("down", KIMI["HIDDEN"], KIMI["INTER"]),
    ]
    for tag, N, K in shapes:
        print(f"== tuning {tag}: E={KIMI['E']} N={N} K={K} on {device} ==")
        table = tune_shape(tag, KIMI["E"], N, K, KIMI["TOP_K"], args.group_size, args.M)
        out = {
            "_provenance": {
                "device": device,
                "date": date,
                "triton": triton.__version__,
                "metric": "median ms, triton.do_bench, bf16 activations",
                "shape": {
                    "E": KIMI["E"], "N": N, "K": K,
                    "top_k": KIMI["TOP_K"], "group_size": args.group_size,
                },
            },
            **table,
        }
        path = os.path.join(
            _config_dir(), _config_filename(KIMI["E"], N, K, device, "int4_w4a16")
        )
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
