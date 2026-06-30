"""Sweep FFN shapes across available backends; print a markdown table.

Usage: python meta/benchmarks/bench_ffn.py [--dtype float16]
"""
from __future__ import annotations

import argparse

import torch

from xkernels import fused_ffn
from xkernels._dispatch import registered_backends
from xkernels.utils.benchmarking import benchmark

SHAPES = [(2048, 4096, 11008), (4096, 4096, 11008), (8192, 8192, 28672)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backends = registered_backends("ffn")

    print("| M | d_model | d_ff | " + " | ".join(b.name for b in backends) + " |")
    print("|---|---|---|" + "|".join(["---"] * len(backends)) + "|")
    for M, d_model, d_ff in SHAPES:
        x = torch.randn(M, d_model, device=device, dtype=dtype)
        wg = torch.randn(d_model, d_ff, device=device, dtype=dtype)
        wu = torch.randn(d_model, d_ff, device=device, dtype=dtype)
        wd = torch.randn(d_ff, d_model, device=device, dtype=dtype)
        times = []
        for b in backends:
            try:
                ms = benchmark(
                    lambda b=b, x=x, wg=wg, wu=wu, wd=wd: fused_ffn(
                        x, wg, wu, wd, backend=b
                    )
                )
                times.append(f"{ms:.3f}ms")
            except Exception:
                times.append("n/a")  # backend not runnable on this device
        print(f"| {M} | {d_model} | {d_ff} | " + " | ".join(times) + " |")


if __name__ == "__main__":
    main()
