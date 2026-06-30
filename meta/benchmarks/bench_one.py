# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Run ONE kernel's bench (from meta/benchmarks/bench_all.py) in its own process.

Why this exists: the NGC PyTorch 24.10 container ships Triton 3.0.0, whose MLIR
`OptimizeThreadLocality` pass segfaults (SIGSEGV) on some reduction kernels when
targeting sm_80. That native crash kills the whole `bench_all.py` process, so the
bristen benchmark loses every row after the crashing kernel. This wrapper runs a
single kernel's bench function per invocation: a SIGSEGV only loses that one row,
and a shell loop over kernels (each calling this script) recovers the survivors
and pinpoints exactly which kernel(s) trigger the Triton bug.

Usage (bristen, one process per kernel):
    for k in merge_state sparse_mla mhc_prenorm mhc_pre_post \\
             dual_rmsnorm moe_sum_reduce moe_align ffn moe_int4; do
        python -u meta/benchmarks/bench_one.py "$k" || echo "FAILED $k"
    done
"""
from __future__ import annotations

import sys

import benchmarks.bench_all as B

# CLI name -> bench_all function (matches bench_all.main()'s loop order).
FNS = {
    "merge_state": B.bench_merge_state,
    "sparse_mla": B.bench_sparse_mla,
    "mhc_prenorm": B.bench_mhc_prenorm,
    "mhc_pre_post": B.bench_mhc_pre_post,
    "dual_rmsnorm": B.bench_dual_rmsnorm,
    "moe_sum_reduce": B.bench_moe_sum_reduce,
    "moe_align": B.bench_moe_align,
    "ffn": B.bench_ffn,
    "moe_int4": B.bench_moe_int4,
}


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("No GPU available.")
    name = sys.argv[1] if len(sys.argv) > 1 else "dual_rmsnorm"
    if name not in FNS:
        raise SystemExit(f"unknown kernel {name!r}; choose from {sorted(FNS)}")
    dev = "cuda"
    print(f"device: {torch.cuda.get_device_name(0)}  |  dtype: {B.DT}  |  kernel: {name}")
    FNS[name](dev)
    for kernel, shape, label, naive_ms, opt_ms in B.RESULTS:
        if opt_ms <= 0:
            print(f"| `{kernel}` | {shape} | n/a ({label}) | n/a | — |")
            continue
        print(
            f"| `{kernel}` | {shape} | {naive_ms:.3f} ms ({label}) "
            f"| {opt_ms:.3f} ms | **{naive_ms / opt_ms:.2f}×** |"
        )


if __name__ == "__main__":
    main()
