#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Loop every bench_all.py kernel through meta/benchmarks/bench_one.py, ONE PROCESS PER
# KERNEL, so a native Triton-compiler SIGSEGV (Triton 3.0.0 OptimizeThreadLocality
# on sm_80) only loses that one row. Run inside the NGC PyTorch container on a
# bristen compute node; PYTHONPATH/REPO are set by the caller (the sbatch).
#
#   bash scripts/bench_kernel_loop_bristen.sh
set +e  # do NOT abort the loop on a per-kernel SIGSEGV
REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
cd "$REPO"

echo "| Kernel | Shape | naive PyTorch | optimized | speedup |"
echo "|--------|-------|--------------:|----------:|--------:|"
for k in merge_state sparse_mla mhc_prenorm mhc_pre_post \
         dual_rmsnorm moe_sum_reduce moe_align ffn moe_int4; do
    echo "########## $k ##########"
    python -u "$REPO/meta/benchmarks/bench_one.py" "$k" 2>"$REPO/bench1-bristen-$k.err"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "| \`$k\` | — | **FAILED (rc=$rc)** | — | — |"
        echo "--- stderr tail for $k ---"
        tail -5 "$REPO/bench1-bristen-$k.err" 2>/dev/null
    fi
done
