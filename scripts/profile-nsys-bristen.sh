#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# System-level profile (NVIDIA Nsight Systems, nsys) of one xkernels kernel on a
# bristen A100. nsys has NO per-kernel replay overhead, so use it for:
#   - steady-state kernel duration (cross-check verify()'s `ms`)
#   - the full CUDA API + kernel + memory-op timeline
#   - host-side gaps / launch overhead between dispatches
# Use profile-ncu-bristen.sh for the deep per-kernel (occupancy/stall/roofline)
# metrics — the two tools are complementary, not substitutes.
#
#   bash scripts/profile-nsys-bristen.sh <kernel>
#
# Outputs: .nsys-workloads/<kernel>/
#            <kernel>.stats.txt    the auto-printed CUDA API/kernel/memory stats
#            <kernel>.nsys-rep     importable in the Nsight Systems GUI
#            <kernel>.sqlite       queryable (one row per event)
set -euo pipefail

KERNEL="${1:-dual_rmsnorm}"
REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
WL="$REPO/.nsys-workloads"
OUT="$WL/$KERNEL"
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

NSYS="$(command -v nsys || true)"
[[ -n "$NSYS" ]] || NSYS="$(compgen -G /opt/nvidia/nsight-systems*/bin/nsys | head -1 || true)"
[[ -n "$NSYS" && -x "$NSYS" ]] || NSYS="/usr/local/cuda/bin/nsys"
PROBE="$REPO/benchmarks/probe_ncu.py"

echo "[nsys] $NSYS  kernel=$KERNEL  -> $OUT"
# --stats=true prints the CUDA GPU/kernel/API + memory summary tables to stdout.
# The probe warm-runs before the measured loop, so the reported steady-state
# kernel duration is the real one (cross-check vs verify()'s ms).
"$NSYS" profile --trace=cuda,nvtx,osrt --stats=true \
    --export=sqlite -o "$OUT/$KERNEL" --force-overwrite=true \
    python3 "$PROBE" "$KERNEL" 2>&1 | tee "$OUT/$KERNEL.stats.txt" | tail -90

echo "[nsys] saved: $OUT/$KERNEL.stats.txt  (.nsys-rep: $OUT/$KERNEL.nsys-rep, sqlite: $OUT/$KERNEL.sqlite)"
