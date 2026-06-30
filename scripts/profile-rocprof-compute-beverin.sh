#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Profile + analyze one xkernels kernel with AMD ROCm Compute Profiler
# (formerly Omniperf) on a beverin GPU. Self-contained: activates the scratch
# install, mirrors the runtime libs into the ROCm lib dir, runs `profile`, then
# `analyze`. Run inside the tokenspeed container env on a compute node:
#
#   bash scripts/profile-rocprof-compute-beverin.sh <kernel> [roof|sq|full]
#
#   kernel : meta/benchmarks/probe_omniperf.py name (dual_rmsnorm, moe_sum_reduce,
#            fused_ffn, mha_merge_state)
#   mode   : roof  = roofline + default metric set (DEFAULT; answers
#                    compute-vs-memory-bound + how far from the line)
#            sq    = SQ scheduler block only (occupancy + stall reasons)
#            full  = every block (many rocprof passes, slow)
#
# Outputs: .omniperf-workloads/<kernel>_<mode>/ (raw + pmc_perf.csv, roofline.csv,
# empirRoof_*.pdf) and <name>.analyze.txt (the metric tables). Submit via
# scripts/slurm/profile_omniperf_beverin.sbatch on the contended mi300 partition.
set -euo pipefail

KERNEL="${1:-dual_rmsnorm}"
MODE="${2:-roof}"
REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
: "${SRC:=/capstor/scratch/cscs/xyao/rocprof-compute-src}"
: "${PYLIBS:=/capstor/scratch/cscs/xyao/rocprof-compute-pylibs}"
: "${LIBS:=/capstor/scratch/cscs/xyao/rocprof-compute-libs}"
WL="$REPO/.omniperf-workloads"
NAME="${KERNEL}_${MODE}"
RCP=(python3 "$SRC/src/rocprof-compute")

# rocprof-compute RESETS the profiler subprocess LD_LIBRARY_PATH to the ROCm lib
# dir only (profiler_rocprofiler_sdk.py ~L73), so mirror the staged runtime libs
# (libdw.so.1 + deps) into /opt/rocm/lib — the one path the loader searches.
# /opt/rocm/lib is writable but per-container-instance, so redo every run.
if [ -d "$LIBS" ]; then
    cp -f "$LIBS"/*.so* /opt/rocm/lib/ 2>/dev/null || true
else
    echo "[omniperf] WARNING: $LIBS missing — run scripts/stage-rocprof-compute-libs-beverin.sh on the login node first" >&2
fi

export PYTHONPATH="$PYLIBS:$SRC/src:$REPO/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$LIBS:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export ROCPC_SRC="$SRC" ROCPC_PYLIBS="$PYLIBS" ROCPC_LIBS="$LIBS"

case "$MODE" in
    roof) PROF_FLAGS=();;                 # default metric set incl. roofline
    sq)   PROF_FLAGS=(-b SQ);;
    full) PROF_FLAGS=(-b SQ LDS SQC TA TD TCP TCC SPI CPC CPF);;
    *) echo "unknown mode '$MODE' (use roof|sq|full)" >&2; exit 2;;
esac

# profile -p is the OUTPUT dir (it writes pmc_perf.csv etc. directly there; -n
# is only a label). Use one dir per (kernel,mode) so runs don't clobber.
OUT="$WL/$NAME"
mkdir -p "$OUT"
# The workload subprocess needs its own PYTHONPATH to find xkernels.
WORKLOAD=(env "PYTHONPATH=$PYTHONPATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
          python3 "$REPO/meta/benchmarks/probe_omniperf.py" "$KERNEL")

echo "[omniperf] profile $KERNEL mode=$MODE -> $OUT"
"${RCP[@]}" profile -n "$NAME" "${PROF_FLAGS[@]}" -p "$OUT" -- "${WORKLOAD[@]}" 2>&1 | tail -15

echo "[omniperf] analyze $OUT"
"${RCP[@]}" analyze -p "$OUT" 2>&1 | tee "$OUT.analyze.txt" | tail -60
echo "[omniperf] saved: $OUT.analyze.txt  (raw data: $OUT/)"
