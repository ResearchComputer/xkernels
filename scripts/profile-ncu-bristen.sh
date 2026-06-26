#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Profile ONE xkernels kernel with NVIDIA Nsight Compute (ncu) on a bristen
# A100 (sm_80). This is the NVIDIA counterpart of profile-rocprof-compute-beverin
# — ncu gives the per-kernel occupancy / stall-reason / memory / roofline data
# that the diagnose-low-occupancy / diagnose-memory-bound / map-to-matrix-cores
# skills branch on (the same vocabulary rocprof-compute yields on AMD).
#
# Run INSIDE the NGC PyTorch container on a compute node; the container ships ncu
# at /opt/nvidia/nsight-compute/<ver>/ncu. The caller (the sbatch wrapper) MUST
# pause DCGM first — see slurm/profile_ncu_bristen.sbatch — because DCGM holds the
# GPU perf counters and ncu fails with "driver resource unavailable" otherwise.
#
#   bash scripts/profile-ncu-bristen.sh <kernel> [roof|sq|full]
#
#   kernel : benchmarks/probe_ncu.py name
#            (dual_rmsnorm, moe_sum_reduce, fused_ffn, mha_merge_state)
#   mode   : roof = SpeedOfLight roofline + compute/memory workload + occupancy
#                    + launch stats  (DEFAULT — answers compute- vs memory-bound)
#            sq   = occupancy + warp stall reasons + scheduler stats
#                   (mirror of rocprof-compute's SQ block)
#            full = every section (slow — one replay per section group)
#
# NOTE on nvprof: bristen's nodes are A100 (sm_80). nvprof is unsupported on
# Volta and later, so it is NOT used here. Use ncu (this script) for per-kernel
# metrics and profile-nsys-bristen.sh for the system timeline.
#
# NOTE on ncu invocation quirks (Nsight Compute 2024.3.2, in the 24.10 container):
#   - Do NOT pass a bare `--` before the target; this build's option parser rejects
#     it as an ambiguous empty option.
#   - `-k` needs the `regex:` prefix to substring-match the Triton kernel name.
#
# Outputs: .ncu-workloads/<kernel>_<mode>/
#            <name>.ncu-rep    importable in the Nsight Compute GUI
#            <name>.report.txt human-readable section tables (also on stdout)
#            <name>.sol.csv    SpeedOfLight rows (peak-compute / HBM %, for the
#                              record_measurement compounding loop)
set -euo pipefail

KERNEL="${1:-dual_rmsnorm}"
MODE="${2:-roof}"
REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
WL="$REPO/.ncu-workloads"
NAME="${KERNEL}_${MODE}"
OUT="$WL/$NAME"
mkdir -p "$OUT"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

NCU="$(command -v ncu || true)"
[[ -n "$NCU" ]] || NCU="$(compgen -G /opt/nvidia/nsight-compute/*/ncu | head -1)"
[[ -n "$NCU" && -x "$NCU" ]] || { echo "[ncu] ncu binary not found in the container" >&2; exit 1; }
PROBE="$REPO/benchmarks/probe_ncu.py"

# Triton kernel names contain these fragments; -k matches them (regex) so ncu
# profiles only the op under test, not the autotune/launch helpers. For ops that
# dispatch several kernels (mhc_pre, moe_align_block_size), the fragment targets
# the dominant/representative one; -c 1 then samples its first steady-state launch.
case "$KERNEL" in
    dual_rmsnorm)         FRAG="rmsnorm";;            # dual_rmsnorm_kernel
    moe_sum_reduce)       FRAG="sum_reduce";;          # moe_sum_reduce_kernel
    fused_ffn)            FRAG="swiglu";;              # _swiglu_kernel (FFN epilogue)
    mha_merge_state)      FRAG="merge_state";;         # merge_state_kernel
    hc_prenorm_gemm)      FRAG="prenorm_gemm";;        # hc_prenorm_gemm_kernel
    mhc_pre)              FRAG="prenorm_gemm";;        # hc_prenorm_gemm_kernel dominates mhc_pre
    sparse_mla_attention) FRAG="sparse_mla";;          # sparse_mla_kernel
    moe_align_block_size) FRAG="align";;               # _align_stage* family
    moe_int4_w4a16)       FRAG="fused_moe_int4";;       # _fused_moe_int4_kernel
    mm_fp8_blockscale)    FRAG="blockscale";;          # mm_fp8_blockscale[_mfma]_kernel
    *)                    FRAG="$KERNEL";;
esac

case "$MODE" in
    roof) SECTIONS=(--section=SpeedOfLight --section=ComputeWorkloadAnalysis
                    --section=MemoryWorkloadAnalysis --section=LaunchStats
                    --section=Occupancy);;
    sq)   SECTIONS=(--section=Occupancy --section=LaunchStats
                    --section=WarpStateStats --section=SchedulerStats);;
    full) SECTIONS=(--set full);;
    *) echo "unknown mode '$MODE' (use roof|sq|full)" >&2; exit 2;;
esac

# Best-effort DCGM pause if dcgmi happens to be reachable inside the container;
# the reliable pause is done host-side by the sbatch wrapper.
if command -v dcgmi >/dev/null 2>&1; then
    dcgmi profile --pause >/dev/null 2>&1 || true
    trap 'dcgmi profile --resume >/dev/null 2>&1 || true' EXIT
fi

# --clock-control=none avoids clock-lock permission noise on shared nodes.
# -c 1 profiles one steady-state dispatch (the probe's warm-up already
# JIT-compiled + filled caches, so the first matching launch is the real kernel).
#
# This ncu build suppresses the stdout section text when --export is used, so we
# capture the run log, then regenerate the human-readable report via --import.
echo "[ncu] $NCU  kernel=$KERNEL (frag=$FRAG)  mode=$MODE  -> $OUT"
"$NCU" --target-processes=all --clock-control=none \
    -k "regex:$FRAG" -c 1 \
    "${SECTIONS[@]}" \
    --export="$OUT/$NAME" --force \
    python3 "$PROBE" "$KERNEL" >"$OUT/$NAME.run.log" 2>&1 || { cat "$OUT/$NAME.run.log"; exit 1; }
tail -6 "$OUT/$NAME.run.log"

# Human-readable section tables (SpeedOfLight roofline + workload + occupancy).
"$NCU" --import "$OUT/$NAME.ncu-rep" 2>/dev/null | tee "$OUT/$NAME.report.txt" | tail -70

# Parseable peak-utilization rows for record_measurement / the roofline math.
"$NCU" --import "$OUT/$NAME.ncu-rep" --csv --page raw 2>/dev/null \
    | grep -iE "sm__cycles_elapsed.avg|sm__pipe.*avg|dram__bytes.sum|achieved.*pct|gpu__time_duration.sum" \
    > "$OUT/$NAME.sol.csv" || true

echo "[ncu] saved: $OUT/$NAME.report.txt  (.ncu-rep: $OUT/$NAME.ncu-rep, run.log: $OUT/$NAME.run.log)"
