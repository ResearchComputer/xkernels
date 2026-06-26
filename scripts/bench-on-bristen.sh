#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Push the local tree to bristen and submit a SLURM job.
#
# Usage:
#   scripts/bench-on-bristen.sh [slurm_script]
#
# Examples:
#   scripts/bench-on-bristen.sh                           # slurm/bench_all_bristen.sbatch
#   scripts/bench-on-bristen.sh slurm/bench_all_bristen.sbatch
#   scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
#   KERNEL=dual_rmsnorm MODE=sq scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch

set -euo pipefail

SCRIPT="${1:-slurm/bench_all_bristen.sbatch}"
REPO_REMOTE="/capstor/scratch/cscs/xyao/xkernels"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: SLURM script not found: $SCRIPT" >&2
    exit 1
fi

echo "[rcc] pushing to bristen..."
rcc --profile bristen push

echo "[rcc] submitting $SCRIPT on bristen (REPO=$REPO_REMOTE)..."
JOBID=$(rcc --profile bristen run -- env REPO="$REPO_REMOTE" sbatch "$SCRIPT" | awk '{print $NF}')
echo "[rcc] submitted job $JOBID"
echo "[rcc] follow: rcc --profile bristen run -- squeue -j $JOBID"
echo "[rcc] tail output: rcc --profile bristen run -- tail -f ${SCRIPT##*/}  # see the sbatch's --output= name"
