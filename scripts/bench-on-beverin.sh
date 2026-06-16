#!/usr/bin/env bash
# Push the local tree to beverin and submit a SLURM benchmark job.
#
# Usage:
#   scripts/bench-on-beverin.sh [slurm_script]
#
# Examples:
#   scripts/bench-on-beverin.sh                    # runs slurm/bench_all_beverin.sbatch
#   scripts/bench-on-beverin.sh slurm/bench_moe_combine_beverin.sbatch
#   scripts/bench-on-beverin.sh slurm/test_mhc_pre_post_beverin.sbatch

set -euo pipefail

SCRIPT="${1:-slurm/bench_all_beverin.sbatch}"
REPO_REMOTE="/capstor/scratch/cscs/xyao/xkernels"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: SLURM script not found: $SCRIPT" >&2
    exit 1
fi

echo "[rcc] pushing to beverin..."
rcc push

echo "[rcc] submitting $SCRIPT on beverin (REPO=$REPO_REMOTE)..."
JOBID=$(rcc run -- env REPO="$REPO_REMOTE" sbatch "$SCRIPT" | awk '{print $NF}')
echo "[rcc] submitted job $JOBID"
echo "[rcc] tail output: rcc run -- tail -f bench-all-${JOBID}.out"
