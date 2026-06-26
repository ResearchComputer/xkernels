#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU Nsight Compute profile on bristen (A100 / sm_80). Runs
# scripts/profile-ncu-bristen.sh over one xkernels kernel inside the NGC PyTorch
# container (torch + triton + ncu). Override KERNEL/MODE to profile others.
#
# This sbatch runs on the compute-node HOST so it can pause DCGM (which holds the
# GPU perf counters and otherwise makes ncu fail with "driver resource
# unavailable"); the actual ncu run happens in the container srun step.
#
#   KERNEL=dual_rmsnorm                  sbatch slurm/profile_ncu_bristen.sbatch
#   KERNEL=moe_sum_reduce MODE=sq        sbatch slurm/profile_ncu_bristen.sbatch
#
#SBATCH --job-name=xk-ncu
#SBATCH --account=a-infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=ncu-%j.out
#SBATCH --error=ncu-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
KERNEL="${KERNEL:-dual_rmsnorm}"
MODE="${MODE:-roof}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
DCGM="${DCGM:-/usr/bin/dcgmi}"

echo "REPO=$REPO IMAGE=$IMAGE KERNEL=$KERNEL MODE=$MODE node=$(hostname)"

# Pause DCGM's profiling field group so ncu can grab the perf counters. The trap
# always resumes it (even on error) so node monitoring is restored.
if [ -x "$DCGM" ]; then
    if "$DCGM" profile --pause >/dev/null 2>&1; then
        echo "[ncu] DCGM profiling paused"
        trap '"$DCGM" profile --resume >/dev/null 2>&1 || true' EXIT
    else
        echo "[ncu] WARN: could not pause DCGM — ncu may report 'driver resource unavailable'"
    fi
else
    echo "[ncu] WARN: dcgmi not found at $DCGM"
fi

srun --container-image="docker://$IMAGE" \
     --container-mounts=/capstor:/capstor,/iopsstor:/iopsstor \
     bash -c '
  set -e
  cd "'"$REPO"'"
  bash scripts/profile-ncu-bristen.sh "'"$KERNEL"'" "'"$MODE"'"
'
