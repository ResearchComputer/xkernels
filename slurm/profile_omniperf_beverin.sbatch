#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU ROCm Compute Profiler job on beverin (gfx942 / MI300A).
# Runs the scratch-installed profiler (see scripts/setup-rocprof-compute-beverin.sh)
# over one xkernels kernel, then analyzes it. Override KERNEL/MODE to profile
# other kernels.
#
#   KERNEL=dual_rmsnorm MODE=sq sbatch slurm/profile_omniperf_beverin.sbatch
#   sbatch slurm/profile_omniperf_beverin.sbatch
#
#SBATCH --job-name=xk-omniperf
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=omniperf-%j.out
#SBATCH --error=omniperf-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
KERNEL="${KERNEL:-dual_rmsnorm}"
MODE="${MODE:-roof}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME KERNEL=$KERNEL MODE=$MODE node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  cd "'"$REPO"'"
  bash scripts/profile-rocprof-compute-beverin.sh "'"$KERNEL"'" "'"$MODE"'"
'
