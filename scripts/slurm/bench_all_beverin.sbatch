#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU consolidated kernel benchmark on beverin (gfx942 / MI300A).
# Runs meta/benchmarks/bench_all.py: each op's optimized backend vs naive PyTorch.
#
#   sbatch scripts/slurm/bench_all_beverin.sbatch
#
#SBATCH --job-name=xk-bench-all
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=bench-all-%j.out
#SBATCH --error=bench-all-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/meta/benchmarks/bench_all.py"
'
