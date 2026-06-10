#!/bin/bash
# SPDX-License-Identifier: MIT
# Sweep INT4 W4A16 fused-MoE configs on beverin (gfx942 / MI300A) and write the
# tuned-config JSON into the repo's tuned_configs/ dir (issue #16).
#
#   sbatch slurm/tune_moe_int4_beverin.sbatch
#
#SBATCH --job-name=xk-tune-moe-int4
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:40:00
#SBATCH --output=tune-moe-int4-%j.out
#SBATCH --error=tune-moe-int4-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/benchmarks/tune_moe_int4_w4a16.py"
'
