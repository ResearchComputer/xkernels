#!/bin/bash
# SPDX-License-Identifier: MIT
# Expert-parallel (ep_size>1) correctness for the INT4 W4A16 fused-MoE GEMM on
# beverin (gfx942 / MI300A) — issue #26. Runs the GPU bf16 EP test: the sum of
# per-rank partials must equal the full (non-EP) MoE output within 2e-2.
#
#   sbatch --export=ALL,REPO=<scratch-path> slurm/probe_moe_ep_beverin.sbatch
#
#SBATCH --job-name=xk-moe-ep
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=moe-ep-%j.out
#SBATCH --error=moe-ep-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  echo "=== GPU EP correctness (issue #26) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_int4_ep.py" -v
  echo "=== full INT4 MoE suite (regression) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_int4_w4a16.py" -q
'
