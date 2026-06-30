#!/bin/bash
# SPDX-License-Identifier: MIT
# Prove the sync-free moe_align (truncate=False) is HIP-graph capturable on
# beverin (gfx942 / MI300A) — issue #18.
#
#   sbatch scripts/slurm/probe_align_capture_beverin.sbatch
#
#SBATCH --job-name=xk-align-capture
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=align-capture-%j.out
#SBATCH --error=align-capture-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  echo "=== capture proof ==="
  python -u "'"$REPO"'/meta/benchmarks/probe_align_capture.py"
  echo "=== GPU correctness (truncate modes) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_align_block_size.py" -q
'
