#!/bin/bash
# SPDX-License-Identifier: MIT
# Fused top-k combine vs (GEMM + moe_sum_reduce) for the INT4 MoE down GEMM on
# beverin (gfx942 / MI300A) — issue #20.
#
#   sbatch slurm/bench_moe_combine_beverin.sbatch
#
#SBATCH --job-name=xk-moe-combine
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=moe-combine-%j.out
#SBATCH --error=moe-combine-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  echo "=== fused-combine bench ==="
  python -u "'"$REPO"'/benchmarks/bench_moe_combine.py"
  echo "=== GPU correctness (fused_combine) ==="
  python -m pytest "'"$REPO"'/tests/test_moe_int4_w4a16.py" -q
'
