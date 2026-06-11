#!/bin/bash
# SPDX-License-Identifier: MIT
# Validate the Triton mxfp4_paged_gather (DeepSeek-V4 DSA indexer, issue #27 /
# tracking #28) on beverin (gfx942 / MI300A): GPU bf16 correctness vs the torch
# oracle + a tiny decode-shape timing.
#
#   sbatch slurm/probe_mxfp4_gather_beverin.sbatch
#
#SBATCH --job-name=xk-mxfp4-gather
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=mxfp4-gather-%j.out
#SBATCH --error=mxfp4-gather-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"

  echo "=== GPU bf16 correctness (Triton vs torch oracle) ==="
  python -m pytest "'"$REPO"'/tests/test_mxfp4_paged_gather.py" -q

  echo "=== decode-shape timing (num_seqs=64, topk=512, head_dim=128) ==="
  python -u "'"$REPO"'/benchmarks/bench_mxfp4_gather.py"
'
