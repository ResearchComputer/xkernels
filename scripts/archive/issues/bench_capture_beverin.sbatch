#!/bin/bash
# SPDX-License-Identifier: MIT
# HIP-graph-captured flat-vs-hierarchical all-reduce on beverin (issue #12).
# Drives collectives through a watchdog-free raw RCCL comm (meta/benchmarks/pynccl_lite.py)
# so the schedule is capturable, plus the OFI capture fix (FI_CXI_OPTIMIZED_MRS=0).
#
#   sbatch --nodes=2 scripts/slurm/bench_capture_beverin.sbatch
#   SIZES="1 8 64 512 2048" sbatch --nodes=2 scripts/slurm/bench_capture_beverin.sbatch
#
#SBATCH --job-name=xk-capture
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=capture-%j.out
#SBATCH --error=capture-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"
OFI_PLUGIN_DIR="${OFI_PLUGIN_DIR:-/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib}"
SIZES="${SIZES:-1 2 4 8 16}"
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-30}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29601}"

echo "REPO=$REPO ENV=$ENV_NAME MASTER_ADDR=$MASTER_ADDR nodes=$SLURM_NNODES tasks=$SLURM_NTASKS SIZES=[$SIZES]"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="'"$OFI_PLUGIN_DIR"':/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/benchmarks:'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/meta/benchmarks/bench_capture_pynccl.py" \
      --ranks-per-node 4 --hidden 7168 --sizes '"$SIZES"' \
      --iters '"$ITERS"' --warmup '"$WARMUP"'
'
