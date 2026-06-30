#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU consolidated kernel benchmark on bristen (A100 / sm_80). Runs
# meta/benchmarks/bench_all.py: each op's optimized backend vs naive PyTorch, inside
# the NGC PyTorch container (torch + triton).
#
#   sbatch scripts/slurm/bench_all_bristen.sbatch
#
#SBATCH --job-name=xk-bench-all
#SBATCH --account=a-infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=bench-all-%j.out
#SBATCH --error=bench-all-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"

echo "REPO=$REPO IMAGE=$IMAGE node=$(hostname)"

srun --container-image="docker://$IMAGE" \
     --container-mounts=/capstor:/capstor,/iopsstor:/iopsstor \
     bash -c '
  set -e
  cd "'"$REPO"'"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/meta/benchmarks/bench_all.py"
'
