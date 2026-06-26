#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU consolidated kernel benchmark on bristen (A100 / sm_80), with
# PER-KERNEL PROCESS ISOLATION. Delegates the loop to
# scripts/bench_kernel_loop_bristen.sh (one `python` process per kernel, so a
# native Triton-3.0.0 OptimizeThreadLocality SIGSEGV only loses that one row).
#
#   sbatch slurm/bench_all_bristen_isolated.sbatch
#
#SBATCH --job-name=xk-bench1
#SBATCH --account=a-infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:25:00
#SBATCH --output=bench1-bristen-%j.out
#SBATCH --error=bench1-bristen-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"

echo "REPO=$REPO IMAGE=$IMAGE node=$(hostname)"

srun --container-image="docker://$IMAGE" \
     --container-mounts=/capstor:/capstor,/iopsstor:/iopsstor \
     bash -c '
  export PYTHONPATH="'"$REPO"'/src:'"$REPO"':${PYTHONPATH:-}"
  bash "'"$REPO"'/scripts/bench_kernel_loop_bristen.sh"
'
