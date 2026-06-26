#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU Nsight Systems (system timeline) profile on bristen (A100 / sm_80).
# Runs scripts/profile-nsys-bristen.sh inside the NGC PyTorch container.
#
#   sbatch slurm/profile_nsys_bristen.sbatch
#   KERNEL=moe_sum_reduce sbatch slurm/profile_nsys_bristen.sbatch
#
#SBATCH --job-name=xk-nsys
#SBATCH --account=a-infra02
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=nsys-%j.out
#SBATCH --error=nsys-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
KERNEL="${KERNEL:-dual_rmsnorm}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"

echo "REPO=$REPO IMAGE=$IMAGE KERNEL=$KERNEL node=$(hostname)"

srun --container-image="docker://$IMAGE" \
     --container-mounts=/capstor:/capstor,/iopsstor:/iopsstor \
     bash -c '
  set -e
  cd "'"$REPO"'"
  bash scripts/profile-nsys-bristen.sh "'"$KERNEL"'"
'
