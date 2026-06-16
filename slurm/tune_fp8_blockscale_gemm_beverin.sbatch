#!/bin/bash
# SPDX-License-Identifier: MIT
# Sweep the CDNA3 autotune config space for the native fp8 MFMA fp8 block-scale
# GEMM (issue #41) across the V4 MLA shapes; bake the winners into
# configs.get_fp8_gemm_config. beverin (gfx942 / MI300A).
#
#   sbatch --export=ALL,REPO=$PWD slurm/tune_fp8_blockscale_gemm_beverin.sbatch
#
#SBATCH --job-name=xk-fp8-tune
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=fp8-tune-%j.out
#SBATCH --error=fp8-tune-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels-issue-41}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  cd "'"$REPO"'"
  python -u benchmarks/tune_fp8_blockscale_gemm.py
'
