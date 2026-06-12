#!/bin/bash
# SPDX-License-Identifier: MIT
# Phase-0 probe (issue #41): does tl.dot on fp8 e4m3 reach native fp8 MFMA on
# gfx942 (MI300A)? Compiles a minimal fp8 dot for float8_e4m3fn AND
# float8_e4m3fnuz, greps the AMDGCN for v_mfma_*_fp8*, and reports parity +
# TFLOP/s per format. Resolves the fn-vs-fnuz unknown before trusting perf.
#
#   sbatch --export=ALL,REPO=$PWD slurm/probe_fp8_mfma_beverin.sbatch
#
#SBATCH --job-name=xk-fp8-probe
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=fp8-probe-%j.out
#SBATCH --error=fp8-probe-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels-issue-41}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  export TRITON_ALWAYS_COMPILE=1
  cd "'"$REPO"'"

  python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, torch.cuda.get_device_name(0))"

  python -u benchmarks/probe_fp8_mfma.py
'
