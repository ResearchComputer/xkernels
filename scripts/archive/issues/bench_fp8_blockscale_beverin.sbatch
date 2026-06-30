#!/bin/bash
# SPDX-License-Identifier: MIT
# Single-GPU fp8 block-scale GEMM bench on beverin (gfx942 / MI300A). Runs
# meta/benchmarks/bench_fp8_blockscale_gemm.py: native fp8 MFMA (e4m3fnuz) vs the
# portable dequant path vs the torch dequant reference, across the V4 MLA shapes.
# gfx942-only: the fnuz fp8 MFMA encoding is an AMD CDNA3 primitive.
#
#   sbatch scripts/slurm/bench_fp8_blockscale_beverin.sbatch
#
#SBATCH --job-name=xk-bench-fp8
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=bench-fp8-%j.out
#SBATCH --error=bench-fp8-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  unset TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  cd "'"$REPO"'"
  python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, \"dev\", torch.cuda.get_device_name(0))"
  python -u "'"$REPO"'/meta/benchmarks/bench_fp8_blockscale_gemm.py"
'
