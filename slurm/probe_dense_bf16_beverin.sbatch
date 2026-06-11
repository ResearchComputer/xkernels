#!/bin/bash
# SPDX-License-Identifier: MIT
# Characterize bf16 dense-GEMM MFMA/hipBLASLt routing on beverin (gfx942 /
# MI300A): run benchmarks/probe_ffn.py across all blas modes (issue #17).
#
#   sbatch slurm/probe_dense_bf16_beverin.sbatch
#
#SBATCH --job-name=xk-probe-bf16
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:30:00
#SBATCH --output=probe-bf16-%j.out
#SBATCH --error=probe-bf16-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  PROBE="'"$REPO"'/benchmarks/probe_ffn.py"

  echo "######## mode=default ########"
  python -u "$PROBE" --mode default

  echo "######## mode=hipblaslt ########"
  TORCH_BLAS_PREFER_HIPBLASLT=1 python -u "$PROBE" --mode hipblaslt

  echo "######## mode=no-hipblaslt ########"
  TORCH_BLAS_PREFER_HIPBLASLT=0 python -u "$PROBE" --mode no-hipblaslt

  echo "######## mode=tunableop ########"
  # Cap per-GEMM tuning so the sweep cannot blow the wall clock; tunableop tunes
  # every unique GEMM on first sight.
  PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 \
    PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=30 \
    python -u "$PROBE" --mode tunableop
'
