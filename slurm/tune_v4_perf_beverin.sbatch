#!/bin/bash
# SPDX-License-Identifier: MIT
# Issue #39 perf pass for the V4 sparse-MLA (#33) + MHC prenorm GEMM (#37) kernels
# on beverin (gfx942 / MI300A). Runs, with TRITON_INTERPRET unset so Triton really
# compiles:
#   1. the offline correctness suites for both kernels + the #39 perf-knob
#      invariance tests (must still pass at bf16 on the GPU),
#   2. a config sweep for each kernel (benchmarks/tune_*.py) that reports median
#      ms + speedup vs the #33 / #36 baseline and vs the naive torch path, with a
#      per-config correctness check.
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-39 \
#          slurm/tune_v4_perf_beverin.sbatch
#
#SBATCH --job-name=xk-v4-perf
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=v4-perf-%j.out
#SBATCH --error=v4-perf-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels-issue-39}"
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

  echo "######## pytest: kernel suites + #39 perf-knob invariance (Triton bf16) ########"
  python -m pytest tests/test_sparse_mla_attention.py tests/test_mhc_prenorm_gemm.py tests/test_issue39_perf_pass.py -q

  echo "######## sweep: MHC prenorm GEMM config space ########"
  python -u benchmarks/tune_mhc_prenorm_gemm.py

  echo "######## sweep: sparse-MLA attention config space ########"
  python -u benchmarks/tune_sparse_mla.py

  echo "ALL DONE"
'
