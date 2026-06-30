#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 MHC prenorm GEMM (issue #36) on
# beverin (gfx942 / MI300A): Triton split-K vs torch oracle, real bf16 A / fp32
# fn on the GPU (TRITON_INTERPRET unset so Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/xkernels-issue-36 \
#          scripts/slurm/test_mhc_prenorm_beverin.sbatch
#
#SBATCH --job-name=xk-mhc-prenorm
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=mhc-prenorm-%j.out
#SBATCH --error=mhc-prenorm-%j.out

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

  echo "######## pytest: MHC prenorm GEMM on GPU (Triton bf16/fp32 vs oracle) ########"
  python -m pytest tests/test_mhc_prenorm_gemm.py -q

  echo "######## V4-Flash MHC shape parity + max|err| ########"
  python -u - <<"PY"
import torch
import torch.nn.functional as F
from xkernels import hc_prenorm_gemm
from xkernels._backends import Backend

torch.manual_seed(0)
dev = "cuda"
# V4-Flash MHC: hc_mult=4, hidden=4096 -> K=16384, N=24. decode-ish T, 16 splits.
T, hc_mult, hidden = 8, 4, 4096
K, N, n_splits = hc_mult * hidden, 24, 16
a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
fn = torch.randn(N, K, device=dev, dtype=torch.float32)
mul, sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)
af = a.float()
fmul = F.linear(af, fn.float())
fsqr = (af * af).sum(-1)
emul = (mul.sum(0) - fmul).abs().max().item()
esqr = (sqr.sum(0) - fsqr).abs().max().item()
rmul = emul / fmul.abs().max().clamp_min(1e-6).item()
print(f"[V4 MHC T={T} K={K} N={N} splits={n_splits}] mul max|err|={emul:.4e} rel={rmul:.4e} | sqrsum max|err|={esqr:.4e}")
assert emul / fmul.abs().max().item() < 5e-2, (emul, rmul)
assert esqr / fsqr.abs().max().item() < 5e-2, esqr
print("PASS: MHC prenorm GEMM correct on gfx942")
PY
'
