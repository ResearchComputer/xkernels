#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 sparse-MLA attention compute
# (issue #32) on beverin (gfx942 / MI300A): run the Triton kernel vs the torch
# oracle with real bf16 q/kv on the GPU (TRITON_INTERPRET unset so Triton
# actually compiles), plus the fp8_ds_mla decode path.
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-32 \
#          scripts/slurm/test_sparse_mla_beverin.sbatch
#
#SBATCH --job-name=xk-sparse-mla
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=sparse-mla-%j.out
#SBATCH --error=sparse-mla-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
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

  echo "######## pytest: sparse-MLA on GPU (Triton bf16 vs oracle) ########"
  python -m pytest tests/test_sparse_mla_attention.py -q

  echo "######## V4-shape prefill parity + max|err| ########"
  python -u - <<"PY"
import torch
from xkernels import sparse_mla_attention
from xkernels._backends import Backend
from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref

torch.manual_seed(0)
dev = "cuda"
# Real V4 prefill shape: H=128 heads, latent D=512 (d_v=448), Flash topk=512.
T, H, D, d_v, Kv, topk = 8, 128, 512, 448, 4096, 512
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
lens = torch.randint(1, topk + 1, (T,), device=dev, dtype=torch.int32)
sink = torch.randn(H, device=dev)

got = sparse_mla_attention(q, kv, idx, sm_scale=1 / D**0.5, topk_length=lens,
                           attn_sink=sink, d_v=d_v, backend=Backend.TRITON)
ref = sparse_mla_attention_ref(q, kv, idx, sm_scale=1 / D**0.5, topk_length=lens,
                               attn_sink=sink, d_v=d_v)
err = (got[0].float() - ref[0].float()).abs().max().item()
rel = err / ref[0].float().abs().max().clamp_min(1e-6).item()
print(f"[V4 prefill H=128 D=512 d_v=448 topk=512] max|err|={err:.4e} rel={rel:.4e}")
assert err < 5e-2, (err, rel)
print("PASS: sparse-MLA prefill compute correct on gfx942")
PY
'
