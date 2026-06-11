#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 DSA indexer logits (issue #27) on
# beverin (gfx942 / MI300A): run the Triton kernel vs the torch oracle with real
# bf16 q/k on the GPU (TRITON_INTERPRET unset so Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-27 \
#          slurm/test_dsa_indexer_beverin.sbatch
#
#SBATCH --job-name=xk-dsa-indexer
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=dsa-indexer-%j.out
#SBATCH --error=dsa-indexer-%j.out

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

  echo "######## pytest: DSA indexer logits on GPU (Triton bf16 vs oracle) ########"
  python -m pytest tests/test_dsa_indexer_logits.py -q

  echo "######## standalone bf16 max|err| + V4-shape top-k parity ########"
  python -u - <<"PY"
import torch
from xkernels import dsa_indexer_logits, dsa_indexer_topk
from xkernels._backends import Backend
from xkernels.ops.attention.dsa_reference import dsa_indexer_logits_ref

torch.manual_seed(0)
dev = "cuda"
# Real V4 indexer shape: H=index_n_heads=64, D=index_head_dim=128; Flash topk=512.
T, H, D, K, topk = 8, 64, 128, 4096, 512
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(K, D, device=dev, dtype=torch.bfloat16)
w = (torch.rand(T, H, device=dev) + 0.1)

got = dsa_indexer_logits(q, k, w, backend=Backend.TRITON)
ref = dsa_indexer_logits_ref(q, k, w)
err = (got - ref).abs().max().item()
rel = err / ref.abs().max().clamp_min(1e-6).item()
print(f"[V4 H=64 D=128 K=4096] max|err|={err:.4e}  rel={rel:.4e}")
assert err < 2e-1 and rel < 2e-2, (err, rel)

# Flash top-512 selection parity (set equality; ties on bf16 noise tolerated).
gi = [set(r.tolist()) for r in dsa_indexer_topk(got, topk).cpu()]
ri = [set(r.tolist()) for r in dsa_indexer_topk(ref, topk).cpu()]
jac = sum(len(a & b) for a, b in zip(gi, ri)) / (T * topk)
print(f"[topk=512] mean Jaccard(top-k set) = {jac:.4f}")
assert jac > 0.99, jac
print("PASS: DSA indexer logits + top-512 selection correct on gfx942")
PY
'
