#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness + perf for the MXFP4 fused-MoE GEMM (issue #43) on
# beverin (gfx942 / MI300A): the Triton grouped GEMM (E2M1 + ue8m0 inline dequant,
# fused clamped-SwiGLU) vs the pure-torch per-expert dequant loop, real bf16 A on
# the GPU (TRITON_INTERPRET unset so Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-43 \
#          slurm/test_moe_mxfp4_beverin.sbatch
#
#SBATCH --job-name=xk-moe-mxfp4
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=moe-mxfp4-%j.out
#SBATCH --error=moe-mxfp4-%j.out

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

  echo "######## pytest: MXFP4 fused-MoE GEMM on GPU (Triton bf16 vs torch oracle) ########"
  python -m pytest tests/test_moe_mxfp4.py -q

  echo "######## V4-Flash decode shape parity + max|err| + perf vs per-expert dequant loop ########"
  python -u - <<"PY"
import time
import torch
import torch.nn.functional as F
from xkernels import fused_moe_mxfp4
from xkernels._backends import Backend
from xkernels.ops.moe.mxfp4 import make_mxfp4_moe_weights, dequant_mxfp4_weight

torch.manual_seed(0)
dev = "cuda"
# DeepSeek-V4-Flash decode-ish: 48 tokens, 256 experts top-6, hidden=4096, ispp=512 (TP=4).
M, E, hidden, ispp, top_k, L = 48, 256, 4096, 512, 6, 10.0
w = make_mxfp4_moe_weights(E, hidden, ispp, group_size=32, with_bias=True, device=dev, seed=1)
A = (torch.randn(M, hidden, device=dev) * 0.1).to(torch.bfloat16)
topk_ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)


def torch_loop():
    out = torch.zeros(M, hidden, dtype=torch.float32, device=dev)
    flat = topk_ids.reshape(-1)
    tok = torch.arange(M, device=dev).repeat_interleave(top_k)
    fw = topk_w.reshape(-1).float()
    for e in torch.unique(flat).tolist():
        sel = flat == e
        t = tok[sel]; wt = fw[sel]
        xe = A[t]
        w13e = dequant_mxfp4_weight(w["w13"][e], w["w13_scale"][e], 32)
        gu = xe @ w13e.T + w["b13"][e]
        g, u = gu.float().chunk(2, -1)
        g = torch.clamp(g, max=L); u = torch.clamp(u, -L, L)
        act = (F.silu(g) * u).to(torch.bfloat16)
        w2e = dequant_mxfp4_weight(w["w2"][e], w["w2_scale"][e], 32)
        d = (act @ w2e.T).float() + w["b2"][e].float()
        out.index_add_(0, t, d * wt.unsqueeze(-1))
    return out


def triton_path():
    return fused_moe_mxfp4(
        A, w["w13"], w["w13_scale"], w["w2"], w["w2_scale"], topk_ids, topk_w,
        b13=w["b13"], b2=w["b2"], swiglu_limit=L, group_size=32, backend=Backend.TRITON,
    ).float()


ref = torch_loop()
got = triton_path()
err = (got - ref).abs().max().item()
rel = err / ref.abs().max().clamp_min(1e-6).item()
print(f"[V4 decode M={M} E={E} hidden={hidden} ispp={ispp} top_k={top_k}] max|err|={err:.4e} rel={rel:.4e}")
assert rel < 2e-2 or err < 2e-2, (err, rel)


def bench(fn, iters=20):
    torch.cuda.synchronize(); fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


t_loop = bench(torch_loop)
t_trit = bench(triton_path)
print(f"per-expert torch dequant loop : {t_loop:8.3f} ms")
print(f"xkernels MXFP4 Triton GEMM    : {t_trit:8.3f} ms  ({t_loop / t_trit:.2f}x)")
print("PASS: MXFP4 fused-MoE GEMM correct on gfx942")
PY
'
