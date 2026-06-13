#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 MHC mhc_pre / mhc_post full fusion
# (issue #44) on beverin (gfx942 / MI300A): the Triton port vs the torch oracle,
# with the layer_input (pre-weighted residual combine) defect branch front and
# centre. Real bf16 residual / fp32 weights on the GPU (TRITON_INTERPRET unset so
# Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-44 \
#          slurm/test_mhc_pre_post_beverin.sbatch
#
#SBATCH --job-name=xk-mhc-prepost
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=mhc-prepost-%j.out
#SBATCH --error=mhc-prepost-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels-issue-44}"
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

  echo "######## pytest: MHC mhc_pre / mhc_post on GPU (Triton bf16 vs oracle) ########"
  python -m pytest tests/test_mhc_pre_post.py -q

  echo "######## V4-Flash MHC pre/post parity + max|err| (layer_input defect branch) ########"
  python -u - <<"PY"
import torch
import torch.nn.functional as F
from xkernels import mhc_pre, mhc_post
from xkernels._backends import Backend

torch.manual_seed(0)
dev = "cuda"
# V4-Flash MHC: hc_mult=4, hidden=4096, sinkhorn_iters=20, rms/hc eps=1e-6.
HC_MULT, HIDDEN = 4, 4096
HC_MULT3 = 2 * HC_MULT + HC_MULT * HC_MULT  # 24
RMS_EPS = HC_EPS = 1e-6
ITERS = 20


def sinkhorn(m, it, eps):
    m = torch.softmax(m, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(it - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def pre_oracle(res, fn, sc, base):
    nt, hc, _ = res.shape
    x = res.flatten(1).float()
    rs = torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    mix = F.linear(x, fn.float()) * rs
    pr, po, cb = torch.split(mix, [hc, hc, hc * hc], dim=-1)
    pb, ob, cbb = torch.split(base.float(), [hc, hc, hc * hc], dim=-1)
    pre = torch.sigmoid(pr * sc[0].float() + pb) + HC_EPS
    post = (torch.sigmoid(po * sc[1].float() + ob) * 2.0).unsqueeze(-1)
    comb = sinkhorn(cb.reshape(nt, hc, hc) * sc[2].float() + cbb.reshape(1, hc, hc), ITERS, HC_EPS)
    li = torch.sum(pre.unsqueeze(-1) * res.float(), dim=1)
    return li.to(res.dtype), post, comb


def post_oracle(h, res, post, comb):
    mr = torch.einsum("tnm,tnh->tmh", comb.float(), res.float())
    bu = post.float() * h.float().unsqueeze(1)
    return (mr + bu).to(h.dtype)


def err(a, b):
    a, b = a.float(), b.float()
    ma = (a - b).abs().max().item()
    return ma, ma / (b.abs().max().item() + 1e-6)


fail = False
for nt in (8, 128):
    res = torch.randn(nt, HC_MULT, HIDDEN, device=dev, dtype=torch.bfloat16)
    fn = (torch.randn(HC_MULT3, HC_MULT * HIDDEN, device=dev) * 0.02).float()
    sc = torch.tensor([0.7, 1.1, 0.5], device=dev, dtype=torch.float32)
    base = torch.randn(HC_MULT3, device=dev, dtype=torch.float32)

    li_o, post_o, comb_o = pre_oracle(res, fn, sc, base)
    li, post, comb = mhc_pre(res, fn, sc, base, RMS_EPS, HC_EPS, ITERS, backend=Backend.TRITON)
    for name, g, r in (("pre.layer_input", li, li_o), ("pre.post", post, post_o), ("pre.comb", comb, comb_o)):
        ma, rel = err(g, r)
        ok = torch.isfinite(g).all().item() and rel < 2e-2
        fail |= not ok
        tag = "OK" if ok else "FAIL"
        print(f"[T={nt:4d}] {name:18s} max_abs={ma:.3e} rel={rel:.3e} {tag}", flush=True)

    h = torch.randn(nt, HIDDEN, device=dev, dtype=torch.bfloat16)
    out_o = post_oracle(h, res, post_o, comb_o)
    out = mhc_post(h, res, post_o, comb_o, backend=Backend.TRITON)
    ma, rel = err(out, out_o)
    ok = torch.isfinite(out).all().item() and rel < 2e-2
    fail |= not ok
    tag = "OK" if ok else "FAIL"
    pname = "post.out"
    print(f"[T={nt:4d}] {pname:18s} max_abs={ma:.3e} rel={rel:.3e} {tag}", flush=True)

print("MHC_PREPOST_FAIL" if fail else "MHC_PREPOST_OK", flush=True)
import sys; sys.exit(1 if fail else 0)
PY
'
