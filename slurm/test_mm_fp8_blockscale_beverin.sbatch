#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 fp8 block-scale dense GEMM (issue #38)
# on beverin (gfx942 / MI300A): Triton MFMA GEMM vs the fp32 dequant oracle, real
# fp8 e4m3 operands on the GPU (TRITON_INTERPRET unset so Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/xkernels-issue-38 \
#          slurm/test_mm_fp8_blockscale_beverin.sbatch
#
#SBATCH --job-name=xk-fp8-gemm
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=fp8-gemm-%j.out
#SBATCH --error=fp8-gemm-%j.out

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

  echo "######## pytest: fp8 block-scale GEMM on GPU (Triton vs oracle) ########"
  python -m pytest tests/test_mm_fp8_blockscale.py -q

  echo "######## V4 MLA-projection shape parity + max|err| + perf ########"
  python -u - <<"PY"
import torch
from xkernels.ops.gemm import (
    mm_fp8_blockscale,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)
from xkernels.ops.gemm.reference import mm_fp8_blockscale_ref
from xkernels._backends import Backend

torch.manual_seed(0)
dev = "cuda"
block = 128
# V4-Flash MLA: hidden K=7168, kv_b-ish N=512, prefill-ish M.
for (M, N, K) in [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]:
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    w = torch.randn(N, K, device=dev, dtype=torch.float32)
    a_fp8, a_s = per_token_group_quant_fp8(a, block=block)
    w_fp8, w_s = per_block_quant_fp8(w, block=block)
    ref = mm_fp8_blockscale_ref(a_fp8, a_s, w_fp8, w_s, block=block, out_dtype=torch.float32)
    got = mm_fp8_blockscale(a_fp8, a_s, w_fp8, w_s, block=block,
                            out_dtype=torch.float32, backend=Backend.TRITON)
    err = (got - ref).abs().max().item()
    rel = err / ref.abs().max().clamp_min(1e-6).item()
    print(f"[M={M} N={N} K={K}] max|err|={err:.4e} rel={rel:.4e}")
    assert rel < 2e-2, (M, N, K, err, rel)

print("PASS: fp8 block-scale GEMM correct on gfx942")

# Perf: Triton MFMA fp8 GEMM (bf16-dot and fp32-dot) vs the slow
# torch_mm_fp8_blockscale reference path, across the V4 MLA shapes.
try:
    from triton.testing import do_bench
    for (M, N, K) in [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168),
                      (4096, 7168, 2048)]:
        a = torch.randn(M, K, device=dev, dtype=torch.float32)
        w = torch.randn(N, K, device=dev, dtype=torch.float32)
        a_fp8, a_s = per_token_group_quant_fp8(a, block=block)
        w_fp8, w_s = per_block_quant_fp8(w, block=block)
        t_bf16 = do_bench(lambda: mm_fp8_blockscale(a_fp8, a_s, w_fp8, w_s, block=block,
                          out_dtype=torch.bfloat16, dot_bf16=True, backend=Backend.TRITON))
        t_fp32 = do_bench(lambda: mm_fp8_blockscale(a_fp8, a_s, w_fp8, w_s, block=block,
                          out_dtype=torch.bfloat16, dot_bf16=False, backend=Backend.TRITON))
        t_ref = do_bench(lambda: mm_fp8_blockscale_ref(a_fp8, a_s, w_fp8, w_s, block=block,
                         out_dtype=torch.bfloat16))
        flops = 2 * M * N * K
        print(f"[perf M={M} N={N} K={K}] "
              f"triton_bf16={t_bf16:.3f}ms ({flops/t_bf16/1e9:.1f} TFLOP/s) "
              f"triton_fp32={t_fp32:.3f}ms "
              f"torch_ref={t_ref:.3f}ms -> bf16 {t_ref/t_bf16:.2f}x ref")
except Exception as e:  # noqa: BLE001
    print("perf bench skipped:", e)
PY
'
