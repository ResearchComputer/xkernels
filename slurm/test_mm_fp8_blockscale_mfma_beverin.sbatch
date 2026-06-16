#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device validation for the native fp8 MFMA fp8 block-scale GEMM (issue #41)
# on beverin (gfx942 / MI300A, TRITON_INTERPRET unset so Triton really compiles):
# pytest + V4-shape parity + perf (mfma-fnuz vs mfma-fn vs portable vs torch_ref)
# + an AMDGCN assertion that the real kernel emits v_mfma_*_fp8 on fnuz operands.
#
#   sbatch --export=ALL,REPO=$PWD slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch
#
#SBATCH --job-name=xk-fp8-mfma
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=fp8-mfma-%j.out
#SBATCH --error=fp8-mfma-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels-issue-41}"
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

  echo "######## pytest: native fp8 MFMA GEMM on GPU ########"
  python -m pytest tests/test_mm_fp8_blockscale_mfma.py -q

  echo "######## V4 MLA parity (fnuz, native fp8 MFMA) + perf vs torch_ref / portable ########"
  python -u - <<"PY"
import torch
from xkernels.ops.gemm import mm_fp8_blockscale, per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.ops.gemm.reference import mm_fp8_blockscale_ref
from xkernels._backends import Backend
from triton.testing import do_bench

torch.manual_seed(0); dev = "cuda"; block = 128
SHAPES = [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]
FNUZ = torch.float8_e4m3fnuz  # native fp8 MFMA on gfx942 (probe: 374 TFLOP/s)
FN = torch.float8_e4m3fn      # falls back to an f16 MFMA (probe: ~30 TFLOP/s)

def quant(M, N, K, fp8):
    a = torch.randn(M, K, device=dev); w = torch.randn(N, K, device=dev)
    a8, as_ = per_token_group_quant_fp8(a, block=block, fp8_dtype=fp8)
    w8, ws_ = per_block_quant_fp8(w, block=block, fp8_dtype=fp8)
    return a8, as_, w8, ws_

print("# parity: native fp8 MFMA (fnuz) vs fp32 dequant oracle")
for (M, N, K) in SHAPES:
    a8, as_, w8, ws_ = quant(M, N, K, FNUZ)
    ref = mm_fp8_blockscale_ref(a8, as_, w8, ws_, block=block, out_dtype=torch.float32)
    got = mm_fp8_blockscale(a8, as_, w8, ws_, block=block, out_dtype=torch.float32,
                            path="mfma", backend=Backend.TRITON)
    rel = (got - ref).abs().max().item() / ref.abs().max().clamp_min(1e-6).item()
    print(f"[M={M} N={N} K={K}] mfma(fnuz) rel={rel:.3e}")
    assert rel < 5e-3, (M, N, K, rel)
print("PASS: native fp8 MFMA correct on gfx942")

print("# perf (bf16 out): mfma(fnuz) vs mfma(fn,f16) vs portable vs torch_ref")
for (M, N, K) in SHAPES:
    az, asz, wz, wsz = quant(M, N, K, FNUZ)
    an, asn, wn, wsn = quant(M, N, K, FN)
    t_mfz = do_bench(lambda: mm_fp8_blockscale(az, asz, wz, wsz, block=block, out_dtype=torch.bfloat16, path="mfma", backend=Backend.TRITON))
    t_mfn = do_bench(lambda: mm_fp8_blockscale(an, asn, wn, wsn, block=block, out_dtype=torch.bfloat16, path="mfma", backend=Backend.TRITON))
    t_port = do_bench(lambda: mm_fp8_blockscale(an, asn, wn, wsn, block=block, out_dtype=torch.bfloat16, path="portable", backend=Backend.TRITON))
    t_ref = do_bench(lambda: mm_fp8_blockscale_ref(an, asn, wn, wsn, block=block, out_dtype=torch.bfloat16))
    fl = 2 * M * N * K
    print(f"[perf M={M} N={N} K={K}] mfma_fnuz={t_mfz:.3f}ms ({fl/t_mfz/1e9:.1f}TF) "
          f"mfma_fn={t_mfn:.3f}ms portable={t_port:.3f}ms torch_ref={t_ref:.3f}ms "
          f"-> fnuz {t_ref/t_mfz:.2f}x ref")
PY

  echo "######## AMDGCN: real mfma kernel must emit v_mfma_*_fp8 on fnuz operands ########"
  AMDGCN_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python - <<"PY" 2>&1 | grep -oE "v_mfma_[a-z0-9_]+" | sort | uniq -c
import torch
from xkernels.ops.gemm import per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton
# small-N -> 16x16x32 fp8 MFMA config; large-N prefill -> 32x32x16 fp8 MFMA config.
for M, N in ((8, 512), (256, 2048)):
    a = torch.randn(M, 4096, device="cuda"); w = torch.randn(N, 4096, device="cuda")
    a8, as_ = per_token_group_quant_fp8(a, block=128, fp8_dtype=torch.float8_e4m3fnuz)
    w8, ws_ = per_block_quant_fp8(w, block=128, fp8_dtype=torch.float8_e4m3fnuz)
    mm_fp8_blockscale_mfma_triton(a8, as_, w8, ws_, block=128)
PY
  echo "(expect v_mfma_f32_16x16x32_fp8_fp8 AND v_mfma_f32_32x32x16_fp8_fp8; absence => fell back to f16)"
'
