#!/usr/bin/env bash
# TEMP (untracked) — Phase D criterion #3: GEMM H1/H2 data point on beverin/ds5.
# Measures, per arch: BLAS ceiling (torch.matmul), H2 (Triton autotune = the H2
# named-edit knob space), H1 (native freehand override). Verdict: H2-achievable?
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
echo "### host: $(hostname)"
python3 - <<'PY'
import torch
from xkernels import verify
from xkernels.utils.benchmarking import benchmark
from xkernels.vkl import spec_of, register_dsl
from xkernels.vkl.examples.gemm_bf16 import gemm_bf16

# arch detection: beverin is ROCm (amd_cdna3/gfx942), ds5 is GB10 (nvidia_sm121).
is_amd = bool(getattr(torch.version, "hip", None))
ARCH = "amd_cdna3" if is_amd else "nvidia_sm121"
VENDOR = "amd" if is_amd else "nvidia"

M = N = K = 4096          # compute-bound big shape for ceiling + H2
M_H1 = N_H1 = K_H1 = 512  # smaller for the FMA native override (avoid long runs)
FLOPS = 2 * M * N * K
FLOPS_H1 = 2 * M_H1 * N_H1 * K_H1
DEV = "cuda"

def tflops(ms, flops):
    return flops / (ms * 1e-3) / 1e12 if ms else float("nan")

print(f"\n=== GEMM bf16 H1/H2 data point ({VENDOR} / {ARCH}) ===")
print(f"dev0: {torch.cuda.get_device_name(0)}")
print(f"shape (ceiling/H2): {M}x{N}x{K}  ({FLOPS/1e9:.1f} GFLOP = {FLOPS/1e12:.3f} TFLOP)")
print(f"shape (H1 native):  {M_H1}x{N_H1}x{K_H1}")

# register the DSL triton backend (H2) + the native override (H1)
spec = spec_of(gemm_bf16)
register_dsl(spec, backend="triton")

# --- CEILING: vendor BLAS via torch.matmul (cuBLAS / rocBLAS) ---
a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
ms_blas = benchmark(lambda: torch.matmul(a, b))
t_blas = tflops(ms_blas, FLOPS)
print(f"\nCEILING (BLAS torch.matmul): ms={ms_blas:.4f}  tflops={t_blas:.1f}")

# --- H2: Triton + its @triton.autotune (the declared BLOCK_M/N/K,num_warps,num_stages space) ---
# Real GPU arch (not "any") so verify measures perf on the device.
r2 = verify("gemm_bf16.triton@1.0.0", arch=ARCH,
            shapes=[{"dtype":"bf16","M":M,"N":N,"K":K}], measure_perf=True)
ms_h2 = r2["perf"]["ms"]
t_h2 = tflops(ms_h2, FLOPS)
print(f"H2 (Triton autotune):       ms={ms_h2:.4f}  tflops={t_h2:.1f}  correct={r2['correctness']['passed']}")
pct_h2 = 100 * t_h2 / t_blas if t_blas else float("nan")
print(f"   -> H2 reaches {pct_h2:.0f}% of BLAS ceiling")

# --- H1: native freehand override (hip on AMD, cuda on NVIDIA) — FMA mechanism-validation ---
if is_amd:
    from xkernels.vkl import register_dsl_hip
    register_dsl_hip(spec, spec.override_for("hip","amd_cdna3"))
    card = "gemm_bf16.hip@1.0.0"
else:
    from xkernels.vkl import register_dsl_cuda
    register_dsl_cuda(spec, spec.override_for("cuda","nvidia_sm121"))
    card = "gemm_bf16.cuda@1.0.0"
r1 = verify(card, arch=ARCH, shapes=[{"dtype":"bf16","M":M_H1,"N":N_H1,"K":K_H1}], measure_perf=True)
ms_h1 = r1["perf"]["ms"]
t_h1 = tflops(ms_h1, FLOPS_H1)
print(f"H1 (native {VENDOR} FMA):      ms={ms_h1:.4f}  tflops={t_h1:.2f}  correct={r1['correctness']['passed']}  [mechanism-validation, NOT a ceiling push]")

verdict = "H2-ACHIEVABLE (named-edit regime suffices; no freehand H1 needed)" if pct_h2 >= 70 else "H1 NEEDED (H2 short of ceiling)"
print(f"\nVERDICT (GEMM bf16, {ARCH}): H2={pct_h2:.0f}% of BLAS ceiling -> {verdict}")
PY
echo "### DONE"
