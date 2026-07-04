#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase D criterion #3 (issue #75): the H1/H2 named-edit-frequency count.

Measures, for one op (GEMM bf16 here) on the current GPU, the three numbers that
feed the H1/H2 verdict:

  * CEILING  — the vendor BLAS baseline (``torch.matmul``: cuBLAS on NVIDIA,
               rocBLAS/hipBLASLt on AMD). The honest perf bar (AGENTS.md §10).
  * H2       — the reliable named-edit regime: the DSL Triton card + its declared
               ``@triton.autotune`` configs (BLOCK_M/N/K, num_warps, num_stages),
               which ARE the closed-enum ``specialization_knobs`` an agent sweeps
               via ``SetKnob``/``Retile``/``SetMapPolicy``. ``verify(...,measure_perf=True)``.
  * H1       — the freehand regime: the native ``@gemm_bf16.target(backend, arch)``
               override body (``lower/cuda.py`` on NVIDIA, ``lower/hip.py`` on AMD).
               Currently FMA mechanism-validation, NOT a ceiling push — its role
               here is to show the H1 path RUNS + measures, framing where a
               matrix-engine (wgmma/MFMA) override would land.

Verdict rule: H2 ≥ 70% of the BLAS ceiling -> ``H2-ACHIEVABLE`` (the reliable
named-edit regime reaches the bar; no freehand override needed). Below 70% ->
``H1 NEEDED`` (closing the gap needs either extending the declared knob space to
the matrix-engine shapes — still H2, but a config-space change — or a native
matrix-core override body — H1).

H1/H2 definitions are recovered from the issue-#75 body (the cited
``docs/brainstorm/09-agent-editable-ir.md`` RFC is deleted in-tree).

Run on a GPU node:
  # beverin (MI300A / gfx942)
  scripts/cluster.sh run --host beverin srun --environment=tokenspeed-rocm-aiter-myofi \\
    --partition=mi300 --gpus-per-node=1 --time=00:20:00 \\
    bash -c 'cd $REPO && PYTHONPATH=src python3 scripts/h1h2_count_gemm.py'
  # ds5 (GB10 / sm_121)
  rcc --profile ds5 run --docker -s 'cd /workspace && python scripts/h1h2_count_gemm.py'

First data point (GEMM bf16, 4096^3) — see issue #75 for the writeup:
  amd_cdna3   BLAS 321.7 TF | H2 173.0 TF (54%) | H1 FMA 7.1 TF  -> H1 NEEDED
  nvidia_sm121 BLAS  88.2 TF | H2  39.5 TF (45%) | H1 FMA 1.5 TF  -> H1 NEEDED
The remaining ~2 ops (rmsnorm, quant) + the sm_90 row are follow-up.
"""
from __future__ import annotations

import torch

from xkernels import verify
from xkernels.utils.benchmarking import benchmark
from xkernels.vkl import spec_of, register_dsl
from xkernels.vkl.examples.gemm_bf16 import gemm_bf16

M = N = K = 4096          # compute-bound big shape for ceiling + H2
M_H1 = N_H1 = K_H1 = 512  # smaller for the FMA native override (avoid long runs)
FLOPS = 2 * M * N * K
FLOPS_H1 = 2 * M_H1 * N_H1 * K_H1
DEV = "cuda"

IS_AMD = bool(getattr(torch.version, "hip", None))
ARCH = "amd_cdna3" if IS_AMD else "nvidia_sm121"
VENDOR = "amd" if IS_AMD else "nvidia"


def tflops(ms: float, flops: int) -> float:
    return flops / (ms * 1e-3) / 1e12 if ms else float("nan")


def main() -> None:
    spec = spec_of(gemm_bf16)
    register_dsl(spec, backend="triton")

    print(f"\n=== GEMM bf16 H1/H2 data point ({VENDOR} / {ARCH}) ===")
    print(f"dev0: {torch.cuda.get_device_name(0)}")
    print(f"shape (ceiling/H2): {M}x{N}x{K}  ({FLOPS/1e9:.1f} GFLOP)")
    print(f"shape (H1 native):  {M_H1}x{N_H1}x{K_H1}")

    # CEILING: vendor BLAS via torch.matmul.
    a = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    ms_blas = benchmark(lambda: torch.matmul(a, b))
    t_blas = tflops(ms_blas, FLOPS)
    print(f"\nCEILING (BLAS torch.matmul): ms={ms_blas:.4f}  tflops={t_blas:.1f}")

    # H2: Triton + its @triton.autotune over the declared specialization_knobs.
    r2 = verify("gemm_bf16.triton@1.0.0", arch=ARCH,
                shapes=[{"dtype": "bf16", "M": M, "N": N, "K": K}], measure_perf=True)
    ms_h2 = r2["perf"]["ms"]
    t_h2 = tflops(ms_h2, FLOPS)
    print(f"H2 (Triton autotune):       ms={ms_h2:.4f}  tflops={t_h2:.1f}  "
          f"correct={r2['correctness']['passed']}")
    pct_h2 = 100 * t_h2 / t_blas if t_blas else float("nan")
    print(f"   -> H2 reaches {pct_h2:.0f}% of BLAS ceiling")

    # H1: native freehand override (hip on AMD, cuda on NVIDIA) — FMA baseline.
    if IS_AMD:
        from xkernels.vkl import register_dsl_hip
        register_dsl_hip(spec, spec.override_for("hip", "amd_cdna3"))
        card = "gemm_bf16.hip@1.0.0"
    else:
        from xkernels.vkl import register_dsl_cuda
        register_dsl_cuda(spec, spec.override_for("cuda", "nvidia_sm121"))
        card = "gemm_bf16.cuda@1.0.0"
    r1 = verify(card, arch=ARCH,
                shapes=[{"dtype": "bf16", "M": M_H1, "N": N_H1, "K": K_H1}], measure_perf=True)
    ms_h1 = r1["perf"]["ms"]
    t_h1 = tflops(ms_h1, FLOPS_H1)
    print(f"H1 (native {VENDOR} FMA):      ms={ms_h1:.4f}  tflops={t_h1:.2f}  "
          f"correct={r1['correctness']['passed']}  [mechanism-validation, NOT a ceiling push]")

    verdict = ("H2-ACHIEVABLE (named-edit regime suffices; no freehand H1 needed)"
               if pct_h2 >= 70 else "H1 NEEDED (H2 short of ceiling)")
    print(f"\nVERDICT (GEMM bf16, {ARCH}): H2={pct_h2:.0f}% of BLAS ceiling -> {verdict}")


if __name__ == "__main__":
    main()
