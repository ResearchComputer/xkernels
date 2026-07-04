#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase D criterion #3 (issue #75): H1/H2 count — the BANDWIDTH-bound ops.

Companion to ``h1h2_count_gemm.py`` (the compute-bound row). The issue asks for
the H1/H2 verdict across ~3 ops (GEMM, rmsnorm, quant); the GEMM script settled
the compute-bound case (H2 caps at ~50% of the compute ceiling -> H1 needed).
This script settles the complementary case: **bandwidth-bound** ops, where the
honest ceiling is the DRAM roofline (not a TFLOP count, and not torch-eager --
which is unfused and therefore not a ceiling at all).

Metric regime difference (the load-bearing methodological point):
  * GEMM (compute-bound): ceiling = vendor BLAS (cuBLAS / hipBLASLt) in TFLOPS;
    verdict = H2_TFLOPS / BLAS_TFLOPS.
  * rmsnorm / quant (bandwidth-bound): ceiling = DRAM roofline (peak GB/s);
    verdict = achieved_BW / DRAM_peak_BW; achieved_BW = bytes_moved / ms.

H1 row for these ops is **NOT NEEDED by construction**: the repo ships no native
freehand override for rmsnorm or per_block_quant_fp8, and a native one would not
help -- they are bandwidth-bound, so the matrix engine is irrelevant; the win is
coalesced vectorized loads, which Triton already emits. So even where H2 falls
short of the roofline, the gap is H2 *tiling/coalescing* work (Retile / SetKnob),
NOT an H1 freehand override. That is the opposite H1 verdict from GEMM, and the
contrast IS the criterion-#3 deliverable.

Byte models (bf16):
  * rmsnorm            : read x[T,d] (2B) + write out[T,d] (2B) + read w[d] (4B)
                         ~ 4*T*d bytes  (the fp32 mean-square reduction is on-chip)
  * per_block_quant_fp8: read x[G,B] (2B) + write q[G,B] fp8 (1B) + write scale[G] (4B)
                         ~ 3*G*B bytes  (one scale per row, per the op spec)

DRAM rooflines (registry/cost_model.arch_peaks):
  amd_cdna3 5300 GB/s | nvidia_sm121 243 GB/s (GB10 is bandwidth-starved)

Data points (this script, 2026-07-04) -- the bandwidth-bound complement to the
GEMM compute-bound rows in h1h2_count_gemm.py. Combined criterion-#3 table:

  op            arch          H2 (Triton)   % of ceiling   H1 verdict
  ------------  ------------  ------------  -------------  ----------------------
  GEMM bf16     amd_cdna3     173.0 TF      54% of BLAS    H1 NEEDED (compute)
  GEMM bf16     nvidia_sm121   39.5 TF      45% of BLAS    H1 NEEDED (compute)
  rmsnorm       amd_cdna3     1606 GB/s     30% of DRAM    H1 not needed* (bw)
  rmsnorm       nvidia_sm121   230 GB/s     95% of DRAM    H1 not needed  (bw)
  per_blk_quant amd_cdna3     1408 GB/s     27% of DRAM    H1 not needed* (bw)
  per_blk_quant nvidia_sm121   240 GB/s     99% of DRAM    H1 not needed  (bw)

* MI300A rmsnorm/quant sit at ~28% of its very wide 5300 GB/s HBM3 roofline
  (H2-SHORT), but H1 -- a native matrix-core override -- is STILL not the fix:
  the gap is H2 tiling/coalescing/occupancy (the matrix engine is irrelevant to
  a bandwidth-bound op). The H1-vs-H2 distinction holds across both regimes:
  H1 (freehand native override) is needed ONLY for compute-bound ops where
  matrix-engine targeting escapes the declared specialization_knobs (GEMM); for
  bandwidth-bound ops H1 is categorically unnecessary. THAT is the honest scope
  of the agent-native (H2) claim -- the criterion-#3 deliverable.

Run on a GPU node:
  # beverin (MI300A / gfx942)
  scripts/cluster.sh run --host beverin \\
    srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \\
    --gpus-per-node=1 --time=00:15:00 \\
    bash -c 'cd $REPO && PYTHONPATH=src python3 scripts/h1h2_count_bw.py'
  # ds5 (GB10 / sm_121)
  rcc --profile ds5 run --docker -s \\
    'cd /workspace && PYTHONPATH=src python scripts/h1h2_count_bw.py'
"""
from __future__ import annotations

import torch

from xkernels import verify
from xkernels.registry.cost_model import arch_peaks
from xkernels.utils.benchmarking import benchmark
from xkernels.vkl import register_dsl, spec_of
from xkernels.vkl.examples.rmsnorm import rmsnorm

# Representative LLM shapes (bf16), large enough to be DRAM-bound not launch-bound.
T, D = 8192, 4096                  # rmsnorm: 8k tokens x 4k hidden
# per_block_quant: needs a LARGE shape to escape launch overhead (small B/G
# combos are kernel-edge or launch-bound; G=32768,B=4096 verifies + is DRAM-bound).
G_QUANT, B_QUANT = 32768, 4096
_RMS_BYTES = 2 * T * D + 2 * T * D + 4 * D        # x bf16 + out bf16 + w fp32
_Q_BYTES = 2 * G_QUANT * B_QUANT + G_QUANT * B_QUANT + 4 * G_QUANT  # x bf16 + q fp8 + scale fp32

DEV = "cuda"
IS_AMD = bool(getattr(torch.version, "hip", None))
ARCH = "amd_cdna3" if IS_AMD else "nvidia_sm121"
VENDOR = "amd" if IS_AMD else "nvidia"
DRAM_BW = arch_peaks(ARCH)["dram_bw_gbs"]         # peak GB/s


def gbs(ms: float, nbytes: int) -> float:
    """Achieved DRAM bandwidth (GB/s) for ``nbytes`` moved in ``ms``."""
    return nbytes / (ms * 1e-3) / 1e9 if ms else float("nan")


def _torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Torch eager baseline (CONTEXT only -- unfused, NOT a ceiling)."""
    xf = x.to(torch.float32)
    inv_rms = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * inv_rms).to(x.dtype) * w


def _torch_per_block_quant(
    x: torch.Tensor, fp8_dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch eager baseline (CONTEXT only). One scale per row over B, per the spec."""
    fp8_max = 240.0 if "fnuz" in str(fp8_dtype) else 448.0
    amax = x.to(torch.float32).abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / fp8_max
    q = (x.to(torch.float32) / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return q, scale.squeeze(-1)


def _verdict(op: str, pct: float) -> None:
    h2_status = ("H2-ACHIEVABLE (Triton at DRAM roofline)"
                 if pct >= 70 else f"H2-SHORT-of-roofline ({pct:.0f}%; H2 tiling work remains)")
    print(f"  VERDICT ({op}, {ARCH}): {h2_status}; "
          f"H1 NOT NEEDED -- bandwidth-bound, matrix engine irrelevant.")


def _run_rmsnorm() -> None:
    print(f"\n--- rmsnorm bf16  T={T} d={D}  ({_RMS_BYTES/1e6:.1f} MB moved) ---")
    x = torch.randn(T, D, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(D, device=DEV, dtype=torch.float32)

    ms_eager = benchmark(lambda: _torch_rmsnorm(x, w))
    print(f"  context (torch eager, unfused): ms={ms_eager:.4f}  "
          f"bw={gbs(ms_eager,_RMS_BYTES):.1f} GB/s")

    register_dsl(spec_of(rmsnorm), backend="triton")
    r = verify("rmsnorm.triton@1.0.0", arch=ARCH,
               shapes=[{"dtype": "bf16", "T": T, "d": D}], measure_perf=True)
    ms_h2 = r["perf"]["ms"]
    bw_h2 = gbs(ms_h2, _RMS_BYTES)
    pct = 100 * bw_h2 / DRAM_BW if DRAM_BW else float("nan")
    print(f"  H2 (Triton autotune):  ms={ms_h2:.4f}  bw={bw_h2:.1f} GB/s  "
          f"({pct:.0f}% of {DRAM_BW:.0f} GB/s roofline)  correct={r['correctness']['passed']}")
    print("  H1 (native freehand):  NOT NEEDED -- bandwidth-bound; matrix engine irrelevant.")
    _verdict("rmsnorm", pct)


def _run_quant() -> None:
    print(f"\n--- per_block_quant_fp8 bf16->fp8  G={G_QUANT} B={B_QUANT}  "
          f"({_Q_BYTES/1e6:.1f} MB moved) ---")
    fp8 = (torch.float8_e4m3fnuz
           if IS_AMD and hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn)
    x = torch.randn(G_QUANT, B_QUANT, device=DEV, dtype=torch.bfloat16)

    ms_eager = benchmark(lambda: _torch_per_block_quant(x, fp8))
    print(f"  context (torch eager, unfused): ms={ms_eager:.4f}  "
          f"bw={gbs(ms_eager,_Q_BYTES):.1f} GB/s")

    r = verify("per_block_quant_fp8.triton@1.0.0", arch=ARCH,
               shapes=[{"dtype": "bf16", "G": G_QUANT, "B": B_QUANT}], measure_perf=True)
    ms_h2 = r["perf"]["ms"]
    bw_h2 = gbs(ms_h2, _Q_BYTES)
    pct = 100 * bw_h2 / DRAM_BW if DRAM_BW else float("nan")
    print(f"  H2 (Triton autotune):  ms={ms_h2:.4f}  bw={bw_h2:.1f} GB/s  "
          f"({pct:.0f}% of {DRAM_BW:.0f} GB/s roofline)  correct={r['correctness']['passed']}")
    print("  H1 (native freehand):  NOT NEEDED -- bandwidth-bound; matrix engine irrelevant.")
    _verdict("per_block_quant_fp8", pct)


def main() -> None:
    print(f"=== H1/H2 bandwidth-bound data point ({VENDOR} / {ARCH}) ===")
    print(f"dev0: {torch.cuda.get_device_name(0)}   DRAM roofline: {DRAM_BW:.0f} GB/s")
    _run_rmsnorm()
    _run_quant()


if __name__ == "__main__":
    main()
