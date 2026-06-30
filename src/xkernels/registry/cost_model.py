# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Analytical FLOP / byte cost models per op — turns ``verify()``'s perf block
from ``ms``-only into a real roofline signal.

These are DERIVED analytics, not contract: the Op Spec is the source of truth
for shapes/dtypes/constraints (the *contract*), and a cost model is the
arithmetic a domain author writes once per op to answer "how many flops / how
many bytes for this point". They live in code (here, next to the input
generators in ``input_gen.py`` — same author who knows the exact byte counts)
rather than in the JSON spec, so the contract stays free of perf plumbing.

Each ``cost_model(op_id, point)`` returns ``(flops, bytes)`` for ONE sweep
point, using the same dtype-aware byte arithmetic the roofline survey
(``scripts/ds5_roofline_survey.py``) validated against measured GB10 numbers.
``_measure_perf`` divides flops/bytes by the measured ``ms`` to fill
``tflops`` and ``achieved_bw_pct`` against the target arch's peak ceiling.

Registering a new op here is OPTIONAL — ops without a model still get ``ms``
(they just leave the derived metrics None, as before). The point is that the 5
ops with cards now give an agent an in-harness memory-vs-compute diagnosis
without leaving for an external profiler.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .dtypes import dtype_bytes

# --- arch peak ceilings -------------------------------------------------------
# Peak fp32 CUDA-core / scalar FLOPS (TFLOPS) and DRAM copy BW (GB/s, R+W).
# These are the ROOFLINE ceilings (the realistic kernel bounds), not marketing
# peaks. GB10 is MEASURED (scripts/ds5_roofline_survey.py); the rest are the
# documented scalar ceilings the library targets. Tensor-core ceilings are op-
# specific (dtype + MMA shape) and intentionally NOT collapsed into one number
# here — a cost model that needs them declares its own peak.
_ARCH_PEAKS: dict[str, dict[str, float]] = {
    # GB10: 48 SMs * 128 fp32 cores * 2.4 GHz * 2 (FMA) = 29.5 TFLOPS; BW measured.
    "nvidia_sm121": {"fp32_tflops": 29.5, "dram_bw_gbs": 243.0},
    # A100 (sm_80): 108 SM * 64 cores * 1.41GHz * 2 ≈ 19.5 TF; ~1550 GB/s HBM2.
    "nvidia_sm80": {"fp32_tflops": 19.5, "dram_bw_gbs": 1550.0},
    # H100 (sm_90): 132 SM * 128 cores * 1.83GHz * 2 ≈ 67 TF; ~3000 GB/s HBM3.
    "nvidia_sm90": {"fp32_tflops": 67.0, "dram_bw_gbs": 3000.0},
    # B100/B200 (sm_100): ~28 TF fp32 scalar; ~8000 GB/s HBM3e.
    "nvidia_sm100": {"fp32_tflops": 28.0, "dram_bw_gbs": 8000.0},
    # MI210/MI250 (cdna2): ~24 TF fp32; ~3276 GB/s HBM2e.
    "amd_cdna2": {"fp32_tflops": 24.0, "dram_bw_gbs": 3276.0},
    # MI300 (cdna3): ~80 TF fp32; ~5300 GB/s HBM3.
    "amd_cdna3": {"fp32_tflops": 80.0, "dram_bw_gbs": 5300.0},
    # "any" / reference: no meaningful ceiling; _measure_perf leaves derived None.
    "any": {"fp32_tflops": 0.0, "dram_bw_gbs": 0.0},
}


def arch_peaks(arch: str) -> dict[str, float]:
    """Return the {fp32_tflops, dram_bw_gbs} ceilings for an arch.

    Unknown archs fall back to zeros (so derived metrics stay None rather than
    crashing) — registering a new arch ceiling is a one-line addition here.
    """
    return _ARCH_PEAKS.get(arch, {"fp32_tflops": 0.0, "dram_bw_gbs": 0.0})


# --- per-op FLOP / byte models ------------------------------------------------
# Each model takes a sweep ``point`` (symbolic dims + dtype) and returns
# (flops, bytes_moved). The byte arithmetic is dtype-aware (bf16=2, fp32=4, ...)
# and matches the kernels' actual read/write footprints. Lifted verbatim from
# the validated roofline survey (scripts/ds5_roofline_survey.py).

def _mm_fp8_blockscale(p: dict[str, Any]) -> tuple[int, int]:
    # GEMM: 2*M*N*K flops. Kernel sees fp32 A[M,K],B[K,N],C[M,N] after host dequant
    # (the dequant bytes are host-side, separate launches — not in the kernel's
    # DRAM traffic). This is the kernel-only footprint the survey measured.
    M, K, N = int(p["M"]), int(p["K"]), int(p["N"])
    flops = 2 * M * N * K
    bytes_rw = 4 * (M * K + K * N + M * N)
    return flops, bytes_rw


def _dual_rmsnorm(p: dict[str, Any]) -> tuple[int, int]:
    # ~5 flops/elem (x*x, sum, rsqrt, *w, *scale): per-elem read x + write out, +w once.
    T = int(p["T"])
    d1 = int(p["d1"])
    d2 = int(p["d2"])
    db = dtype_bytes(p.get("dtype", "fp32"))
    flops = 5 * T * (d1 + d2)
    bytes_rw = db * T * (d1 + d2) + db * (d1 + d2) + db * T * (d1 + d2)
    return flops, bytes_rw


def _moe_sum_reduce(p: dict[str, Any]) -> tuple[int, int]:
    # top_k FMAs w/ Kahan (~5 flops/k) + final scale; read top_k*y + top_k*w(fp32) + write out.
    M, top_k, H = int(p["M"]), int(p["top_k"]), int(p["H"])
    yb = dtype_bytes(p.get("dtype", "fp32"))
    flops = (5 * top_k + 1) * M * H
    bytes_rw = yb * M * top_k * H + 4 * M * top_k + 4 * M * H  # out is fp32 (Kahan)
    return flops, bytes_rw


def _mha_merge_state(p: dict[str, Any]) -> tuple[int, int]:
    # ~8 flops/D-elem (2 exp, max, 2 mul+add, div); read 2*bf16 D-tensors + 2 lse + write out+lse.
    T, H, D = int(p["T"]), int(p["H"]), int(p["D"])
    ob = dtype_bytes(p.get("dtype", "fp32"))
    flops = 8 * T * H * D
    bytes_rw = 2 * (T * H * D) * ob + 2 * (T * H) * 4 + (T * H * D) * 4 + (T * H) * 4
    return flops, bytes_rw


def _hc_prenorm_gemm(p: dict[str, Any]) -> tuple[int, int]:
    # GEMM 2*T*N*K + squared-sum T*K; read a(in)+fn(fp32), write mul+sqr (both fp32).
    T, K, N = int(p["T"]), int(p["K"]), int(p["N"])
    ab = dtype_bytes(p.get("dtype", "fp32"))
    flops = 2 * T * N * K + T * K
    bytes_rw = ab * T * K + 4 * N * K + 4 * T * N + 4 * T
    return flops, bytes_rw


def _fused_ffn(p: dict[str, Any]) -> tuple[int, int]:
    # 3 GEMMs (gate, up, down) each 2*M*K*N; bytes ≈ 3*(read M*K + N*K) + 2 inter M*N.
    M, K, N = int(p["M"]), int(p["K"]), int(p["N"])
    db = dtype_bytes(p.get("dtype", "fp32"))
    flops = 3 * 2 * M * K * N
    bytes_rw = db * (M * K + N * K) * 3 + db * M * N * 2
    return flops, bytes_rw


_MODELS: dict[str, Callable[[dict[str, Any]], tuple[int, int]]] = {
    "mm_fp8_blockscale@1.0.0": _mm_fp8_blockscale,
    "dual_rmsnorm@1.0.0": _dual_rmsnorm,
    "moe_sum_reduce@1.0.0": _moe_sum_reduce,
    "mha_merge_state@1.0.0": _mha_merge_state,
    "hc_prenorm_gemm@1.0.0": _hc_prenorm_gemm,
    "fused_ffn@1.0.0": _fused_ffn,
}


def has_model(op_id: str) -> bool:
    return op_id in _MODELS


def cost_model(op_id: str, point: dict[str, Any]) -> tuple[int, int] | None:
    """Return (flops, bytes_moved) for one sweep point, or None if no model.

    None is honest: an op without a registered model still gets a measured
    ``ms``; only the two DERIVED metrics (tflops, achieved_bw_pct) stay None.
    """
    fn = _MODELS.get(op_id)
    if fn is None:
        return None
    return fn(point)
