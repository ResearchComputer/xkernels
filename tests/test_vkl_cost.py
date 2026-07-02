# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 2.2b + 2.3 gates: the cost model predicts before measuring.

``cost.py`` is the analytical layer the Phase 1 ``ir/schedule.py`` needed but
didn't have. Three closed-form signals, each validated against the Phase 2.2a
MEASURED numbers (the honest bar — a cost model is only useful if it would have
predicted the outcome):

  * **scratch footprint** (2.2b) — the two 4096³ FAILs in 2.2a were smem-overflow
    kernel crashes. ``overflows_scratch`` must predict them (256 KB > 228 KB).
  * **roofline** (2.3) — at 4096³ bf16 the GEMM is deeply compute-bound (AI ≈
    1365); the wgmma ceiling is 989 TFLOPS.
  * **the 70% gate** (2.3) — the 2.2a winner (461 TFLOPS) is 46.6% of the wgmma
    ceiling → BELOW_BAR, the honest verdict (cuBLAS-parity on Triton, not the
    theoretical vendor peak).

CPU gates throughout (the cost model is closed-form arithmetic). GPU validation
of the scratch pre-check against the real sweep is in ``test_vkl_sweep.py``.
"""
from __future__ import annotations

import pytest

from xkernels.vkl.cost import (
    Occupancy,
    Roofline,
    occupancy,
    overflows_scratch,
    predict_scratch,
    roofline,
    roofline_gate,
    workload,
)

_ARCH = "nvidia_sm90"
_DTYPE = "bf16"
_4096 = {"dtype": "bf16", "M": 4096, "N": 4096, "K": 4096}

# The Phase 2.2a measured configs (frozen from the real sweep).
_WINNER = {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4}
_FAIL_4 = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4}
_FAIL_8 = {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 4}


# ─── workload (flops, bytes) ─────────────────────────────────────────────────


def test_workload_gemm_matches_2mnk():
    flops, bytes_rw = workload("gemm_bf16@1.0.0", _4096)
    assert flops == 2 * 4096 ** 3
    # bf16 = 2 bytes: read A[M,K] + B[K,N] + write C[M,N]
    assert bytes_rw == 2 * (4096 * 4096 + 4096 * 4096 + 4096 * 4096)


def test_workload_delegates_to_substrate_for_hand_ops():
    # dual_rmsnorm is BOTH a DSL op AND in the substrate's _MODELS; the DSL model
    # mirrors the substrate, so either path gives the same answer. A purely hand
    # op (fused_ffn) delegates to the substrate.
    flops, _ = workload("fused_ffn@1.0.0", {"dtype": "fp32", "M": 128, "K": 128, "N": 128})
    assert flops == 3 * 2 * 128 * 128 * 128


def test_workload_none_for_unknown_op():
    assert workload("nope@1.0.0", _4096) is None


# ─── scratch footprint (the 2.2b gate signal) ────────────────────────────────


class TestScratch:
    def test_tiled_2d_footprint_is_stages_times_two_operand_tiles(self):
        # BLOCK_M=128, BLOCK_K=64, BLOCK_N=256, stages=4, bf16 (2 bytes)
        # = 4 * (128*64*2 + 64*256*2) = 4 * (16384 + 32768) = 196608 = 192 KB
        assert predict_scratch("tiled_2d", _WINNER, _DTYPE, _ARCH) == 4 * (
            128 * 64 * 2 + 64 * 256 * 2
        )

    def test_predicts_the_two_2_2a_fails_as_overflow(self):
        # both FAILs are 256x256x64 at stages=4 → 256 KB > 228 KB budget
        for cfg in (_FAIL_4, _FAIL_8):
            assert overflows_scratch("tiled_2d", cfg, _DTYPE, _ARCH), cfg

    def test_predicts_the_winner_as_within_budget(self):
        assert not overflows_scratch("tiled_2d", _WINNER, _DTYPE, _ARCH)

    def test_fail_footprint_exceeds_budget_by_known_amount(self):
        # 256 KB - 228 KB = 28 KB over
        budget = 228 * 1024
        for cfg in (_FAIL_4, _FAIL_8):
            assert predict_scratch("tiled_2d", cfg, _DTYPE, _ARCH) - budget == 28 * 1024

    def test_any_target_never_overflows(self):
        # the portable 'any' target has budget 0 → no scratch claim
        assert not overflows_scratch("tiled_2d", _FAIL_4, _DTYPE, "any")

    def test_rowwise_has_no_scratch_footprint(self):
        assert predict_scratch("rowwise", {"num_warps": 4}, _DTYPE, _ARCH) == 0


# ─── roofline ─────────────────────────────────────────────────────────────────


class TestRoofline:
    def test_4096_gemm_is_compute_bound_at_wgmma_ceiling(self):
        rl = roofline("gemm_bf16@1.0.0", _4096, _ARCH, "wgmma")
        assert isinstance(rl, Roofline)
        assert rl.bottleneck == "compute"
        assert rl.compute_ceil == 989.0  # H100 SXM bf16 wgmma
        assert rl.tflops == 989.0  # min(compute, mem) = compute
        assert rl.arithmetic_intensity > 100  # deeply compute-bound

    def test_memory_bound_when_ai_low(self):
        # tiny K → low arithmetic intensity → memory-bound
        pt = {"dtype": "bf16", "M": 1024, "N": 1024, "K": 16}
        rl = roofline("gemm_bf16@1.0.0", pt, _ARCH, "wgmma")
        assert rl.bottleneck == "memory"
        assert rl.tflops == rl.mem_ceil < rl.compute_ceil

    def test_none_when_no_workload_model(self):
        assert roofline("nope@1.0.0", _4096, _ARCH) is None


# ─── occupancy (the smem/warp half; register half TBD) ───────────────────────


class TestOccupancy:
    def test_smem_limited_winner(self):
        # winner: 192 KB scratch, budget 228 KB → 1 CTA/SM × 8 warps = 8 warps
        occ = occupancy("tiled_2d", _WINNER, _DTYPE, _ARCH)
        assert isinstance(occ, Occupancy)
        assert occ.warps_per_sm == 8
        assert occ.limit == "smem"
        assert occ.fraction == pytest.approx(8 / 64)

    def test_full_occupancy_with_small_tiles(self):
        # small tiles: 16 KB scratch → 14 CTAs/SM by smem, but only 8 CTAs by
        # warps (64//8); 8 CTAs × 8 warps = 64 = max → full occupancy ('max').
        cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 8, "num_stages": 2}
        occ = occupancy("tiled_2d", cfg, _DTYPE, _ARCH)
        assert occ.warps_per_sm == 64  # full
        assert occ.limit == "max"
        assert occ.fraction == 1.0

    def test_warps_limit_with_low_warps_count(self):
        # 4 warps/CTA: 64//4 = 16 CTAs by warps, but a big-tile smem cap binds first
        # → if smem allows more CTAs than warps, the warp count limits occupancy.
        # Use a config where smem allows ≥16 CTAs but the cap is warps-count.
        cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}
        occ = occupancy("tiled_2d", cfg, _DTYPE, _ARCH)
        # 1 stage × 8 KB = 8 KB → 28 CTAs by smem; 64//4 = 16 CTAs by warps;
        # 16 CTAs × 4 warps = 64 = max → 'max' (full occupancy via many small CTAs)
        assert occ.warps_per_sm == 64
        assert occ.limit == "max"

    def test_unknown_arch(self):
        occ = occupancy("tiled_2d", _WINNER, _DTYPE, "any")
        assert occ.limit == "unknown"
        assert occ.fraction == 0.0


# ─── the Phase 2 gate (70% vendor ceiling) ───────────────────────────────────


class TestRooflineGate:
    def test_below_bar_at_measured_winner(self):
        # 461 TFLOPS (2.2a winner) / 989 wgmma ceiling = 46.6% < 70%
        v = roofline_gate(0.298, "gemm_bf16@1.0.0", _4096, _ARCH, "wgmma")
        assert v is not None
        assert v.verdict == "BELOW_BAR"
        assert not v.passed
        assert v.frac == pytest.approx(0.466, abs=0.01)
        assert v.measured_tflops == pytest.approx(461.2, abs=1.0)

    def test_pass_when_above_bar(self):
        # synthesize a fast measurement: 0.15ms → ~915 TFLOPS → 92.5% → PASS
        v = roofline_gate(0.15, "gemm_bf16@1.0.0", _4096, _ARCH, "wgmma")
        assert v.verdict == "PASS"
        assert v.passed

    def test_none_when_no_model(self):
        assert roofline_gate(0.3, "nope@1.0.0", _4096, _ARCH) is None

    def test_uses_fma_ceiling_for_scalar_instruction(self):
        # the portable 'fma' ceiling (67 TFLOPS) is a much lower bar — the
        # scalar-engine baseline, the honest "no tensor cores" reference
        v = roofline_gate(0.298, "gemm_bf16@1.0.0", _4096, _ARCH, "fma")
        assert v.ceiling_tflops == 67.0
        assert v.passed  # 461 / 67 >> 0.70
