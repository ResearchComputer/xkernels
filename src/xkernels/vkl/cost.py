# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The IR cost model — predict before measuring (docs/brainstorm/10 §4, 11 §2.3).

Composes with the substrate's ``registry/cost_model.py`` rather than replacing it.
Three jobs, each honest about its accuracy:

  * **scratch footprint** (§4) — the closed-form smem/lds bytes a schedule's tile
    + pipeline stages consume. This is the load-bearing *decidable* signal: a
    config that overflows the arch's scratch budget is rejected by the gate BEFORE
    it launches (and before a kernel crash), turning the Phase 2.2a smem-overflow
    FAILs into clean gate rejects (Phase 2.2b).
  * **roofline** (§4.2) — predicted TFLOPS = min(compute_ceil, bw_ceil * AI), reusing
    the substrate's ``(flops, bytes)`` per op + ``archdb``'s per-instruction peaks.
    The min's winner is the *bottleneck* label — the ``diagnose-memory-bound`` /
    ``diagnose-low-occupancy`` routing signal an agent fires next.
  * **occupancy** (§4.3) — CTAs/SM from smem pressure + warps/CTA. The register-
    pressure half is profile-calibrated (honest: cold-start-unknown, overwritten
    by ncu/rocprof's achieved waves/SM); the smem/warp half is closed-form.

And the Phase 2 gate's decision rule: ``roofline_gate(measured_tflops, arch,
instruction) -> Gate`` returns Pass/Fail at the §2 bar (≥ 70% of the vendor
ceiling). This is the honest go/no-go the Phase 2 plan pre-committed (§3).

DSL-authored ops (``gemm_bf16``, ``dual_rmsnorm``) carry their cost models HERE
(no-touch: ``registry/cost_model.py``'s ``_MODELS`` is not edited — same additive
discipline as ``register_input_gen``). Hand ops delegate to the substrate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..registry.cost_model import arch_peaks
from ..registry.cost_model import cost_model as _substrate_cost_model
from ..registry.dtypes import dtype_bytes
from . import archdb
from .lower.mathbody import TRITON_TILE_KNOBS

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Per-op (flops, bytes) — DSL ops here, hand ops via the substrate
# ═══════════════════════════════════════════════════════════════════════════════


def _gemm_bf16(p: dict[str, Any]) -> tuple[int, int]:
    """Dense GEMM: 2*M*N*K flops; read A[M,K]+B[K,N], write C[M,N], in point dtype."""
    M, K, N = int(p["M"]), int(p["K"]), int(p["N"])
    db = dtype_bytes(p.get("dtype", "bf16"))
    flops = 2 * M * N * K
    bytes_rw = db * (M * K + K * N + M * N)
    return flops, bytes_rw


def _dual_rmsnorm(p: dict[str, Any]) -> tuple[int, int]:
    """Mirror of the substrate's model (so the DSL op has one source of truth)."""
    T, d1, d2 = int(p["T"]), int(p["d1"]), int(p["d2"])
    db = dtype_bytes(p.get("dtype", "fp32"))
    flops = 5 * T * (d1 + d2)
    bytes_rw = db * T * (d1 + d2) + db * (d1 + d2) + db * T * (d1 + d2)
    return flops, bytes_rw


_VKL_MODELS: dict[str, Any] = {
    "gemm_bf16@1.0.0": _gemm_bf16,
    "dual_rmsnorm@1.0.0": _dual_rmsnorm,
}


def workload(op_id: str, point: dict[str, Any]) -> tuple[int, int] | None:
    """Return ``(flops, bytes_moved)`` for one sweep point.

    DSL ops resolve from the vkl table; hand ops delegate to the substrate's
    ``cost_model`` (no-touch). ``None`` if neither has a model (honest: derived
    metrics then stay None, matching ``verify``'s stub).
    """
    fn = _VKL_MODELS.get(op_id)
    if fn is not None:
        return fn(point)
    return _substrate_cost_model(op_id, point)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Scratch footprint — the decidable gate signal (Phase 2.2b)
# ═══════════════════════════════════════════════════════════════════════════════


def tile_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes for a dense tile of ``shape`` in ``dtype``."""
    n = 1
    for d in shape:
        n *= int(d)
    return n * dtype_bytes(dtype)


def predict_scratch(
    pattern: str, config: dict[str, int], dtype: str, arch: str
) -> int:
    """The closed-form smem/lds bytes a ``(pattern, config)`` schedule consumes.

    This is what makes the Phase 2.2b gate bite: a config whose derived stages
    exceed ``archdb.scratch_budget(arch)`` is rejected before launch (and before a
    kernel crash — the two Phase 2.2a FAILs at 4096³ were exactly this).

    ``tiled_2d`` (the GEMM K-loop): both operands stage through scratch at
    ``num_stages`` depth — A tile ``[BLOCK_M, BLOCK_K]`` + B tile ``[BLOCK_K,
    BLOCK_N]`` in the operand dtype, times ``num_stages``. (The accumulator lives
    in registers, not scratch.)

    ``rowwise`` (the dual_rmsnorm reduction): the row tile is register/L1-resident;
    scratch ≈ 0 (no multi-buffered global→smem staging). Honest: if a rowwise op
    later stages through LDS, add its model here.
    """
    if pattern == "tiled_2d":
        bm = config.get("BLOCK_M", 64)
        bn = config.get("BLOCK_N", 64)
        bk = config.get("BLOCK_K", 32)
        stages = config.get("num_stages", 1)
        a_tile = tile_bytes((bm, bk), dtype)
        b_tile = tile_bytes((bk, bn), dtype)
        return stages * (a_tile + b_tile)
    if pattern == "rowwise":
        return 0
    return 0  # unknown pattern: no decidable footprint


def overflows_scratch(
    pattern: str, config: dict[str, int], dtype: str, arch: str
) -> bool:
    """True iff the config's scratch footprint exceeds the arch's budget.

    The 'any' target has budget 0 → never overflows (the portable target makes no
    scratch claim). A concrete arch (sm_90: 228K, cdna3: 64K) rejects greedily.
    """
    budget = archdb.scratch_budget(arch)
    if budget == 0:
        return False
    return predict_scratch(pattern, config, dtype, arch) > budget


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Roofline — predicted TFLOPS + the bottleneck label
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Roofline:
    """The roofline prediction for one (op, point, arch, instruction)."""

    tflops: float              # predicted (the min of the two ceilings)
    compute_ceil: float        # TFLOPS, the mapped instruction's peak
    mem_ceil: float            # TFLOPS achievable at the DRAM bandwidth roof
    arithmetic_intensity: float  # flops / byte
    bottleneck: str            # "compute" | "memory" — the skill-routing signal

    def to_dict(self) -> dict[str, Any]:
        return {
            "tflops": round(self.tflops, 1),
            "compute_ceil": round(self.compute_ceil, 1),
            "mem_ceil": round(self.mem_ceil, 1),
            "arithmetic_intensity": round(self.arithmetic_intensity, 2),
            "bottleneck": self.bottleneck,
        }


def roofline(
    op_id: str, point: dict[str, Any], arch: str, instruction: str = "fma"
) -> Roofline | None:
    """The roofline prediction (docs/brainstorm/10 §4.2).

    Reuses the substrate's ``(flops, bytes)`` per op (via ``workload``) and
    ``archdb``'s per-instruction peaks. The bottleneck label is the causal routing
    signal: ``"memory"`` → ``diagnose-memory-bound``; ``"compute"`` with low
    occupancy → ``diagnose-low-occupancy``; ``"compute"`` with healthy occupancy →
    ``map-to-matrix-cores`` if not already on the L5 engine.
    """
    wb = workload(op_id, point)
    if wb is None:
        return None
    flops, bytes_rw = wb
    if flops == 0 or bytes_rw == 0:
        return None
    compute_ceil = archdb.instr_peak(arch, instruction)
    dram_bw_bs = arch_peaks(arch)["dram_bw_gbs"] * 1e9  # bytes/sec
    ai = flops / bytes_rw
    mem_ceil = (dram_bw_bs * ai) / 1e12  # TFLOPS at the bandwidth roof
    if mem_ceil <= compute_ceil:
        return Roofline(mem_ceil, compute_ceil, mem_ceil, ai, "memory")
    return Roofline(compute_ceil, compute_ceil, mem_ceil, ai, "compute")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Occupancy — CTAs/SM from smem + warps (register half is profile-calibrated)
# ═══════════════════════════════════════════════════════════════════════════════


# Max warps/SM (NVIDIA) / waves-per-SIMD×SIMDs-as-active (AMD). H100: 64 warps/SM.
_ARCH_MAX_WARPS_PER_SM = {"nvidia_sm90": 64, "nvidia_sm80": 64}


@dataclass(frozen=True)
class Occupancy:
    """The closed-form occupancy estimate (the smem/warp half; register half TBD)."""

    warps_per_sm: int
    max_warps_per_sm: int
    fraction: float            # warps_per_sm / max (1.0 = full occupancy)
    limit: str                 # "smem" | "warps" | "max" | "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "warps_per_sm": self.warps_per_sm,
            "fraction": round(self.fraction, 3),
            "limit": self.limit,
        }


def occupancy(
    pattern: str, config: dict[str, int], dtype: str, arch: str
) -> Occupancy:
    """Predict warps/SM from smem pressure + warps/CTA (docs/brainstorm/10 §4.3).

    NVIDIA model: a CTA's smem footprint caps CTAs/SM at ``budget // scratch``;
    its ``num_warps`` caps CTAs/SM at ``max_warps // num_warps``. Active warps =
    min of the two × num_warps. The register-pressure half (regs/thread → warp
    limit) is genuinely profile-calibrated and left ``unknown`` on a cold start
    (the maturity-note discipline: predict what's closed-form, stub the rest).
    """
    max_warps = _ARCH_MAX_WARPS_PER_SM.get(arch, 0)
    if max_warps == 0:
        return Occupancy(0, 0, 0.0, "unknown")
    num_warps = int(config.get("num_warps", 4))
    scratch = predict_scratch(pattern, config, dtype, arch)
    budget = archdb.scratch_budget(arch)
    # CTAs/SM from smem (≥1 if there's any budget; scratch=0 → uncapped by smem)
    if scratch > 0 and budget > 0:
        ctas_smem = max(1, budget // scratch)
    else:
        ctas_smem = max_warps  # no smem limit → warps-only bound
    ctas_warps = max_warps // max(1, num_warps)
    ctas = min(ctas_smem, ctas_warps)
    warps = ctas * num_warps
    limit = "max" if warps >= max_warps else ("smem" if ctas_smem <= ctas_warps else "warps")
    return Occupancy(warps, max_warps, warps / max_warps, limit)


# ═══════════════════════════════════════════════════════════════════════════════
# §5  The Phase 2 gate — the 70% vendor-ceiling decision rule
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GateVerdict:
    """The Phase 2 gate verdict on a measured card (docs/brainstorm/11 §2)."""

    passed: bool               # measured ≥ frac × vendor ceiling
    frac: float                # achieved_fraction
    bar: float                 # the bar (frac × ceiling)
    measured_tflops: float
    ceiling_tflops: float
    instruction: str
    verdict: str               # "PASS" | "BELOW_BAR" — agent-readable

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "passed": self.passed,
            "achieved_fraction": round(self.frac, 3),
            "bar": round(self.bar, 1),
            "measured_tflops": round(self.measured_tflops, 1),
            "ceiling_tflops": round(self.ceiling_tflops, 1),
            "instruction": self.instruction,
        }


def roofline_gate(
    measured_ms: float,
    op_id: str,
    point: dict[str, Any],
    arch: str,
    instruction: str = "wgmma",
    *,
    frac: float = 0.70,
) -> GateVerdict | None:
    """The §2 Phase 2 gate: did the measured card reach ``frac`` of the vendor ceiling?

    ``measured_ms`` is wall-clock (``verify``'s ``perf.ms``); TFLOPS derived from
    the op's workload model. The ceiling is ``archdb.instr_peak(arch, instruction)``
    — graded against the VENDOR ceiling (§10), never against another backend's card.

    Returns ``None`` if the op has no workload model (can't grade). A BELOW_BAR
    verdict is the trigger for the §2 A2 scope-reduction conversation — recorded
    honestly, not papered over.
    """
    wb = workload(op_id, point)
    if wb is None or measured_ms <= 0:
        return None
    flops, _bytes = wb
    measured_tflops = (flops / (measured_ms * 1e-3)) / 1e12
    ceiling = archdb.instr_peak(arch, instruction)
    achieved = measured_tflops / ceiling if ceiling > 0 else 0.0
    passed = achieved >= frac
    return GateVerdict(
        passed=passed, frac=achieved, bar=frac * ceiling,
        measured_tflops=measured_tflops, ceiling_tflops=ceiling,
        instruction=instruction,
        verdict="PASS" if passed else "BELOW_BAR",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Tile-knob helpers (shared by the sweep + the launchers)
# ═══════════════════════════════════════════════════════════════════════════════


def is_tile_config(config: dict[str, int]) -> bool:
    """True iff ``config`` carries any of the canonical Triton tile knob names."""
    return any(k in config for k in TRITON_TILE_KNOBS)
