# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The architecture database extension (docs/brainstorm/10 §4.1).

Extends ``registry/archs.py`` + ``registry/cost_model.py`` with the three pieces
the IR's cost model + check gate need that the substrate does not yet model:

  * ``ARCH_INSTR_PEAK``  — per-INSTRUCTION ceilings (wgmma vs FMA: the ~15× lever
                           the substrate's scalar-only ``fp32_tflops`` hides).
  * ``ARCH_NATIVE_SHAPE`` — the L5 MMA shapes that must divide L2 tiles
                            (the ``Retile`` gate, docs/brainstorm/10 §5 row 1).
  * ``ARCH_SCRATCH_BYTES`` — the L2 budget ``AddStage``/``Tile`` are checked against.

This module *calls* ``cost_model.arch_peaks`` for the scalar ceiling (so it
stays in sync with the substrate) and *adds* the matrix-engine data on top. It
never edits the existing tables (the no-touch rule, docs/brainstorm/11 §0).
"""
from __future__ import annotations

from ..registry.cost_model import arch_peaks

# Per-arch, per-instruction peak TFLOPS (bf16 unless noted). The scalar ``fma``
# ceiling is copied from ``arch_peaks`` so a cold-start prediction is honest
# even before any matrix-engine mapping; the matrix-engine ceilings are the new
# data. Phase 1 records bf16 + fp32; fp8 (~2× bf16) lands with Phase 2's fp8 cards.
ARCH_INSTR_PEAK: dict[str, dict[str, float]] = {
    "nvidia_sm90": {  # H100 SXM
        "fma": arch_peaks("nvidia_sm90")["fp32_tflops"],  # 67.0 — scalar ceiling
        "wmma": 989.0,   # bf16 tensor-core ceiling (wmma family)
        "wgmma": 989.0,  # bf16 wgmma (sm_90 native), dense
    },
    "nvidia_sm80": {  # A100
        "fma": arch_peaks("nvidia_sm80")["fp32_tflops"],  # 19.5
        "wmma": 312.0,   # bf16 tensor-core ceiling
    },
    "nvidia_sm121": {  # GB10 Grace-Blackwell (DGX Spark, aarch64)
        "fma": arch_peaks("nvidia_sm121")["fp32_tflops"],  # 29.5 scalar
        "wgmma": 92.0,   # bf16 tensor-core ceiling (measured cuBLAS, 2048^3)
        # fp32 via TF32 tensor mode reaches ~38 TFLOPS but the kernel measures
        # against an fp32 reference; triton 3.6 on sm_121 does NOT honor
        # input_precision=ieee for fp32 tl.dot (falls to tf32) — the native
        # override path restores true fp32 (Phase 2.1).
    },
    "amd_cdna3": {  # MI300
        "fma": arch_peaks("amd_cdna3")["fp32_tflops"],  # 80.0
        "mfma": 1300.0,  # bf16 MFMA ceiling (32×32 instr family)
    },
    "amd_cdna2": {  # MI250
        "fma": arch_peaks("amd_cdna2")["fp32_tflops"],  # 24.0
        "mfma": 382.0,   # bf16 MFMA ceiling
    },
    # "any" / reference: no matrix engine; only scalar FMA is meaningful.
    "any": {"fma": 0.0},
}

# The L5 native MMA shapes that must divide the L2 output tile. The Retile gate
# (docs/brainstorm/10 §5 row 1) rejects a tile whose M dim isn't divisible by the
# mapped engine's native m — this is the rule that catches "warp=32"-style hidden
# assumptions at edit time, before they become a compile failure.
ARCH_NATIVE_SHAPE: dict[str, dict[str, dict[str, int]]] = {
    "nvidia_sm90": {"wmma": {"m": 16, "k": 16}, "wgmma": {"m": 64, "k": 16}},
    "nvidia_sm121": {"wgmma": {"m": 16, "k": 16}},  # Blackwell tensor core
    "nvidia_sm80": {"wmma": {"m": 16, "k": 16}},
    "amd_cdna3": {"mfma": {"m": 32, "k": 16}},
    "amd_cdna2": {"mfma": {"m": 16, "k": 16}},
}

# Scratch budget per CTA/workgroup (the L2 pipeline budget). Phase 1 uses these
# as the AddStage/Tile overflow ceiling (docs/brainstorm/10 §5, scratch row).
ARCH_SCRATCH_BYTES: dict[str, int] = {
    "nvidia_sm90": 228 * 1024,  # shared mem per CTA, H100
    "nvidia_sm121": 48 * 1024,  # shared mem per CTA default, GB10 (48 SMs, opt-in higher)
    "nvidia_sm80": 164 * 1024,  # shared mem per CTA, A100 (configurable up to 164K)
    "amd_cdna3": 64 * 1024,     # LDS per workgroup, MI300A
    "amd_cdna2": 64 * 1024,     # LDS per workgroup, MI250
    "any": 0,                   # no scratch budget for the portable target
}


def legal_instructions(arch: str) -> tuple[str, ...]:
    """The instructions a schedule may map to on ``arch`` (vendor-honest)."""
    return tuple(ARCH_INSTR_PEAK.get(arch, {"fma": 0.0}).keys())


def native_shape(arch: str, instruction: str) -> dict[str, int] | None:
    """The L5 native shape for (arch, instruction), or None for scalar fma."""
    return ARCH_NATIVE_SHAPE.get(arch, {}).get(instruction)


def scratch_budget(arch: str) -> int:
    """The L2 scratch byte budget for ``arch`` (0 for the portable 'any' target)."""
    return ARCH_SCRATCH_BYTES.get(arch, 0)


def instr_peak(arch: str, instruction: str) -> float:
    """The peak TFLOPS for (arch, instruction). KeyError if illegal — by design."""
    return ARCH_INSTR_PEAK[arch][instruction]
