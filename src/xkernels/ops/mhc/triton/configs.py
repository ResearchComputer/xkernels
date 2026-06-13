# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Launch-config knobs for the MHC prenorm GEMM perf pass (issue #39).

Target: AMD MI300A (gfx942, CDNA3). The MHC prenorm GEMM (#36) is a
**memory-bound tall-skinny** problem: huge ``K = hc_mult*hidden`` (16384 at
V4-Flash ``hc_mult=4, hidden=4096``) and tiny ``N = 2*hc_mult + hc_mult**2`` (24).
The dominant cost is streaming ``A [T, K]`` once; the ``A @ fnᵀ`` matmul has a
tiny ``N`` so the ``tl.dot`` is small and the per-row ``Σ A²`` reuses the same
loads for free.

What the perf pass exposes
--------------------------
The original #36 kernel hard-coded ``BLOCK_M=64, BLOCK_K=64``, ``BLOCK_N =
next_pow2(N)``, default ``num_warps``/``num_stages`` and **no** AMD lowering
knobs. This module turns those into a small, CDNA3-reasoned config space and
threads the AMD-only kwargs (``waves_per_eu``, ``matrix_instr_nonkdim``,
``kpack``) through the launch — exactly the mechanism the INT4 MoE GEMM
(``ops/moe/triton/configs.py``) already uses. The knobs are ignored by non-AMD
Triton backends, so the same configs stay portable / interpreter-safe.

Selection is deliberately conservative:

* ``XKERNELS_MHC_GEMM_CONFIG`` (env, JSON dict) overrides everything — this is
  the knob the on-device sweep (``benchmarks/tune_mhc_prenorm_gemm.py``) drives
  to characterize candidates without editing code.
* Otherwise the **default** config is returned, which reproduces the #36 launch
  (``BLOCK_M=BLOCK_K=64``) so behavior is unchanged until a measured winner is
  promoted here.

CDNA3 reasoning for the candidate space
---------------------------------------
* Wavefront = 64 lanes; ``num_warps`` counts 64-lane wavefronts.
* ``BLOCK_K`` drives the K-streaming granularity. Larger ``BLOCK_K`` (128/256)
  gives wider, better-coalesced ``A`` reads (the bandwidth bottleneck) at the
  cost of LDS/VGPR; pair big ``BLOCK_K`` with ``num_stages<=2``.
* ``waves_per_eu`` raises occupancy to hide the long K-stream global-load
  latency; with tiny ``N`` the accumulator is small, so VGPR headroom is large
  and ``waves_per_eu`` 2-4 is affordable.
* ``matrix_instr_nonkdim=16`` selects the 16x16 MFMA — appropriate because the M
  dimension (decode T) is small; a 32x32 MFMA would waste M lanes.
* ``kpack=2`` packs two K per VGPR for the MFMA feed, improving the
  ds_read/MFMA ratio over the wide effective K tile.
"""

from __future__ import annotations

import json
import os

import triton

__all__ = [
    "BASELINE_MHC_GEMM_CONFIG",
    "DEFAULT_MHC_GEMM_CONFIG",
    "get_autotune_configs",
    "resolve_mhc_gemm_config",
]

# Measured default (issue #39, beverin / MI300A, job 384616). The on-device sweep
# found BLOCK_M=32, BLOCK_K=128, waves_per_eu=4 to be the fastest config at every
# decode batch size tested (T=1/8/64) on the V4-Flash MHC shape (K=16384, N=24):
# 1.48-1.63x faster than the #36 baseline (BLOCK_M=BLOCK_K=64), rel_err ~5e-4
# (within fp32-accumulation-order tolerance). The smaller BLOCK_M (32) packs the
# tiny-T rows tighter and frees VGPRs for higher occupancy (waves_per_eu=4),
# which on this memory-bound K-stream hides the global-load latency better; the
# wider BLOCK_K=128 doubles the per-load A/fn read width. The #36 baseline lives
# on as BASELINE_MHC_GEMM_CONFIG for A/B regression.
DEFAULT_MHC_GEMM_CONFIG: dict = {
    "BLOCK_M": 32,
    "BLOCK_K": 128,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 4,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}

# The original #36 launch, retained for A/B comparison and as a documented
# fallback (set XKERNELS_MHC_GEMM_CONFIG to this dict to reproduce #36 timing).
BASELINE_MHC_GEMM_CONFIG: dict = {
    "BLOCK_M": 64,
    "BLOCK_K": 64,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}


def _cfg(bm, bk, *, num_warps, num_stages, waves_per_eu, kpack=2, nonkdim=16):
    """Build a Triton ``Config`` carrying the AMD lowering knobs as kwargs.

    ``BLOCK_N`` is intentionally *not* in the autotune space: ``N`` is tiny (24)
    so ``BLOCK_N = next_pow2(N)`` is fixed by the wrapper; tuning it buys nothing.
    """
    return triton.Config(
        {
            "BLOCK_M": bm,
            "BLOCK_K": bk,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


def get_autotune_configs():
    """Candidate launch configs for the MHC prenorm GEMM on gfx942.

    The space sweeps the K-streaming granularity (``BLOCK_K`` in 64/128/256), the
    row tile (``BLOCK_M`` in 32/64/128 — small for decode, larger for prefill),
    occupancy (``waves_per_eu``) and ``num_warps``/``num_stages``. It is small on
    purpose: the problem is memory-bound, so the meaningful axes are read width
    (``BLOCK_K``) and occupancy.
    """
    # On-device LDS constraint (issue #39, beverin / gfx942): CDNA3 has 64 KB
    # (65536 B) LDS per CU. The fp32 fn tile is [BLOCK_K, BLOCK_N=next_pow2(24)=32];
    # with software-pipelining (num_stages>1) Triton allocates num_stages copies
    # of the A and fn tiles. BLOCK_K=256 fp32 at num_stages=2 needs 96 KB and
    # raises OutOfResources(98304, 65536). So BLOCK_K=256 candidates are pinned to
    # num_stages=1 (halves the LDS); even then they may not fit on every shape, so
    # the sweep / tests treat OutOfResources as "config infeasible here", not a bug.
    return [
        # ---- baseline (#36) ----
        _cfg(64, 64, num_warps=4, num_stages=2, waves_per_eu=0),
        # ---- wider K reads to raise effective bandwidth ----
        _cfg(64, 128, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 256, num_warps=4, num_stages=1, waves_per_eu=2),
        _cfg(32, 128, num_warps=4, num_stages=2, waves_per_eu=4),
        _cfg(32, 256, num_warps=4, num_stages=1, waves_per_eu=3),
        # ---- more warps to stream K faster (tiny N keeps VGPRs free) ----
        _cfg(64, 128, num_warps=8, num_stages=2, waves_per_eu=2),
        _cfg(64, 256, num_warps=8, num_stages=1, waves_per_eu=1),
        _cfg(128, 128, num_warps=8, num_stages=2, waves_per_eu=1),
        # ---- higher occupancy for tiny decode T ----
        _cfg(32, 128, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 256, num_warps=2, num_stages=1, waves_per_eu=4),
    ]


def resolve_mhc_gemm_config() -> dict:
    """Resolve the launch config for the MHC prenorm GEMM.

    Order: ``XKERNELS_MHC_GEMM_CONFIG`` (JSON dict) env override, else the
    behavior-preserving #36 default. Unknown keys in the override are ignored by
    the wrapper; missing keys fall back to the default, so a partial override
    (e.g. ``{"BLOCK_K": 256}``) is valid.
    """
    cfg = dict(DEFAULT_MHC_GEMM_CONFIG)
    env = os.environ.get("XKERNELS_MHC_GEMM_CONFIG")
    if env:
        try:
            override = json.loads(env)
        except ValueError as exc:  # pragma: no cover - operator typo
            raise ValueError(
                f"XKERNELS_MHC_GEMM_CONFIG is not valid JSON: {env!r}"
            ) from exc
        if not isinstance(override, dict):
            raise ValueError(
                f"XKERNELS_MHC_GEMM_CONFIG must be a JSON object, got {override!r}"
            )
        cfg.update(override)
    return cfg
