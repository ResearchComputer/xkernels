# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Autotune config space for the native fp8 MFMA block-scale GEMM, reasoned for
CDNA3 (gfx942 / MI300A). Mirrors ``ops/moe/triton/configs.py``.

CDNA3 notes driving the choices:

* Wavefront = 64 lanes; ``num_warps`` counts wavefronts. The fp8 MFMA pipe wants
  enough wavefronts to hide the global-load latency of the (already small) fp8
  operands.
* MFMA shape via ``matrix_instr_nonkdim``: ``16`` -> ``16x16x32`` fp8 MFMA (good
  for tiny-M decode, where a 32x32 would waste M lanes); ``32`` -> ``32x32x16``
  (packs large-M prefill tiles better, lower issue overhead).
* LDS = 64 KB/CU. fp8 operands are HALF the bytes of #40's fp32 tiles, so
  ``BLOCK_K=128`` plus pipelining fits where #40's fp32 64x128 hit
  ``OutOfResources``.
* ``waves_per_eu`` is an occupancy hint (higher hides latency, costs VGPRs);
  ``kpack=2`` packs two K elements per VGPR for the MFMA feed (ds_read/MFMA ratio).

The AMD knobs ride in the ``Config`` kwargs dict: the ``tokenspeed_triton`` AMD
backend reads them; stock Triton forwards-and-ignores them -> portable.
"""
from __future__ import annotations

import triton

__all__ = ["get_autotune_configs", "fp8_gemm_prune_configs", "get_fp8_gemm_config"]

_LDS_BYTES = 64 * 1024  # CDNA3 LDS per CU


def _cfg(bm, bn, bk, gm, *, num_warps, num_stages, waves_per_eu, nonkdim=16, kpack=2):
    return triton.Config(
        {
            "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps, num_stages=num_stages,
    )


def get_autotune_configs():
    """Candidate configs spanning decode (tiny M) -> prefill (large M) on gfx942."""
    return [
        # decode / tiny M: small BLOCK_M, 16x16 MFMA, more occupancy
        _cfg(16, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 256, 128, 1, num_warps=8, num_stages=2, waves_per_eu=2),
        # mid M
        _cfg(64, 128, 128, 4, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 256, 128, 4, num_warps=8, num_stages=2, waves_per_eu=1),
        _cfg(128, 128, 128, 8, num_warps=8, num_stages=2, waves_per_eu=1, nonkdim=32),
        # large M / prefill: 32x32 MFMA, fewer stages to fit LDS
        _cfg(128, 256, 128, 8, num_warps=8, num_stages=1, waves_per_eu=0, nonkdim=32),
        _cfg(256, 128, 128, 8, num_warps=8, num_stages=1, waves_per_eu=0, nonkdim=32),
        # BLOCK_K=64 variant (two sub-dots/block) for LDS-tight large tiles
        _cfg(128, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1, nonkdim=32),
    ]


def _lds_ok(bm, bn, bk, num_stages, op_bytes=1):
    # fp8 A tile [bm,bk] + B tile [bk,bn], double-buffered by num_stages.
    return (bm * bk + bk * bn) * op_bytes * max(1, num_stages) <= _LDS_BYTES


def fp8_gemm_prune_configs(configs, named_args, **kwargs):
    """Drop configs that violate ``BLOCK_K | 128`` or overflow the CDNA3 LDS."""
    out = []
    for c in configs:
        k = c.kwargs
        if 128 % k["BLOCK_K"]:
            continue
        if not _lds_ok(k["BLOCK_M"], k["BLOCK_N"], k["BLOCK_K"], c.num_stages):
            continue
        out.append(c)
    return out or list(configs)


def get_fp8_gemm_config(M: int, N: int, K: int) -> dict:
    """Baked direct-launch config (no runtime autotune).

    Refined by the beverin sweep
    (``slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch``); these are the
    CDNA3-reasoned starting points keyed on the M regime.
    """
    if M <= 16:        # decode
        bm, bn, bk, gm, nw, ns, we, nk = 16, 128, 128, 1, 4, 2, 2, 16
    elif M <= 128:     # mid
        bm, bn, bk, gm, nw, ns, we, nk = 64, 128, 128, 4, 8, 2, 2, 16
    else:              # prefill
        bm, bn, bk, gm, nw, ns, we, nk = 128, 256, 128, 8, 8, 1, 0, 32
    return {
        "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm,
        "waves_per_eu": we, "matrix_instr_nonkdim": nk, "kpack": 2,
        "num_warps": nw, "num_stages": ns,
    }
