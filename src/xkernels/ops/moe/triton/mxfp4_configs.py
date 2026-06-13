# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Autotune config space for the MXFP4 fused-MoE GEMM, reasoned for CDNA3.

Target: AMD MI300A (gfx942, CDNA3). The architectural reasoning is identical to
the INT4 W4A16 kernel (see ``configs.py``): wavefront = 64 lanes, 16x16 MFMA for
the tiny decode M, LDS-fit drives ``num_stages``, ``waves_per_eu`` trades
occupancy for VGPRs. The MXFP4 differences vs INT4:

* **Pack factor 2** (two E2M1 nibbles per uint8) instead of 8 nibbles per int32,
  so ``BLOCK_SIZE_K`` only needs to be a multiple of the group size (32). The
  weight read is one coalesced uint8 tile per 2 logical-K.
* **Two GEMMs per token block** in the gate_up stage (gate + up halves of
  ``w13`` into the same N-tile), so the gate_up tiles carry two fp32 accumulators
  — slightly higher VGPR pressure than the single-accumulator INT4 kernel; the
  config space therefore leans toward modest ``BLOCK_SIZE_N`` and ``waves_per_eu``.

The same config list drives both the gate_up (K = hidden, N = 2*ispp) and the
down (K = ispp, N = hidden) launches; the autotuner keys on ``(N, K, EM, STAGE)``.
"""

from __future__ import annotations

import triton

__all__ = ["get_autotune_configs", "prune_configs", "align_block_m", "get_default_config"]


def _cfg(bm, bn, bk, gm, *, num_warps, num_stages, waves_per_eu, kpack=2, nonkdim=16):
    return triton.Config(
        {
            "BLOCK_SIZE_M": bm,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": bk,
            "GROUP_SIZE_M": gm,
            "waves_per_eu": waves_per_eu,
            "matrix_instr_nonkdim": nonkdim,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
    )


def get_autotune_configs():
    """Candidate configs for the MXFP4 fused-MoE GEMM on gfx942.

    Three regimes (decode / light / heavy prefill), with ``BLOCK_SIZE_K`` a
    multiple of the group size (32). The gate_up stage runs two accumulators, so
    tiles stay moderate to avoid VGPR spills.
    """
    return [
        # ---- decode / tiny-M: weight-read bound, maximize occupancy ----
        _cfg(16, 64, 64, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 64, 128, 1, num_warps=2, num_stages=2, waves_per_eu=3),
        _cfg(16, 128, 64, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(16, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 64, 64, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(32, 128, 64, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(32, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        # ---- light prefill: balanced tiles ----
        _cfg(64, 64, 64, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 64, 8, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 128, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        # ---- heavy prefill: large tiles, low occupancy to avoid VGPR spill ----
        _cfg(128, 128, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        _cfg(128, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=0),
    ]


def align_block_m(M: int) -> int:
    """Token-slot alignment block for ``moe_align_block_size`` — must equal the
    kernel ``BLOCK_SIZE_M`` (one ``expert_ids`` entry per ``BLOCK_SIZE_M`` block).
    """
    return 16 if M <= 32 else 64


def get_default_config(M: int) -> dict:
    """A safe, fixed launch config for the production (non-autotuned) path.

    The fused-MoE down stage atomic-accumulates into the ``[M, hidden]`` output,
    so it MUST NOT run under ``@triton.autotune`` against the real buffer (every
    benchmarked config would add its result, multiplying the output). The wrapper
    therefore resolves one config here and takes the direct launch path. The
    ``BLOCK_SIZE_M`` matches :func:`align_block_m` so the sort/pad granularity and
    the kernel's per-block ``expert_ids`` indexing agree. ``BLOCK_SIZE_K=64`` is a
    multiple of the group size (32) and divides the V4 contracted dims
    (hidden=4096, ispp=512). Offline-tuned winners can be swapped in later, exactly
    like the INT4 ``tuned_configs/`` path.
    """
    bm = align_block_m(M)
    if M <= 32:  # decode: tiny M, wide-ish N, high occupancy
        return {
            "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "kpack": 2,
        }
    return {  # prefill: larger M, balanced tile, low occupancy
        "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2,
    }


def prune_configs(configs, named_args, **kwargs):
    """Drop configs that cannot run or mis-align for the given problem.

    Removes ``BLOCK_SIZE_K`` not a multiple of the group size, over-large
    ``BLOCK_SIZE_N`` for tiny ``N``, and — when the token count is known — configs
    whose ``BLOCK_SIZE_M`` differs from ``align_block_m(M)`` (the GEMM indexes
    ``expert_ids`` per ``BLOCK_SIZE_M`` block, so the sort/pad granularity must
    match the tile M).
    """

    def g(k, default=None):
        if k in named_args:
            return named_args[k]
        return kwargs.get(k, default)

    group_k = g("group_k", 32)
    N = g("N")
    K = g("K")
    nvt = g("num_valid_tokens")
    top_k = g("top_k")
    bm_required = None
    if nvt is not None and top_k:
        bm_required = align_block_m(int(nvt) // int(top_k))

    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_SIZE_K"]
        bn = c.kwargs["BLOCK_SIZE_N"]
        bm = c.kwargs["BLOCK_SIZE_M"]
        if bk % group_k != 0:
            continue
        if K is not None and bk > max(K, group_k):
            continue
        if N is not None and bn > 2 * N:
            continue
        if bm_required is not None and bm != bm_required:
            continue
        pruned.append(c)
    if pruned:
        return pruned
    if bm_required is not None:
        keep = [c for c in configs if c.kwargs["BLOCK_SIZE_M"] == bm_required]
        if keep:
            return keep
    return list(configs)
