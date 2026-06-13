# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Launch-config knobs for the sparse-MLA attention compute perf pass (issue #39).

Target: AMD MI300A (gfx942, CDNA3). The sparse-MLA compute (#33) runs one Triton
program per ``(token, head)`` and streams the top-k selected latent KV in
``BLOCK_N`` chunks with online (flash) softmax. At V4 decode (``H=128``,
``D=512``, top-k 512, small ``T``) this is a grid of many tiny programs, each
streaming ``top-k × D`` KV bytes — it is **bandwidth/occupancy bound**.

What the perf pass exposes
--------------------------
The #33 kernel hard-coded ``BLOCK_N=64`` and used default
``num_warps``/``num_stages`` with **no** AMD lowering knobs. This module turns
``BLOCK_N`` and the AMD-only kwargs (``waves_per_eu``, ``matrix_instr_nonkdim``,
``kpack``) into an env-overridable config. ``BLOCK_N`` is a **pure perf knob**:
the flash reduction is exact for any chunk size (the ``m_new == -inf`` guard
already handles partial / all-masked chunks), so changing it does not change the
result — only how many chunks each program streams and how wide each KV load is.

Selection order:

* ``XKERNELS_SPARSE_MLA_CONFIG`` (env, JSON dict) overrides everything — the knob
  the on-device sweep (``benchmarks/tune_sparse_mla.py``) drives.
* Otherwise the behavior-preserving #33 default (``BLOCK_N=64``).

CDNA3 reasoning
---------------
* Larger ``BLOCK_N`` (128/256) makes each KV load wider and better-coalesced and
  cuts the chunk count, but grows the per-program score/value tiles (VGPR / LDS);
  pair with ``num_stages<=2``.
* ``waves_per_eu`` raises occupancy to hide the KV global-load latency; with a
  ``[D]`` (512) fp32 accumulator the VGPR budget is the constraint, so high
  ``waves_per_eu`` is only safe at smaller ``BLOCK_N``.
* ``matrix_instr_nonkdim`` / ``kpack`` are carried for parity with the rest of
  the AMD config space and a future MFMA-tiled score/value path; the current
  kernel scores with ``tl.sum`` (not ``tl.dot``), so they are inert today but
  cost nothing.
"""

from __future__ import annotations

import json
import os

__all__ = [
    "DECODE_SPARSE_MLA_CONFIG",
    "DEFAULT_SPARSE_MLA_CONFIG",
    "resolve_sparse_mla_config",
]

# Measured outcome (issue #39, beverin / MI300A, job 384616). Unlike the MHC GEMM
# there is *no single winner*: the best BLOCK_N depends on the query-token count.
#
#   Tq>1 (prefill / multi-token decode): BLOCK_N=64 (the #33 default) is fastest;
#     larger BLOCK_N regresses (the [D]=512 fp32 accumulator dominates the VGPR/LDS
#     budget, so a wider chunk hurts occupancy — at Tq=8 topk=1024, BLOCK_N=256 is
#     ~4.3x *slower*). So BLOCK_N=64 stays the default — no regression for the
#     common multi-token case.
#   Tq=1 (single-token decode): BLOCK_N=128 num_warps=8 waves_per_eu=1 is 1.13-1.24x
#     faster than BLOCK_N=64 (the grid is only H=128 programs, so wider chunks +
#     more warps raise utilization). This is shipped as the opt-in
#     DECODE_SPARSE_MLA_CONFIG, off by default (mirrors the issue #20/#12
#     precedent: a measured but conditional optimization is opt-in, not hidden).
DEFAULT_SPARSE_MLA_CONFIG: dict = {
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 1,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}

# Opt-in single-token (Tq=1) decode config — measured 1.13-1.24x vs the default at
# Tq=1 on the V4 geometry. Select it by exporting it as XKERNELS_SPARSE_MLA_CONFIG
# when the caller knows it is in the single-token decode regime. NOT the default,
# because it regresses the multi-token case.
DECODE_SPARSE_MLA_CONFIG: dict = {
    "BLOCK_N": 128,
    "num_warps": 8,
    "num_stages": 1,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "kpack": 2,
}


def resolve_sparse_mla_config() -> dict:
    """Resolve the launch config for the sparse-MLA attention compute.

    Order: ``XKERNELS_SPARSE_MLA_CONFIG`` (JSON dict) env override, else the
    behavior-preserving #33 default. Missing keys fall back to the default, so a
    partial override (e.g. ``{"BLOCK_N": 128}``) is valid.
    """
    cfg = dict(DEFAULT_SPARSE_MLA_CONFIG)
    env = os.environ.get("XKERNELS_SPARSE_MLA_CONFIG")
    if env:
        try:
            override = json.loads(env)
        except ValueError as exc:  # pragma: no cover - operator typo
            raise ValueError(
                f"XKERNELS_SPARSE_MLA_CONFIG is not valid JSON: {env!r}"
            ) from exc
        if not isinstance(override, dict):
            raise ValueError(
                f"XKERNELS_SPARSE_MLA_CONFIG must be a JSON object, got {override!r}"
            )
        cfg.update(override)
    block_n = int(cfg["BLOCK_N"])
    if block_n < 1 or (block_n & (block_n - 1)) != 0:
        raise ValueError(f"BLOCK_N must be a positive power of 2, got {block_n}")
    cfg["BLOCK_N"] = block_n
    return cfg
