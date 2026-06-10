# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Autotune config space for the INT4 W4A16 fused-MoE GEMM, reasoned for CDNA3.

Target: AMD MI300A (gfx942, CDNA3). Notes on the architecture that drive the
choices below:

* Wavefront = 64 lanes. ``num_warps`` here counts 64-lane wavefronts; a CU has
  4 SIMD32 units, and the MFMA pipe wants enough wavefronts to hide the
  global-load latency of the packed-int4 weights.
* MFMA: CDNA3 has ``v_mfma_*_16x16x*`` and ``32x32x*`` instructions. We pin
  ``matrix_instr_nonkdim=16`` (16x16) because the decode regime has tiny M
  (1 token x top_k), so a 32x32 MFMA would waste most of its M lanes; 16x16
  packs the work better and keeps register pressure low.
* LDS = 64 KB/CU. With double/triple buffering the A and dequantized-B tiles
  must fit; large BLOCK_N x BLOCK_K bf16 B tiles + 2-3 stages can overflow LDS,
  so big tiles are paired with ``num_stages<=2``.
* ``waves_per_eu`` (occupancy hint): higher values let more wavefronts hide
  memory latency, but cost VGPRs. Small-M tiles have spare VGPRs (M is tiny) so
  they can afford ``waves_per_eu=2-4``; large prefill tiles set it to 0/1 so the
  big accumulator does not spill.
* ``kpack=2`` packs two K elements per VGPR for the MFMA feed on CDNA3,
  improving the ds_read/MFMA ratio — relevant since our effective K tile is wide
  after unpack.

Memory-bandwidth framing (decode). At M=1xtop_k the GEMM is *weight-read bound*:
each expert weight is read once and barely reused (M ~ 1-8 rows). Packed int4 is
~4x smaller than bf16, so the design goal is to (a) keep the weight read one
coalesced int32 load per 8 K, (b) minimize per-element dequant overhead, and
(c) pick BLOCK_N large enough to amortize the per-tile scale fetch and launch
overhead while keeping BLOCK_M tiny. BLOCK_K is a multiple of both the pack
factor (8) and the group size (32) so unpack and scale-broadcast are clean.
"""

from __future__ import annotations

import json
import os
import warnings

import triton

__all__ = [
    "get_autotune_configs",
    "prune_configs",
    "align_block_m",
    "get_moe_int4_config",
    "load_tuned_config",
]


def _cfg(bm, bn, bk, gm, *, num_warps, num_stages, waves_per_eu, kpack=2, nonkdim=16):
    """Build a Triton ``Config``.

    AMD-only lowering knobs (``waves_per_eu``, ``matrix_instr_nonkdim``,
    ``kpack``) are passed through the kwargs dict; the Triton AMD backend reads
    them and they are ignored on other backends, so the same config list is
    portable. ``matrix_instr_nonkdim=16`` selects the 16x16 MFMA.
    """
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
    """Return the candidate configs for the INT4 W4A16 fused-MoE GEMM on gfx942.

    The space spans three regimes:

    * **Decode (tiny M)** — ``BLOCK_M in {16, 32}``: minimal M, wide-ish N to
      amortize scale + launch overhead, ``waves_per_eu`` high to hide the int4
      weight read, modest ``num_warps`` (2-4) since the M dim is too small to
      feed 8 wavefronts. ``GROUP_SIZE_M=1`` (no L2 super-grouping needed — at
      tiny M there is little weight reuse to exploit).
    * **Light prefill (M ~ 64-256)** — balanced square-ish tiles.
    * **Heavy prefill (M >= 512)** — large ``BLOCK_M``/``BLOCK_N``, more warps,
      ``waves_per_eu`` low to avoid VGPR spills on the big accumulator,
      ``GROUP_SIZE_M`` larger for L2 reuse across the now-shared weights.

    ``BLOCK_SIZE_K`` candidates are multiples of 32 (group size) and 8 (pack):
    64/128/256. 256 maximizes weight-read coalescing (32 int32 per row) but
    needs ``num_stages<=2`` to fit LDS; 64 keeps register/LDS pressure low for
    the smallest tiles.
    """
    configs = [
        # ---- decode / tiny-M: weight-read bound, maximize occupancy ----
        _cfg(16, 64, 128, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 64, 256, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(16, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(16, 128, 256, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(16, 256, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 64, 128, 1, num_warps=2, num_stages=2, waves_per_eu=4),
        _cfg(32, 128, 128, 1, num_warps=4, num_stages=2, waves_per_eu=3),
        _cfg(32, 128, 256, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(32, 256, 128, 1, num_warps=8, num_stages=2, waves_per_eu=2),
        # ---- light prefill: balanced tiles ----
        _cfg(64, 64, 128, 1, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 128, 8, num_warps=4, num_stages=2, waves_per_eu=2),
        _cfg(64, 128, 64, 8, num_warps=4, num_stages=3, waves_per_eu=2),
        _cfg(64, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        _cfg(128, 128, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1),
        # ---- heavy prefill: large tiles, low occupancy to avoid VGPR spill ----
        _cfg(128, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=0),
        _cfg(256, 128, 64, 8, num_warps=8, num_stages=2, waves_per_eu=0),
        _cfg(128, 128, 128, 4, num_warps=8, num_stages=2, waves_per_eu=1),
    ]
    return configs


def prune_configs(configs, named_args, **kwargs):
    """Drop configs that cannot run (or would mis-align) for the given problem.

    Removes configs whose ``BLOCK_SIZE_K`` is not a multiple of the quant group
    size (the scale-broadcast reshape requires ``BLOCK_K % group_k == 0``) or of
    the pack factor (8), over-large ``BLOCK_SIZE_N`` for tiny ``N``, and — when
    the token count is known — configs whose ``BLOCK_SIZE_M`` does not equal
    ``align_block_m(M)``. The last guard keeps the fallback autotune path
    consistent with the wrapper's ``moe_align_block_size`` granularity, since the
    kernel indexes ``expert_ids`` per ``BLOCK_SIZE_M``-block.
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
        if bk % 8 != 0:  # pack factor
            continue
        if K is not None and bk > max(K, group_k):
            continue
        if N is not None and bn > 2 * N:  # allow one over-tile for masking
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


# --- tuned-config persistence (issue #16) ----------------------------------
# Checked-in winners live in tuned_configs/E=..,N=..,K=..,device_name=..,
# dtype=int4_w4a16.json, mapping a token-batch M-bucket -> launch config. The
# production launch path resolves one directly (no runtime autotune); untuned
# shapes fall back to @triton.autotune. Keys starting with "_" are metadata.

_TUNED_CACHE: dict = {}
_DEVICE_NAME_MEMO: list = []  # one-element memo for the live device name


def align_block_m(M: int) -> int:
    """Token-slot alignment block for ``moe_align_block_size``.

    Must equal the kernel ``BLOCK_SIZE_M``: the grouped GEMM reads one
    ``expert_ids`` entry per ``BLOCK_SIZE_M``-block, so the sort/pad granularity
    and the tile M must match or the kernel reads the wrong expert. Small-M
    decode uses 16; larger M uses 64.
    """
    return 16 if M <= 32 else 64


def _device_name(arch: str | None = None) -> str | None:
    """Normalized device string used in tuned-config filenames, or ``None``.

    ``arch`` overrides; then ``$XKERNELS_MOE_ARCH``; then the live CUDA/ROCm
    device name. Returns ``None`` when no device is visible (CPU / interpreter),
    which makes ``get_moe_int4_config`` a no-op so the autotune fallback runs.
    """
    if arch is not None:
        return arch.replace(" ", "_")
    env = os.environ.get("XKERNELS_MOE_ARCH")
    if env:
        return env.replace(" ", "_")
    if _DEVICE_NAME_MEMO:
        return _DEVICE_NAME_MEMO[0]
    name = None
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).replace(" ", "_")
    except Exception:
        name = None
    _DEVICE_NAME_MEMO.append(name)
    return name


def _config_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "tuned_configs")


def _config_filename(E: int, N: int, K: int, device: str, dtype: str) -> str:
    return f"E={E},N={N},K={K},device_name={device},dtype={dtype}.json"


def load_tuned_config(E: int, N: int, K: int, device: str, dtype: str = "int4_w4a16"):
    """Load (and cache) the checked-in tuned-config table for a shape, or ``None``."""
    key = (E, N, K, device, dtype)
    if key in _TUNED_CACHE:
        return _TUNED_CACHE[key]
    path = os.path.join(_config_dir(), _config_filename(E, N, K, device, dtype))
    table = None
    if os.path.exists(path):
        try:
            with open(path) as fh:
                table = json.load(fh)
        except (OSError, ValueError):
            warnings.warn(f"could not read tuned MoE config {path!r}", stacklevel=2)
            table = None
    _TUNED_CACHE[key] = table
    return table


def _select_config(table: dict, M: int):
    """Pick the config for the closest tabulated bucket <= M (clamped to range)."""
    buckets = sorted(int(k) for k in table if not str(k).startswith("_"))
    if not buckets:
        return None
    chosen = buckets[0]
    for b in buckets:
        if b <= M:
            chosen = b
        else:
            break
    return {k: v for k, v in table[str(chosen)].items() if not str(k).startswith("_")}


def get_moe_int4_config(
    E: int, N: int, K: int, M: int, dtype: str = "int4_w4a16", arch: str | None = None
):
    """Return the tuned launch config for ``(shape, M)`` on this device, or ``None``."""
    device = _device_name(arch)
    if device is None:
        return None
    table = load_tuned_config(E, N, K, device, dtype)
    if not table:
        return None
    return _select_config(table, M)
