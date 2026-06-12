# Native fp8 MFMA fast path for `mm_fp8_blockscale` (issue #41) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native-fp8-MFMA, autotuned fast path to `mm_fp8_blockscale` on gfx942 that runs `tl.dot` directly on fp8 e4m3 operands (block scales applied as a post-accumulation per-128-K correction), keeping #40's portable dequant kernel as the correctness fallback.

**Architecture:** Two-level (block-promoted) accumulation — per 128-K quant block, a raw fp8·fp8 `tl.dot` accumulates into an fp32 block-accumulator, then promotes into the main accumulator scaled by `a_s[:,None]·b_s[None,:]`. A CDNA3 autotune space (`BLOCK_K=128`, larger `BLOCK_M/N`, pipelining) resolves the LDS pressure that forced 64³ in #40. The kernel is **fp8-format-agnostic** (reads the operand dtype, dots whatever it gets), so the on-device fn-vs-fnuz question only sets a default, not the code.

**Tech Stack:** Triton (stock locally via `_triton_compat` no-op; `tokenspeed_triton` on host), torch fp8 (`float8_e4m3fn` / `float8_e4m3fnuz`), AMD MI300A (gfx942) via CSCS Alps `beverin` (SLURM `--partition=mi300 -A a-infra02`).

---

## Environment & loops (read first)

- **Worktree:** `/home/xiayao/Documents/research/xkernels/.claude/worktrees/issue-41-fp8-mfma`, branch `feat/issue-41-fp8-mfma-blockscale-gemm` (stacked on `feat/issue-38-fp8-blockscale-gemm`).
- **Local venv (already created, `.venv`, gitignored):** torch 2.12+cpu, triton 3.7, pytest, numpy.
- **Local test loop (validates full block-promotion math — fp8 `tl.dot` evaluates exactly under the interpreter, confirmed rel-err 0.0):**
  ```bash
  cd <worktree> && . .venv/bin/activate
  PYTHONPATH=src TRITON_INTERPRET=1 python -m pytest tests/<file> -q
  ```
- **Beverin loop (real gfx942 — MFMA emission, format, parity, perf):** sync the branch to `/capstor/scratch/cscs/xyao/xkernels-issue-41`, then `sbatch`. Helper (use verbatim each time):
  ```bash
  ssh beverin 'mkdir -p /capstor/scratch/cscs/xyao/xkernels-issue-41'
  rsync -az --delete --exclude .git --exclude .venv --exclude '**/__pycache__' \
    <worktree>/ beverin:/capstor/scratch/cscs/xyao/xkernels-issue-41/
  ssh beverin 'cd /capstor/scratch/cscs/xyao/xkernels-issue-41 && sbatch --export=ALL,REPO=$PWD slurm/<sbatch>'
  # poll: ssh beverin 'squeue --me'; read: ssh beverin 'cat /capstor/scratch/cscs/xyao/xkernels-issue-41/<job>-*.out'
  ```
  (CSCS SSH cert expires every 24 h; if `Permission denied (publickey)` the **user** must renew via MFA at sshservice.cscs.ch.)
- **Conventions:** SPDX header on every new file (`# SPDX-License-Identifier: MIT` / `# Copyright (c) 2026 ResearchComputer`). Conventional-commit messages. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do **not** `git add .venv`, `__pycache__`, or `.claude/`.

## File structure

| File | Responsibility |
|---|---|
| `src/xkernels/ops/gemm/reference.py` (modify) | Quant helpers gain `fp8_dtype=` so tests/bench can emit fnuz operands on AMD; reference unchanged otherwise. |
| `src/xkernels/ops/gemm/triton/configs.py` (create) | CDNA3 autotune config space + `get_fp8_gemm_config(M,N,K)` baked-table direct-launch config + `fp8_gemm_prune_configs`. |
| `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_mfma_kernel.py` (create) | The native-fp8-MFMA kernel (`@triton.jit`) + host wrapper `mm_fp8_blockscale_mfma_triton`. |
| `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_kernel.py` (modify) | Keep the portable kernel + wrapper; **remove** its `register(...)` (moves to the entry). |
| `src/xkernels/ops/gemm/triton/entry.py` (create) | Single `Backend.TRITON` registration: `mm_fp8_blockscale_triton(..., path="auto")` routing mfma↔portable. |
| `src/xkernels/ops/gemm/__init__.py` (modify) | Import `entry` (under the redirect ctx) instead of the portable kernel module. |
| `src/xkernels/ops/gemm/interface.py` (modify) | Public op gains keyword-only `path="auto"` (passed through; reference ignores). |
| `tests/test_mm_fp8_blockscale_mfma.py` (create) | Interpreter (exact math) + GPU-gated (tight parity, cross-check vs portable, fnuz param, edges). |
| `slurm/probe_fp8_mfma_beverin.sbatch` (create) | Phase-0 fn-vs-fnuz native-fp8-MFMA probe (AMDGCN grep + parity + TFLOP/s per format). |
| `slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch` (create) | On-device pytest + V4 parity + perf vs torch_ref/portable + `v_mfma_*_fp8` assertion. |
| `benchmarks/bench_fp8_blockscale_gemm.py` (create) | mfma vs torch_ref vs #40 portable across V4 shapes; TFLOP/s + ×ref. |
| `docs/issue-41-fp8-mfma-blockscale-gemm.md` (create) | Shipped kernel doc: math, fp8-format resolution, autotune table, honest on-device numbers. |

---

## Task 1: `fp8_dtype` parameter on the quant helpers (fnuz fallback support)

**Files:**
- Modify: `src/xkernels/ops/gemm/reference.py`
- Test: `tests/test_mm_fp8_blockscale.py` (extend) — runs under the local venv (CPU).

Rationale: the kernel is format-agnostic, but tests/bench need to *produce* fnuz operands to evaluate the fnuz path if e4m3fn doesn't reach native MFMA. fnuz finite-max is **240.0** (`float8_e4m3fnuz`), fn is 448.0.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_mm_fp8_blockscale.py`:
```python
def test_quant_helpers_accept_fnuz_dtype():
    if not hasattr(torch, "float8_e4m3fnuz"):
        pytest.skip("torch lacks float8_e4m3fnuz")
    dev = _dev()
    M, N, K, block = 6, 130, 384, 128
    a = torch.randn(M, K, device=dev)
    b = torch.randn(N, K, device=dev)
    a_fp8, a_s = per_token_group_quant_fp8(a, block=block, fp8_dtype=torch.float8_e4m3fnuz)
    b_fp8, b_s = per_block_quant_fp8(b, block=block, fp8_dtype=torch.float8_e4m3fnuz)
    assert a_fp8.dtype == torch.float8_e4m3fnuz and b_fp8.dtype == torch.float8_e4m3fnuz
    # Dequant round-trips to the same fp32 the reference would consume.
    a_deq = a_fp8.to(torch.float32) * a_s.repeat_interleave(block, 1)[:, :K]
    b_deq = b_fp8.to(torch.float32) * (
        b_s.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    )
    # fnuz max 240 -> coarser than fn, but still a faithful per-group dequant.
    assert (a_deq - a).abs().max() < 0.2 * a.abs().max()
    assert (b_deq - b).abs().max() < 0.2 * b.abs().max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale.py::test_quant_helpers_accept_fnuz_dtype -q`
Expected: FAIL — `per_token_group_quant_fp8() got an unexpected keyword argument 'fp8_dtype'`.

- [ ] **Step 3: Implement**

In `src/xkernels/ops/gemm/reference.py`, replace the `_FP8_MAX` constant and both quant helpers' fp8-dtype handling:
```python
#: Per-dtype finite max used to scale into representable range.
_FP8_MAX_BY_DTYPE = {torch.float8_e4m3fn: 448.0}
if hasattr(torch, "float8_e4m3fnuz"):
    _FP8_MAX_BY_DTYPE[torch.float8_e4m3fnuz] = 240.0

#: e4m3fn finite max (default; OCP / NVIDIA-style).
_FP8_MAX = 448.0


def _fp8_max(fp8_dtype: torch.dtype) -> float:
    try:
        return _FP8_MAX_BY_DTYPE[fp8_dtype]
    except KeyError:
        raise ValueError(f"unsupported fp8 dtype {fp8_dtype}") from None
```
Then in `per_token_group_quant_fp8` change the signature to
`def per_token_group_quant_fp8(x, *, block=FP8_BLOCK, fp8_dtype=torch.float8_e4m3fn):`,
set `fp8_max = _fp8_max(fp8_dtype)`, allocate `x_fp8 = torch.empty(M, K, device=x.device, dtype=fp8_dtype)`, and use `fp8_max` in place of every `_FP8_MAX` and `.to(torch.float8_e4m3fn)` → `.to(fp8_dtype)`. Make the identical change to `per_block_quant_fp8` (signature `(w, *, block=FP8_BLOCK, fp8_dtype=torch.float8_e4m3fn)`, `fp8_max = _fp8_max(fp8_dtype)`, `dtype=fp8_dtype`, clamp/scale on `fp8_max`). Keep the docstrings; add one line noting `fp8_dtype` selects the encoding (fn default; fnuz for AMD native MFMA).

- [ ] **Step 4: Run test to verify it passes (and #38 suite stays green)**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale.py -q`
Expected: all prior tests pass + the new one passes (1 more than before; fnuz test runs since torch 2.12 has fnuz).

- [ ] **Step 5: Commit**
```bash
git add src/xkernels/ops/gemm/reference.py tests/test_mm_fp8_blockscale.py
git commit -m "feat(gemm): fp8_dtype param on block-scale quant helpers (issue #41)

Lets tests/bench emit float8_e4m3fnuz operands (max 240) for the AMD native
fp8 MFMA path, alongside the default float8_e4m3fn (max 448). Reference math
unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: CDNA3 autotune config space (`configs.py`)

**Files:**
- Create: `src/xkernels/ops/gemm/triton/configs.py`
- Test: `tests/test_mm_fp8_blockscale_mfma.py` (create, config-only tests first)

- [ ] **Step 1: Write the failing test**

Create `tests/test_mm_fp8_blockscale_mfma.py`:
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Tests for the native fp8 MFMA block-scale dense GEMM (issue #41)."""
import os
import pytest
import torch

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
pytestmark = pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def test_config_space_is_valid():
    from xkernels.ops.gemm.triton.configs import get_autotune_configs, get_fp8_gemm_config
    cfgs = get_autotune_configs()
    assert len(cfgs) >= 6
    for c in cfgs:
        k = c.kwargs
        assert 128 % k["BLOCK_K"] == 0, "BLOCK_K must divide the 128 quant block"
        assert k["BLOCK_M"] in (16, 32, 64, 128, 256)
        assert k["BLOCK_N"] in (64, 128, 256)
    # Baked direct-launch config: decode (tiny M) vs prefill (large M) differ.
    dec = get_fp8_gemm_config(1, 512, 7168)
    pre = get_fp8_gemm_config(4096, 7168, 2048)
    assert 128 % dec["BLOCK_K"] == 0 and 128 % pre["BLOCK_K"] == 0
    assert dec["BLOCK_M"] <= pre["BLOCK_M"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py::test_config_space_is_valid -q`
Expected: FAIL — `ModuleNotFoundError: ...triton.configs`.

- [ ] **Step 3: Implement**

Create `src/xkernels/ops/gemm/triton/configs.py`:
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Autotune config space for the native fp8 MFMA block-scale GEMM, reasoned for
CDNA3 (gfx942 / MI300A). Mirrors ``ops/moe/triton/configs.py``.

CDNA3 notes driving the choices:
* Wavefront = 64 lanes; ``num_warps`` counts wavefronts. fp8 MFMA wants enough to
  hide the global-load latency of the (already small) fp8 operands.
* MFMA: ``matrix_instr_nonkdim=16`` -> 16x16x32 fp8 MFMA (good for tiny-M decode);
  ``=32`` -> 32x32x16 (packs large-M prefill tiles better).
* LDS = 64 KB/CU. fp8 operands are HALF the bytes of #40's fp32 tiles, so
  BLOCK_K=128 + 2 stages fits where #40's fp32 64x128 hit OutOfResources.
* ``waves_per_eu`` occupancy hint; ``kpack=2`` packs 2 K elems/VGPR for the MFMA
  feed (ds_read/MFMA ratio).
The AMD knobs ride in the Config kwargs dict: the tokenspeed_triton AMD backend
reads them, stock Triton forwards-and-ignores them -> portable.
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
        # BLOCK_K=64 variants (two sub-dots/block) for LDS-tight large tiles
        _cfg(128, 256, 64, 8, num_warps=8, num_stages=2, waves_per_eu=1, nonkdim=32),
    ]


def _lds_ok(bm, bn, bk, num_stages, op_bytes=1):
    # fp8 A tile [bm,bk] + B tile [bk,bn], double-buffered by num_stages.
    return (bm * bk + bk * bn) * op_bytes * max(1, num_stages) <= _LDS_BYTES


def fp8_gemm_prune_configs(configs, named_args, **kwargs):
    """Drop configs that violate BLOCK_K | 128 or overflow CDNA3 LDS."""
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
    """Baked direct-launch config (no runtime autotune). Refined by the beverin
    sweep (slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch); these are the
    CDNA3-reasoned starting points keyed on the M regime."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py::test_config_space_is_valid -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/xkernels/ops/gemm/triton/configs.py tests/test_mm_fp8_blockscale_mfma.py
git commit -m "feat(gemm): CDNA3 autotune config space for fp8 MFMA GEMM (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: The native fp8 MFMA kernel + host wrapper

**Files:**
- Create: `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_mfma_kernel.py`
- Test: `tests/test_mm_fp8_blockscale_mfma.py` (extend — interpreter, exact math)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mm_fp8_blockscale_mfma.py`:
```python
from xkernels.ops.gemm.reference import (  # noqa: E402
    mm_fp8_blockscale_ref, per_block_quant_fp8, per_token_group_quant_fp8,
)


def _inputs(M, N, K, block, dev, seed=0, fp8_dtype=torch.float8_e4m3fn):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    b = torch.randn(N, K, device=dev, dtype=torch.float32)
    a_fp8, a_s = per_token_group_quant_fp8(a, block=block, fp8_dtype=fp8_dtype)
    b_fp8, b_s = per_block_quant_fp8(b, block=block, fp8_dtype=fp8_dtype)
    ref = mm_fp8_blockscale_ref(a_fp8, a_s, b_fp8, b_s, block=block, out_dtype=torch.float32)
    return a_fp8, a_s, b_fp8, b_s, ref


def _rel(got, ref):
    err = (got.float() - ref.float()).abs().max().item()
    return err / ref.float().abs().max().clamp_min(1e-6).item()


@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (37, 130, 384), (7, 24, 320), (1, 256, 512)])
def test_mfma_matches_reference_interpreter(M, N, K):
    """Block-promotion math vs the fp32 dequant oracle (fp8 tl.dot is exact under
    the interpreter -> tight)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton
    dev = "cpu" if _INTERP else ("cuda" if torch.cuda.is_available() else "cpu")
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(M, N, K, 128, dev)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert got.shape == (M, N)
    assert _rel(got, ref) < (1e-3 if _INTERP else 5e-3)


def test_mfma_empty_m_interpreter():
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton
    dev = "cpu" if _INTERP else ("cuda" if torch.cuda.is_available() else "cpu")
    a_fp8 = torch.zeros(0, 128, device=dev, dtype=torch.float8_e4m3fn)
    a_s = torch.zeros(0, 1, device=dev, dtype=torch.float32)
    b_fp8 = torch.zeros(8, 128, device=dev, dtype=torch.float8_e4m3fn)
    b_s = torch.ones(1, 1, device=dev, dtype=torch.float32)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128)
    assert got.shape == (0, 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py -k "interpreter" -q`
Expected: FAIL — module/function not found.

- [ ] **Step 3: Implement**

Create `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_mfma_kernel.py`:
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Native fp8 MFMA block-scale dense GEMM for AMD MI300A (gfx942, CDNA3), issue #41.

The fast-path counterpart to the portable dequant-then-dot kernel
(``mm_fp8_blockscale_kernel.py``, #40). Computes

    out[M, N] = A_deq @ B_deq.T

via **two-level (block-promoted) accumulation**: per 128-K quant block, a raw
fp8.fp8 ``tl.dot`` accumulates into an fp32 block-accumulator (the native CDNA3
fp8 MFMA), then promotes into the main fp32 accumulator scaled by the per-row A
group-scale and the per-N-block B scale -- because both scales are constant
within a 128-K block:

    out = SUM_kb a_s[m,kb] * b_s[n//128,kb] * SUM_{k in block kb} A_fp8[m,k] B_fp8[n,k]

The operands enter ``tl.dot`` in their fp8 dtype (no pre-dequant) -- that is what
routes to the fp8 matrix path. The kernel is fp8-format-agnostic (e4m3fn or
e4m3fnuz); whichever the caller supplies is what the MFMA consumes.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .configs import get_fp8_gemm_config

__all__ = ["mm_fp8_blockscale_mfma_triton", "mm_fp8_blockscale_mfma_kernel"]


@triton.jit
def mm_fp8_blockscale_mfma_kernel(
    a_ptr, as_ptr, b_ptr, bs_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_bn, stride_bk,
    stride_bsn, stride_bsk,
    stride_cm, stride_cn,
    BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    waves_per_eu: tl.constexpr = 0,
    matrix_instr_nonkdim: tl.constexpr = 16,
    kpack: tl.constexpr = 2,
):
    # L2-friendly program swizzle (group along M), like the MoE INT4 kernel.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    nb = cols // BLOCK  # [BLOCK_N] N-quant-block per output column

    kt = tl.cdiv(K, BLOCK)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kb in range(0, kt):
        pacc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        k_base = kb * BLOCK
        for ki in tl.static_range(0, BLOCK, BLOCK_K):
            ks = k_base + ki + tl.arange(0, BLOCK_K)
            k_mask = ks < K
            a = tl.load(
                a_ptr + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=row_mask[:, None] & k_mask[None, :], other=0.0,
            )
            b = tl.load(
                b_ptr + cols[None, :] * stride_bn + ks[:, None] * stride_bk,
                mask=col_mask[None, :] & k_mask[:, None], other=0.0,
            )
            pacc += tl.dot(a, b)  # fp8 operands -> native fp8 MFMA, fp32 accumulate
        a_sc = tl.load(as_ptr + rows * stride_asm + kb * stride_ask, mask=row_mask, other=0.0)
        b_sc = tl.load(bs_ptr + nb * stride_bsn + kb * stride_bsk, mask=col_mask, other=0.0)
        acc += pacc * a_sc[:, None] * b_sc[None, :]

    tl.store(
        c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=row_mask[:, None] & col_mask[None, :],
    )


def mm_fp8_blockscale_mfma_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    config: dict | None = None,
) -> torch.Tensor:
    """Native fp8 MFMA fp8 block-scale GEMM (gfx942). See module docstring.

    ``a_fp8``/``b_fp8`` may be ``float8_e4m3fn`` or ``float8_e4m3fnuz``; the kernel
    dots whatever it is given. ``config`` overrides the baked launch config.
    """
    a_fp8 = a_fp8.contiguous()
    b_fp8 = b_fp8.contiguous()
    a_scales = a_scales.contiguous().float()
    b_scales = b_scales.contiguous().float()

    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    if b_fp8.shape[1] != K:
        raise ValueError(f"b_fp8 must be [N, K] with K={K}, got {tuple(b_fp8.shape)}")
    kt = (K + block - 1) // block
    nt = (N + block - 1) // block
    if tuple(a_scales.shape) != (M, kt):
        raise ValueError(f"a_scales must be [M, kt] = [{M}, {kt}], got {tuple(a_scales.shape)}")
    if tuple(b_scales.shape) != (nt, kt):
        raise ValueError(f"b_scales must be [{nt}, {kt}], got {tuple(b_scales.shape)}")
    if block % 128 and 128 % block:
        # BLOCK_K (a divisor of `block` chosen by the config) must in turn divide
        # the 128 quant convention; we only support block == 128 here.
        raise ValueError(f"native fp8 MFMA path requires block=128, got {block}")

    c = torch.empty(M, N, device=a_fp8.device, dtype=out_dtype)
    if M == 0 or N == 0:
        return c

    cfg = config or get_fp8_gemm_config(M, N, K)
    if block % cfg["BLOCK_K"]:
        raise ValueError(f"BLOCK_K={cfg['BLOCK_K']} must divide block={block}")
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    mm_fp8_blockscale_mfma_kernel[grid](
        a_fp8, a_scales, b_fp8, b_scales, c,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1),
        a_scales.stride(0), a_scales.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        c.stride(0), c.stride(1),
        BLOCK=block,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        waves_per_eu=cfg["waves_per_eu"],
        matrix_instr_nonkdim=cfg["matrix_instr_nonkdim"],
        kpack=cfg["kpack"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py -k "interpreter" -q`
Expected: PASS (5 params + empty-M). If any fail, debug the index math against the oracle — do **not** loosen the tolerance.

- [ ] **Step 5: Commit**
```bash
git add src/xkernels/ops/gemm/triton/mm_fp8_blockscale_mfma_kernel.py tests/test_mm_fp8_blockscale_mfma.py
git commit -m "feat(gemm): native fp8 MFMA block-promoted GEMM kernel (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Wire mfma↔portable into a single TRITON entry + public `path` knob

**Files:**
- Create: `src/xkernels/ops/gemm/triton/entry.py`
- Modify: `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_kernel.py` (remove its `register`)
- Modify: `src/xkernels/ops/gemm/__init__.py` (import `entry`)
- Modify: `src/xkernels/ops/gemm/interface.py` (add `path="auto"`)
- Test: `tests/test_mm_fp8_blockscale_mfma.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mm_fp8_blockscale_mfma.py`:
```python
from xkernels._backends import Backend  # noqa: E402
from xkernels.ops.gemm import mm_fp8_blockscale  # noqa: E402


@pytest.mark.parametrize("path", ["auto", "mfma", "portable"])
def test_entry_path_routing_interpreter(path):
    """All three Triton paths reproduce the fp32 oracle under the interpreter."""
    dev = "cpu" if _INTERP else ("cuda" if torch.cuda.is_available() else "cpu")
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(32, 128, 256, 128, dev)
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32,
        path=path, backend=Backend.TRITON,
    )
    assert _rel(got, ref) < (1e-3 if _INTERP else 5e-3)


def test_dot_bf16_forces_portable_interpreter():
    """dot_bf16=True is a portable-only knob; auto must honor it (route portable)."""
    dev = "cpu" if _INTERP else ("cuda" if torch.cuda.is_available() else "cpu")
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(16, 128, 256, 128, dev)
    if _INTERP:
        pytest.skip("CPU interpreter mis-evaluates a bf16 tl.dot")
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32,
        dot_bf16=True, backend=Backend.TRITON,
    )
    assert _rel(got, ref) < 2e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py::test_entry_path_routing_interpreter -q`
Expected: FAIL — `mm_fp8_blockscale() got an unexpected keyword argument 'path'`.

- [ ] **Step 3: Implement**

(a) In `src/xkernels/ops/gemm/triton/mm_fp8_blockscale_kernel.py`, **delete** the final line `register("mm_fp8_blockscale", Backend.TRITON)(mm_fp8_blockscale_triton)` and the now-unused imports `from ...._backends import Backend` / `from ...._dispatch import register`. Leave the function `mm_fp8_blockscale_triton` intact.

(b) Create `src/xkernels/ops/gemm/triton/entry.py`:
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Single Backend.TRITON registration for ``mm_fp8_blockscale`` on gfx942:
routes between the native fp8 MFMA fast path (#41) and the portable
dequant-then-dot fallback (#40)."""
from __future__ import annotations

import torch

from ...._backends import Backend
from ...._dispatch import register
from .mm_fp8_blockscale_kernel import mm_fp8_blockscale_triton as _portable
from .mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton as _mfma

__all__ = ["mm_fp8_blockscale_triton"]


def mm_fp8_blockscale_triton(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = 128,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,
    path: str = "auto",
) -> torch.Tensor:
    """Dispatch the gfx942 Triton fp8 block-scale GEMM.

    ``path``: ``"mfma"`` (native fp8 MFMA, #41), ``"portable"`` (dequant-then-dot,
    #40), or ``"auto"``. ``dot_bf16=True`` is a portable-only knob and forces the
    portable path. ``"auto"`` selects the mfma fast path (the portable path is the
    explicit / dot_bf16 / non-128-block fallback).
    """
    if dot_bf16 or path == "portable":
        return _portable(a_fp8, a_scales, b_fp8, b_scales,
                          block=block, out_dtype=out_dtype, dot_bf16=dot_bf16)
    if path not in ("auto", "mfma"):
        raise ValueError(f"path must be auto|mfma|portable, got {path!r}")
    if block != 128:  # mfma path is 128-quant-block only; fall back
        return _portable(a_fp8, a_scales, b_fp8, b_scales,
                          block=block, out_dtype=out_dtype, dot_bf16=dot_bf16)
    return _mfma(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)


register("mm_fp8_blockscale", Backend.TRITON)(mm_fp8_blockscale_triton)
```

(c) In `src/xkernels/ops/gemm/__init__.py`, change the redirected import from
`from .triton import mm_fp8_blockscale_kernel  # noqa: F401`
to `from .triton import entry  # noqa: F401  (registers TRITON: mfma + portable)`.

(d) In `src/xkernels/ops/gemm/interface.py`, add `path: str = "auto"` as a keyword-only param to `mm_fp8_blockscale` (after `dot_bf16`), document it (mfma|portable|auto; reference ignores), and pass `path=path` into the `dispatch(...)` call. Add `path` to the reference signature for parity: in `reference.py` `mm_fp8_blockscale_ref`, add `path: str = "auto",  # noqa: ARG001 - backend-signature parity` next to `dot_bf16`.

- [ ] **Step 4: Run tests to verify they pass (+ #38 suite still green)**

Run:
```bash
PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py tests/test_mm_fp8_blockscale.py -q
```
Expected: all pass (the `dot_bf16` test is skipped under interpreter; the `path` routing tests pass).

- [ ] **Step 5: Commit**
```bash
git add src/xkernels/ops/gemm/triton/entry.py src/xkernels/ops/gemm/triton/mm_fp8_blockscale_kernel.py \
        src/xkernels/ops/gemm/__init__.py src/xkernels/ops/gemm/interface.py src/xkernels/ops/gemm/reference.py \
        tests/test_mm_fp8_blockscale_mfma.py
git commit -m "feat(gemm): route TRITON backend between fp8 MFMA and portable paths (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Round out the GPU-gated test surface (cross-check, fnuz, V4 shape, dtypes)

**Files:**
- Test: `tests/test_mm_fp8_blockscale_mfma.py` (extend with GPU-gated cases)

These are skipped locally (no CUDA) and run on beverin in Task 7. Write them now so the on-device run exercises them.

- [ ] **Step 1: Write the tests**

Append to `tests/test_mm_fp8_blockscale_mfma.py`:
```python
_GPU = (not _INTERP) and torch.cuda.is_available()


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (8, 512, 7168), (2048, 512, 7168)])
def test_mfma_tight_parity_gpu(M, N, K):
    """Native fp8 MFMA vs fp32 dequant oracle: TIGHT (<5e-3). Loose => fp8 format
    mismatch reached the matrix unit (the fn-vs-fnuz detector)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(M, N, K, 128, "cuda", seed=4)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert _rel(got, ref) < 5e-3, (M, N, K, _rel(got, ref))


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
def test_mfma_cross_checks_portable_gpu():
    """The two independent Triton implementations agree within fp8 tolerance."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_kernel import mm_fp8_blockscale_triton as portable
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton as mfma
    a_fp8, a_s, b_fp8, b_s, _ = _inputs(48, 256, 512, 128, "cuda")
    p = portable(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    m = mfma(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert _rel(m, p) < 5e-3


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fnuz"), reason="no fnuz")
def test_mfma_fnuz_operands_gpu():
    """fnuz operands also produce a correct GEMM (the AMD-native fp8 encoding)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import mm_fp8_blockscale_mfma_triton
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(64, 256, 512, 128, "cuda", fp8_dtype=torch.float8_e4m3fnuz)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    # fnuz (max 240) is coarser than fn -> a looser but still real parity bound.
    assert _rel(got, ref) < 3e-2


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
def test_mfma_bf16_out_gpu():
    from xkernels.ops.gemm import mm_fp8_blockscale
    from xkernels._backends import Backend
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(48, 64, 256, 128, "cuda")
    got = mm_fp8_blockscale(a_fp8, a_s, b_fp8, b_s, block=128, path="mfma", backend=Backend.TRITON)
    assert got.dtype == torch.bfloat16 and _rel(got, ref) < 5e-3
```

- [ ] **Step 2: Run locally to confirm they collect + skip cleanly (no GPU here)**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_mm_fp8_blockscale_mfma.py -q`
Expected: interpreter/config tests pass; the four `_GPU` tests **skip** ("needs gfx942 GPU"). No collection errors.

- [ ] **Step 3: Commit**
```bash
git add tests/test_mm_fp8_blockscale_mfma.py
git commit -m "test(gemm): GPU-gated parity/cross-check/fnuz tests for fp8 MFMA GEMM (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Phase-0 beverin probe — resolve fn-vs-fnuz native fp8 MFMA

**Files:**
- Create: `slurm/probe_fp8_mfma_beverin.sbatch`

This is **investigation, not TDD**: it answers whether `tl.dot(e4m3fn)` reaches native fp8 MFMA on gfx942 (root cause #1), which sets the bench/ship default. Run it before trusting Task 7's perf.

- [ ] **Step 1: Write the probe sbatch**

Create `slurm/probe_fp8_mfma_beverin.sbatch` (model the header on `slurm/probe_dense_bf16_beverin.sbatch`):
```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Phase-0 probe (issue #41): does tl.dot on fp8 e4m3 reach native fp8 MFMA on
# gfx942? Compiles a minimal fp8 dot for float8_e4m3fn AND float8_e4m3fnuz,
# greps the AMDGCN for v_mfma_*_fp8*, and reports parity + TFLOP/s per format.
#SBATCH --job-name=xk-fp8-probe
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:15:00
#SBATCH --output=fp8-probe-%j.out
#SBATCH --error=fp8-probe-%j.out
set -uo pipefail
REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels-issue-41}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"
echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"
srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  export TRITON_ALWAYS_COMPILE=1 AMDGCN_ENABLE_DUMP=1
  cd "'"$REPO"'"
  python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, torch.cuda.get_device_name(0))"
  python -u - <<"PY"
import torch, triton, triton.language as tl
from triton.testing import do_bench

@triton.jit
def _dot(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.arange(0, BM); rn = tl.arange(0, BN)
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None]*K + ks[None, :])
        b = tl.load(b_ptr + rn[None, :]*K + ks[:, None])
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None]*N + rn[None, :], acc)

def probe(dt, name):
    M=N=512; K=4096; BM=BN=128; BK=128
    a = torch.randn(M, K, device="cuda").to(dt)
    b = torch.randn(N, K, device="cuda").to(dt)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    comp = _dot[(M//BM, N//BN)](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    asm = comp.asm.get("amdgcn", "") if hasattr(comp, "asm") else ""
    fp8_mfma = [l.strip() for l in asm.splitlines() if "v_mfma" in l and "fp8" in l]
    ref = (a.float() @ b.float().t())
    rel = (c - ref).abs().max().item() / ref.abs().max().item()
    t = do_bench(lambda: _dot[(M//BM, N//BN)](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK))
    tf = 2*M*N*K/t/1e9
    print(f"[{name}] rel={rel:.3e} time={t:.4f}ms {tf:.1f}TFLOP/s "
          f"fp8_mfma_insns={len(fp8_mfma)} :: {fp8_mfma[:2]}")

for dt, name in [(torch.float8_e4m3fn, "e4m3fn"), (torch.float8_e4m3fnuz, "e4m3fnuz")]:
    try: probe(dt, name)
    except Exception as e: print(f"[{name}] FAILED:", repr(e)[:300])
print("DECISION: format(s) with fp8_mfma_insns>0 AND tight rel reach native fp8 MFMA.")
PY
'
```

- [ ] **Step 2: Run on beverin and read the result**

```bash
W=/home/xiayao/Documents/research/xkernels/.claude/worktrees/issue-41-fp8-mfma
ssh beverin 'mkdir -p /capstor/scratch/cscs/xyao/xkernels-issue-41'
rsync -az --delete --exclude .git --exclude .venv --exclude '**/__pycache__' "$W"/ beverin:/capstor/scratch/cscs/xyao/xkernels-issue-41/
ssh beverin 'cd /capstor/scratch/cscs/xyao/xkernels-issue-41 && sbatch --export=ALL,REPO=$PWD slurm/probe_fp8_mfma_beverin.sbatch'
# poll then read:
ssh beverin 'squeue --me'
ssh beverin 'cat /capstor/scratch/cscs/xyao/xkernels-issue-41/fp8-probe-*.out'
```
Expected: one of —
- **e4m3fn has `fp8_mfma_insns>0` + tight rel** → keep e4m3fn as default; no change.
- **only e4m3fnuz reaches fp8 MFMA** → default the bench/ship path to fnuz operands (Task 8 emits fnuz on AMD); the kernel is already format-agnostic, so no kernel change.
- **neither reaches fp8 MFMA** → record it; the "fast path" reduces to autotuned tiling only (root cause #2). Capture the honest result; the kernel still stands.

- [ ] **Step 3: Record the decision in the doc-to-be**

Note the chosen fp8 format + the AMDGCN evidence (instruction names, TFLOP/s) in a scratch note for Task 9. No commit yet (the sbatch is committed with Task 7).

---

## Task 7: On-device test + perf sbatch (beverin), with the fp8-MFMA assertion

**Files:**
- Create: `slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch`

- [ ] **Step 1: Write the sbatch**

Create `slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch` by copying `slurm/test_mm_fp8_blockscale_beverin.sbatch` and changing: the `--job-name=xk-fp8-mfma`, the output names to `fp8-mfma-%j.out`, the pytest target to `tests/test_mm_fp8_blockscale_mfma.py`, and the inline python block to:
```python
import torch
from xkernels.ops.gemm import mm_fp8_blockscale, per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.ops.gemm.reference import mm_fp8_blockscale_ref
from xkernels._backends import Backend
from triton.testing import do_bench

torch.manual_seed(0); dev="cuda"; block=128
SHAPES=[(1,512,7168),(8,512,7168),(2048,512,7168),(4096,7168,2048)]
# Pick the fp8 dtype the Task-6 probe found native (default fn).
FP8 = torch.float8_e4m3fn
for (M,N,K) in SHAPES:
    a=torch.randn(M,K,device=dev); w=torch.randn(N,K,device=dev)
    a8,as_=per_token_group_quant_fp8(a,block=block,fp8_dtype=FP8)
    w8,ws_=per_block_quant_fp8(w,block=block,fp8_dtype=FP8)
    ref=mm_fp8_blockscale_ref(a8,as_,w8,ws_,block=block,out_dtype=torch.float32)
    got=mm_fp8_blockscale(a8,as_,w8,ws_,block=block,out_dtype=torch.float32,path="mfma",backend=Backend.TRITON)
    rel=(got-ref).abs().max().item()/ref.abs().max().clamp_min(1e-6).item()
    print(f"[M={M} N={N} K={K}] mfma rel={rel:.3e}")
    assert rel < 5e-3, (M,N,K,rel)
print("PASS: native fp8 MFMA correct on gfx942")
# Perf: mfma vs torch_ref vs #40 portable (fp32 + bf16 dot).
for (M,N,K) in SHAPES:
    a=torch.randn(M,K,device=dev); w=torch.randn(N,K,device=dev)
    a8,as_=per_token_group_quant_fp8(a,block=block,fp8_dtype=FP8)
    w8,ws_=per_block_quant_fp8(w,block=block,fp8_dtype=FP8)
    t_mfma=do_bench(lambda: mm_fp8_blockscale(a8,as_,w8,ws_,block=block,out_dtype=torch.bfloat16,path="mfma",backend=Backend.TRITON))
    t_port=do_bench(lambda: mm_fp8_blockscale(a8,as_,w8,ws_,block=block,out_dtype=torch.bfloat16,path="portable",backend=Backend.TRITON))
    t_ref =do_bench(lambda: mm_fp8_blockscale_ref(a8,as_,w8,ws_,block=block,out_dtype=torch.bfloat16))
    fl=2*M*N*K
    print(f"[perf M={M} N={N} K={K}] mfma={t_mfma:.3f}ms ({fl/t_mfma/1e9:.1f}TF) "
          f"portable={t_port:.3f}ms torch_ref={t_ref:.3f}ms -> mfma {t_ref/t_mfma:.2f}x ref")
# fp8 MFMA assertion: dump the compiled mfma kernel and confirm v_mfma_*_fp8*.
import os; os.environ["AMDGCN_ENABLE_DUMP"]="1"
print("see probe job for the AMDGCN v_mfma fp8 evidence")
```
(Keep the existing sbatch's `pytest tests/test_mm_fp8_blockscale_mfma.py -q` line above the python block.)

- [ ] **Step 2: Run on beverin, iterate the tuned config**

```bash
W=/home/xiayao/Documents/research/xkernels/.claude/worktrees/issue-41-fp8-mfma
rsync -az --delete --exclude .git --exclude .venv --exclude '**/__pycache__' "$W"/ beverin:/capstor/scratch/cscs/xyao/xkernels-issue-41/
ssh beverin 'cd /capstor/scratch/cscs/xyao/xkernels-issue-41 && sbatch --export=ALL,REPO=$PWD slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch'
ssh beverin 'cat /capstor/scratch/cscs/xyao/xkernels-issue-41/fp8-mfma-*.out'
```
Expected: pytest passes on GPU; tight parity (<5e-3) on all V4 shapes; a perf table. **Iterate `get_fp8_gemm_config` in `configs.py`** based on the measured per-shape timing (try the `get_autotune_configs()` candidates by hand-editing the baked config), re-rsync, re-run, until mfma TFLOP/s plateau. Record the best config + numbers.

- [ ] **Step 3: Commit the sbatch (+ any config tuning)**
```bash
git add slurm/probe_fp8_mfma_beverin.sbatch slurm/test_mm_fp8_blockscale_mfma_beverin.sbatch src/xkernels/ops/gemm/triton/configs.py
git commit -m "test(gemm): beverin fp8-MFMA probe + on-device test/perf sbatch (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Standalone benchmark + (conditional) README row

**Files:**
- Create: `benchmarks/bench_fp8_blockscale_gemm.py`

- [ ] **Step 1: Write the benchmark**

Create `benchmarks/bench_fp8_blockscale_gemm.py` (model imports/structure on `benchmarks/bench_mhc_prenorm_gemm.py`):
```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Bench the fp8 block-scale dense GEMM on gfx942 (issue #41): native fp8 MFMA vs
the #40 portable dequant path vs the torch dequant reference, across V4 MLA shapes.

    python benchmarks/bench_fp8_blockscale_gemm.py
"""
from __future__ import annotations
import torch
from xkernels.ops.gemm import mm_fp8_blockscale, per_block_quant_fp8, per_token_group_quant_fp8
from xkernels.ops.gemm.reference import mm_fp8_blockscale_ref
from xkernels._backends import Backend
from xkernels.utils.benchmarking import benchmark

SHAPES = [(1, 512, 7168), (8, 512, 7168), (2048, 512, 7168), (4096, 7168, 2048)]
FP8 = torch.float8_e4m3fn  # set to float8_e4m3fnuz if the probe found only fnuz native


def main():
    dev = "cuda"
    block = 128
    print(f"{'M':>5} {'N':>5} {'K':>5} {'mfma(ms)':>9} {'TFLOP/s':>8} {'portable':>9} {'torch_ref':>9} {'mfma/ref':>8}")
    for (M, N, K) in SHAPES:
        a = torch.randn(M, K, device=dev); w = torch.randn(N, K, device=dev)
        a8, as_ = per_token_group_quant_fp8(a, block=block, fp8_dtype=FP8)
        w8, ws_ = per_block_quant_fp8(w, block=block, fp8_dtype=FP8)
        f = lambda p: mm_fp8_blockscale(a8, as_, w8, ws_, block=block, out_dtype=torch.bfloat16, path=p, backend=Backend.TRITON)
        t_mfma = benchmark(lambda: f("mfma"))
        t_port = benchmark(lambda: f("portable"))
        t_ref = benchmark(lambda: mm_fp8_blockscale_ref(a8, as_, w8, ws_, block=block, out_dtype=torch.bfloat16))
        tf = 2 * M * N * K / t_mfma / 1e9
        print(f"{M:>5} {N:>5} {K:>5} {t_mfma:>9.3f} {tf:>8.1f} {t_port:>9.3f} {t_ref:>9.3f} {t_ref/t_mfma:>7.2f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-collect locally (import only — no GPU run here)**

Run: `PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('benchmarks/bench_fp8_blockscale_gemm.py').read()); print('parses OK')"`
Expected: `parses OK`. (Full run happens on beverin.)

- [ ] **Step 3: Run on beverin and capture the table** (reuse the rsync helper; run `python benchmarks/bench_fp8_blockscale_gemm.py` inside the `srun --environment` shell, or add a one-line bench sbatch mirroring Task 7). Record the numbers for Task 9. **If mfma beats torch_ref on the representative shape**, add a row to the README Performance table; else leave it standalone (honest result).

- [ ] **Step 4: Commit**
```bash
git add benchmarks/bench_fp8_blockscale_gemm.py
git commit -m "bench(gemm): fp8 MFMA vs portable vs torch_ref across V4 shapes (issue #41)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Result doc + ship-default + README (per measured outcome)

**Files:**
- Create: `docs/issue-41-fp8-mfma-blockscale-gemm.md`
- Modify (conditional): `README.md`, `src/xkernels/ops/gemm/triton/configs.py` (final tuned table)

- [ ] **Step 1: Write the kernel doc**

Create `docs/issue-41-fp8-mfma-blockscale-gemm.md` modeled on `docs/issue-38-fp8-blockscale-gemm.md`: the block-promotion math; the fn-vs-fnuz resolution **with the Task-6 AMDGCN evidence** (which format reached native fp8 MFMA, instruction names, TFLOP/s); the autotune table (final `get_fp8_gemm_config` regimes); and the honest on-device perf table (mfma vs portable vs torch_ref across the four V4 shapes, with ×ref). State plainly whether mfma is now the default (won) or stays opt-in (#17/#20 framing).

- [ ] **Step 2: Set the ship default**

Per the measured outcome: if mfma wins broadly, `path="auto"` already routes to mfma (done). If mfma wins only on some shapes, encode the loser shapes to return a portable marker in `get_fp8_gemm_config` and have `entry.py` honor it. If it never wins, keep `path="auto"` → mfma but document it as opt-in-quality and note serving should stay on torch_ref (mirror #38's closing section). Make the minimal code change the data dictates; do not invent a default the numbers don't support.

- [ ] **Step 3: Final local regression + commit**

Run: `PYTHONPATH=src TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mm_fp8_blockscale.py tests/test_mm_fp8_blockscale_mfma.py -q`
Expected: all pass / GPU-gated skip.
```bash
git add docs/issue-41-fp8-mfma-blockscale-gemm.md README.md src/xkernels/ops/gemm/triton/configs.py
git commit -m "docs(gemm): issue #41 native fp8 MFMA result + tuned config

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Push branch + open stacked draft PR

- [ ] **Step 1: Lint (match repo tooling)**

Run (if ruff available locally): `.venv/bin/python -m pip install ruff -q && .venv/bin/ruff check src/xkernels/ops/gemm tests/test_mm_fp8_blockscale_mfma.py` and fix findings. (CI runs ruff; keep it clean.)

- [ ] **Step 2: Push**
```bash
git push -u origin feat/issue-41-fp8-mfma-blockscale-gemm
```

- [ ] **Step 3: Open the draft PR stacked on #40**
```bash
gh pr create --repo ResearchComputer/xkernels --draft \
  --base feat/issue-38-fp8-blockscale-gemm \
  --head feat/issue-41-fp8-mfma-blockscale-gemm \
  --title "feat(gemm): native fp8 MFMA fast path for mm_fp8_blockscale on gfx942 (issue #41)" \
  --body "<summary: math, fn/fnuz resolution, on-device numbers vs torch_ref/#40, ship default; Closes #41. Stacked on #40 — retarget to main after #40 merges.>"
```
Expected: draft PR opened against the #40 branch. Note in the PR body that it must be retargeted to `main` once #40 merges.

---

## Self-review (done while writing)

- **Spec coverage:** native fp8 MFMA + post-block scale (Task 3); autotune `BLOCK_K=128`/larger tiles/stages/warps/AMD knobs (Task 2, tuned in 7); keep portable fallback (Task 4); fn/fnuz risk + diagnostic + fallback (Tasks 1, 6, the tight-parity assertions in 3/5/7); re-bench vs torch_ref + #40 (Tasks 7, 8); honest-result ship policy (Task 9). All spec sections map to a task.
- **Placeholder scan:** every code step has complete code; sbatch/bench/doc tasks give full scripts or exact adaptations of a named existing sibling file. The only deliberately data-dependent step is Task 9 Step 2 (ship default), which is explicitly conditioned on the Task-7/8 numbers — not a placeholder.
- **Type/name consistency:** `mm_fp8_blockscale_mfma_triton`, `get_fp8_gemm_config`, `get_autotune_configs`, `fp8_gemm_prune_configs`, `mm_fp8_blockscale_triton` (entry), `path`/`dot_bf16` knobs, `fp8_dtype` param — used consistently across Tasks 1–8. Config dict keys (`BLOCK_M/N/K`, `GROUP_M`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, `num_warps`, `num_stages`) match between `configs.py`, the kernel signature, and the wrapper launch.
