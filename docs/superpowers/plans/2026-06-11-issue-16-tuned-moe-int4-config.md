# Tuned INT4 W4A16 fused-MoE config (issue #16) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and check in tuned Triton configs for the two Kimi-K2.6 INT4 W4A16 fused-MoE GEMM shapes on gfx942, with a loader that uses them at launch with zero runtime autotune (so the "default config" warning disappears in production), falling back to the existing autotune for untuned shapes.

**Architecture:** Factor the kernel body into a plain `@triton.jit` function. The production launch (`int4_w4a16_moe_gemm`) resolves a config from a checked-in JSON (`get_moe_int4_config`, vLLM-style file per `(E,N,K,device,dtype)`) and launches the jit directly with explicit meta-params. Untuned shapes keep the `@triton.autotune`-wrapped entry point. The token-slot alignment block (`moe_align_block_size`) is made to match the chosen `BLOCK_SIZE_M` (a correctness invariant the kernel requires). An offline `do_bench` harness sweeps the pruned config space on-device and persists winners.

**Tech Stack:** Python 3.11, Triton 3.7 (ROCm fork on device), PyTorch, pytest (`TRITON_INTERPRET=1` for CPU correctness), SLURM/enroot on CSCS beverin (MI300A, gfx942).

**Local commands:** run Python via `.venv/bin/python`; tests via `TRITON_INTERPRET=1 .venv/bin/python -m pytest ...`; lint via `.venv/bin/ruff check .`.

---

## File structure

- **Create** `src/xkernels/ops/moe/triton/tuned_configs/` — dir holding the checked-in JSON winners (populated by the on-device run in Task 6).
- **Modify** `src/xkernels/ops/moe/triton/configs.py` — add `align_block_m`, device-name + JSON loader, bucket selection, `get_moe_int4_config`; tighten `prune_configs` to enforce `BLOCK_SIZE_M == align block`.
- **Modify** `src/xkernels/ops/moe/triton/moe_int4_kernel.py` — split jit body, add direct tuned-launch path + `config=` arg, resolve config in the registered wrapper.
- **Create** `benchmarks/tune_moe_int4_w4a16.py` — offline tuner that writes the JSON.
- **Create** `slurm/tune_moe_int4_beverin.sbatch` — runs the tuner on the `mi300` partition.
- **Create** `tests/test_moe_int4_tuned_config.py` — loader/selection unit tests (skip without triton).
- **Modify** `tests/test_moe_int4_w4a16.py` — add a tuned-path-matches-reference integration test.
- **Modify** `pyproject.toml` — add `ops/**/*.json` to `package-data`.
- **Modify** `README.md` — update the `moe_int4_w4a16` perf note with tuned figures (Task 6).

---

### Task 1: Config loader, device name, bucket selection, alignment helper

**Files:**
- Modify: `src/xkernels/ops/moe/triton/configs.py`
- Test: `tests/test_moe_int4_tuned_config.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_moe_int4_tuned_config.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Unit tests for the tuned INT4 W4A16 MoE config loader + selection (issue #16).

Pure-Python (no GPU). Skipped where Triton is absent, because the config module
imports Triton for the autotune ``Config`` builders.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("triton")

from xkernels.ops.moe.triton import configs as C  # noqa: E402


def test_align_block_m():
    assert C.align_block_m(1) == 16
    assert C.align_block_m(16) == 16
    assert C.align_block_m(32) == 16
    assert C.align_block_m(33) == 64
    assert C.align_block_m(4096) == 64


def _table():
    return {
        "_provenance": {"device": "X"},
        "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
              "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2, "_ms": 0.01},
        "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
        "64": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
               "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2},
    }


def test_select_exact_bucket():
    cfg = C._select_config(_table(), 8)
    assert cfg["BLOCK_SIZE_N"] == 128 and cfg["num_warps"] == 4


def test_select_closest_below():
    assert C._select_config(_table(), 5)["BLOCK_SIZE_N"] == 64    # -> bucket 1
    assert C._select_config(_table(), 40)["num_warps"] == 4       # -> bucket 8


def test_select_clamps():
    assert C._select_config(_table(), 0)["BLOCK_SIZE_N"] == 64    # below min -> min
    assert C._select_config(_table(), 100000)["BLOCK_SIZE_M"] == 64  # above max -> max


def test_select_strips_metadata_keys():
    assert "_ms" not in C._select_config(_table(), 1)


def test_get_config_missing_returns_none():
    assert C.get_moe_int4_config(48, 1, 1, 1, arch="No_Such_Device") is None


def test_loader_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_config_dir", lambda: str(tmp_path))
    C._TUNED_CACHE.clear()
    fname = C._config_filename(48, 4096, 7168, "Test_Dev", "int4_w4a16")
    (tmp_path / fname).write_text(json.dumps(_table()))
    cfg = C.get_moe_int4_config(48, 4096, 7168, 8, arch="Test Dev")
    assert cfg is not None and cfg["num_warps"] == 4
    C._TUNED_CACHE.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_tuned_config.py -v`
Expected: FAIL/ERROR — `AttributeError: module ... has no attribute 'align_block_m'` (functions not yet defined).

- [ ] **Step 3: Implement the loader in `configs.py`**

At the top of `src/xkernels/ops/moe/triton/configs.py`, after `import triton`, add the new imports:

```python
import json
import os
import warnings
```

Replace the `__all__` line with:

```python
__all__ = [
    "get_autotune_configs",
    "prune_configs",
    "align_block_m",
    "get_moe_int4_config",
    "load_tuned_config",
]
```

Append to the end of the file:

```python
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


def get_moe_int4_config(E: int, N: int, K: int, M: int, dtype: str = "int4_w4a16", arch: str | None = None):
    """Return the tuned launch config for ``(shape, M)`` on this device, or ``None``."""
    device = _device_name(arch)
    if device is None:
        return None
    table = load_tuned_config(E, N, K, device, dtype)
    if not table:
        return None
    return _select_config(table, M)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_tuned_config.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check src/xkernels/ops/moe/triton/configs.py tests/test_moe_int4_tuned_config.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/xkernels/ops/moe/triton/configs.py tests/test_moe_int4_tuned_config.py
git commit -m "feat(moe): tuned INT4 MoE config loader + alignment helper (issue #16)"
```

---

### Task 2: Enforce alignment in `prune_configs`

**Files:**
- Modify: `src/xkernels/ops/moe/triton/configs.py:110-134` (the `prune_configs` function)

**Why:** The grouped GEMM requires the autotuned `BLOCK_SIZE_M` to equal the alignment block used by `moe_align_block_size`. The current space mixes `BLOCK_SIZE_M` ∈ {16,32,64,128,256}; on the fallback path the wrapper aligns with `align_block_m(M)` (16 or 64), so autotune must only consider matching-`BLOCK_SIZE_M` configs.

- [ ] **Step 1: Replace `prune_configs`**

Replace the entire `prune_configs` function body with:

```python
def prune_configs(configs, named_args, **kwargs):
    """Drop configs that cannot run (or would mis-align) for the given problem.

    Removes configs whose ``BLOCK_SIZE_K`` is not a multiple of the quant group
    size or the pack factor (8), over-large ``BLOCK_SIZE_N`` for tiny ``N``, and
    — when the token count is known — configs whose ``BLOCK_SIZE_M`` does not
    equal ``align_block_m(M)``. The last guard keeps the fallback autotune path
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
```

- [ ] **Step 2: Run the existing INT4 MoE tests (interpreter) to confirm no regression**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_w4a16.py -v`
Expected: PASS (same as before; pinned config has `BLOCK_SIZE_M=16`, all test M ≤ 16 → `align_block_m=16` → kept).

- [ ] **Step 3: Lint + commit**

```bash
.venv/bin/ruff check src/xkernels/ops/moe/triton/configs.py
git add src/xkernels/ops/moe/triton/configs.py
git commit -m "fix(moe): prune autotune configs to the alignment BLOCK_SIZE_M (issue #16)"
```

---

### Task 3: Split the kernel body and add the direct tuned-launch path

**Files:**
- Modify: `src/xkernels/ops/moe/triton/moe_int4_kernel.py`

- [ ] **Step 1: Update the imports**

Change the configs import line (currently `from .configs import get_autotune_configs, prune_configs`) to:

```python
from .configs import (
    align_block_m,
    get_autotune_configs,
    get_moe_int4_config,
    prune_configs,
)
```

- [ ] **Step 2: Convert the decorated kernel into a plain jit body + explicit autotuned wrapper**

Remove the `@triton.autotune(...)` and `@triton.heuristics(...)` decorators that sit above `@triton.jit def fused_moe_int4_kernel(`. Rename the jit function to `_fused_moe_int4_kernel` (leave its argument list and entire body unchanged). So the decorator stack becomes just:

```python
@triton.jit
def _fused_moe_int4_kernel(
    a_ptr,  # [M, K] bf16 activations (token rows, pre-permute)
    ...  # (unchanged signature and body)
```

Immediately **after** the function body (after the final `tl.store(...)` of the kernel, before `def int4_w4a16_moe_gemm`), add the autotuned wrapper that preserves the old name:

```python
# Autotuned entry point (unchanged name): used for untuned shapes and by the
# offline tuner. Built explicitly from the jit body so the production launch can
# also call _fused_moe_int4_kernel directly with a resolved config.
fused_moe_int4_kernel = triton.autotune(
    configs=get_autotune_configs(),
    key=["N", "K", "EM", "num_valid_tokens"],
    prune_configs_by={"early_config_prune": prune_configs},
)(
    triton.heuristics(
        {"EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0}
    )(_fused_moe_int4_kernel)
)
```

- [ ] **Step 3: Add `config=` and the direct launch to `int4_w4a16_moe_gemm`**

In `int4_w4a16_moe_gemm`, add a keyword-only `config: dict | None = None` parameter at the end of the signature (after `filter_expert: bool = True`). Then replace the body from the `def grid(meta):` block through the `fused_moe_int4_kernel[grid](...)` call (the launch) with:

```python
    common = (
        a,
        b_packed,
        c,
        b_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b_packed.stride(0),
        b_packed.stride(1),
        b_packed.stride(2),
        c.stride(0),
        c.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        b_scale.stride(2),
    )
    common_kw = dict(
        group_k=group_size,
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        compute_type=compute_type,
        FILTER_EXPERT=filter_expert,
    )

    if config is not None:
        bm = config["BLOCK_SIZE_M"]
        bn = config["BLOCK_SIZE_N"]
        grid = (triton.cdiv(sorted_token_ids.shape[0], bm) * triton.cdiv(N, bn),)
        _fused_moe_int4_kernel[grid](
            *common,
            **common_kw,
            BLOCK_SIZE_M=bm,
            BLOCK_SIZE_N=bn,
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            EVEN_K=(K % config["BLOCK_SIZE_K"] == 0),
            waves_per_eu=config.get("waves_per_eu", 0),
            matrix_instr_nonkdim=config.get("matrix_instr_nonkdim", 16),
            kpack=config.get("kpack", 2),
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        return c

    def grid(meta):
        return (
            triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(N, meta["BLOCK_SIZE_N"]),
        )

    fused_moe_int4_kernel[grid](*common, **common_kw)
    return c
```

- [ ] **Step 4: Resolve the config in the registered wrapper**

Replace the body of `_moe_int4_w4a16_triton` with:

```python
def _moe_int4_w4a16_triton(
    A: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    group_size: int = 32,
    mul_routed_weight: bool = True,
) -> torch.Tensor:
    M, top_k = topk_ids.shape
    E, N, kp = packed.shape
    K = kp * 8
    # Resolve a checked-in tuned config first; the token-slot alignment block
    # MUST equal the kernel BLOCK_SIZE_M (see align_block_m), so derive it from
    # the config when present, else from the decode/prefill M heuristic.
    config = get_moe_int4_config(E, N, K, M)
    block_m = config["BLOCK_SIZE_M"] if config is not None else align_block_m(M)
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    c = torch.zeros((M * top_k, N), dtype=A.dtype, device=A.device)
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float32
    int4_w4a16_moe_gemm(
        A,
        packed,
        scale,
        c,
        topk_w.reshape(-1).float(),
        sorted_ids,
        expert_ids,
        num_post,
        top_k=top_k,
        group_size=group_size,
        mul_routed_weight=mul_routed_weight,
        compute_type=compute_type,
        filter_expert=False,
        config=config,
    )
    return c.view(M, top_k, N).sum(dim=1)
```

- [ ] **Step 5: Run the existing INT4 MoE tests (interpreter)**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_w4a16.py -v`
Expected: PASS — `get_moe_int4_config` returns `None` on CPU (no device) so the fallback autotune path runs exactly as before; `_pin_single_config` still finds `.configs` on `fused_moe_int4_kernel`.

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/xkernels/ops/moe/triton/moe_int4_kernel.py
git add src/xkernels/ops/moe/triton/moe_int4_kernel.py
git commit -m "feat(moe): direct tuned-config launch path for INT4 MoE GEMM (issue #16)"
```

---

### Task 4: Tuned-path integration test (interpreter)

**Files:**
- Modify: `tests/test_moe_int4_w4a16.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_moe_int4_w4a16.py`:

```python
def test_tuned_config_path_matches_reference(monkeypatch):
    """A resolved tuned config drives the direct (non-autotune) launch correctly.

    Monkeypatches ``get_moe_int4_config`` (as imported into the kernel module) to
    return a valid config; the wrapper then aligns to its BLOCK_SIZE_M and takes
    the direct launch path. Output must still match the grouped-MoE oracle.
    """
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered (triton not installed)")
    dev = _device()
    from xkernels.ops.moe.triton import moe_int4_kernel as kmod

    cfg = {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 2,
    }
    monkeypatch.setattr(kmod, "get_moe_int4_config", lambda *a, **k: cfg)

    group_size = 32
    M, E, N, K, top_k = 4, 8, 128, 128, 4
    packed, scale, A, topk_ids, topk_w = _inputs(M, E, N, K, top_k, dev, group_size)
    got = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=True, backend=Backend.TRITON,
    )
    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, True)
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)
```

- [ ] **Step 2: Run to verify it passes**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_w4a16.py::test_tuned_config_path_matches_reference -v`
Expected: PASS — the direct path runs the same math; `BLOCK_SIZE_K=64` is a multiple of group(32) and pack(8); `K=128 % 64 == 0` so `EVEN_K=True`. (If it errors on import of `get_moe_int4_config`, ensure Task 3 Step 1 added it to the kernel module's imports.)

- [ ] **Step 3: Run the full interpreter MoE suite + lint**

Run: `TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_moe_int4_w4a16.py tests/test_moe_int4_tuned_config.py -v`
Expected: all PASS.
Run: `.venv/bin/ruff check tests/test_moe_int4_w4a16.py`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/test_moe_int4_w4a16.py
git commit -m "test(moe): tuned-config direct launch matches reference (issue #16)"
```

---

### Task 5: Offline tuning harness

**Files:**
- Create: `benchmarks/tune_moe_int4_w4a16.py`

- [ ] **Step 1: Write the tuner**

Create `benchmarks/tune_moe_int4_w4a16.py`:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Sweep the INT4 W4A16 fused-MoE autotune space on-device and persist winners.

For each Kimi-K2.6 production shape (gate_up: E=48,N=4096,K=7168; down:
E=48,N=7168,K=2048) and each token-batch M-bucket, time every valid candidate
config via the kernel's *direct* launch path (each candidate aligned to its own
BLOCK_SIZE_M) and keep the fastest. Writes one JSON per (E,N,K,device,dtype)
into the kernel's tuned_configs/ dir, mapping str(M) -> winning config.

Run on real gfx942 (needs a GPU); does NOT submit a cluster job::

    python benchmarks/tune_moe_int4_w4a16.py
    python benchmarks/tune_moe_int4_w4a16.py --M 1 2 4 8 16
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import torch

from xkernels.ops.moe import make_w4a16_weights, moe_align_block_size_ref
from xkernels.ops.moe.triton.configs import (
    _config_dir,
    _config_filename,
    _device_name,
    get_autotune_configs,
    prune_configs,
)

KIMI = dict(E=48, HIDDEN=7168, INTER=2048, TOP_K=8)


def _candidate_configs(N, K, group_size):
    # No num_valid_tokens -> prune does not apply the BLOCK_SIZE_M filter, so all
    # valid BMs are explored; each is benchmarked with matching alignment below.
    return prune_configs(
        get_autotune_configs(), {"group_k": group_size, "N": N, "K": K}
    )


def _bench_config(cfg, a, packed, scale, c, topk_ids, topk_w, top_k, group_size):
    import triton
    import triton.language as tl

    from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

    E = packed.shape[0]
    block_m = cfg.kwargs["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
    launch_cfg = dict(cfg.kwargs)
    launch_cfg["num_warps"] = cfg.num_warps
    launch_cfg["num_stages"] = cfg.num_stages

    def run():
        int4_w4a16_moe_gemm(
            a, packed, scale, c, topk_w, sorted_ids, expert_ids, num_post,
            top_k=top_k, group_size=group_size, mul_routed_weight=False,
            compute_type=tl.bfloat16, filter_expert=False, config=launch_cfg,
        )

    try:
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        return triton.testing.do_bench(run, rep=50)
    except Exception as exc:
        print(f"    skip {launch_cfg}: {str(exc)[:90]}")
        return float("inf")


def tune_shape(tag, E, N, K, top_k, group_size, Ms):
    dev = "cuda"
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    cands = _candidate_configs(N, K, group_size)
    table = {}
    for M in Ms:
        a = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
        topk_ids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
        ).to(torch.int32)
        topk_w = torch.rand(M * top_k, device=dev, dtype=torch.float32)
        c = torch.zeros((M * top_k, N), dtype=torch.bfloat16, device=dev)
        best_ms, best = float("inf"), None
        for cfg in cands:
            ms = _bench_config(cfg, a, packed, scale, c, topk_ids, topk_w, top_k, group_size)
            if ms < best_ms:
                best_ms, best = ms, cfg
        if best is None:
            print(f"  [{tag}] M={M:5d} -> NO VALID CONFIG")
            continue
        entry = dict(best.kwargs)
        entry["num_warps"] = best.num_warps
        entry["num_stages"] = best.num_stages
        entry["_ms"] = round(best_ms, 5)
        table[str(M)] = entry
        print(f"  [{tag}] M={M:5d} -> {best_ms:.5f} ms  {entry}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 512, 4096])
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU; tuning requires gfx942 (or any CUDA/ROCm GPU).")
        return

    import triton

    device = _device_name()
    date = datetime.date.today().isoformat()
    os.makedirs(_config_dir(), exist_ok=True)
    shapes = [
        ("gate_up", 2 * KIMI["INTER"], KIMI["HIDDEN"]),
        ("down", KIMI["HIDDEN"], KIMI["INTER"]),
    ]
    for tag, N, K in shapes:
        print(f"== tuning {tag}: E={KIMI['E']} N={N} K={K} on {device} ==")
        table = tune_shape(tag, KIMI["E"], N, K, KIMI["TOP_K"], args.group_size, args.M)
        out = {
            "_provenance": {
                "device": device,
                "date": date,
                "triton": triton.__version__,
                "metric": "median ms, triton.do_bench, bf16 activations",
                "shape": {
                    "E": KIMI["E"], "N": N, "K": K,
                    "top_k": KIMI["TOP_K"], "group_size": args.group_size,
                },
            },
            **table,
        }
        path = os.path.join(
            _config_dir(), _config_filename(KIMI["E"], N, K, device, "int4_w4a16")
        )
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the non-GPU guard locally**

Run: `.venv/bin/python benchmarks/tune_moe_int4_w4a16.py`
Expected: prints `No GPU; tuning requires gfx942 ...` and exits 0 (torch.cuda unavailable locally).

- [ ] **Step 3: Lint + commit**

Run: `.venv/bin/ruff check benchmarks/tune_moe_int4_w4a16.py`
Expected: no errors.

```bash
git add benchmarks/tune_moe_int4_w4a16.py
git commit -m "bench(moe): on-device tuner for INT4 MoE configs (issue #16)"
```

---

### Task 6: SLURM job + package-data

**Files:**
- Create: `slurm/tune_moe_int4_beverin.sbatch`
- Modify: `pyproject.toml` (package-data)

- [ ] **Step 1: Write the SLURM script**

Create `slurm/tune_moe_int4_beverin.sbatch`:

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# Sweep INT4 W4A16 fused-MoE configs on beverin (gfx942 / MI300A) and write the
# tuned-config JSON into the repo's tuned_configs/ dir (issue #16).
#
#   sbatch slurm/tune_moe_int4_beverin.sbatch
#
#SBATCH --job-name=xk-tune-moe-int4
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:40:00
#SBATCH --output=tune-moe-int4-%j.out
#SBATCH --error=tune-moe-int4-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  python -u "'"$REPO"'/benchmarks/tune_moe_int4_w4a16.py"
'
```

- [ ] **Step 2: Add JSON to package-data**

In `pyproject.toml`, change the `package-data` line:

```toml
[tool.setuptools.package-data]
xkernels = ["ops/**/*.cu", "ops/**/*.cpp", "ops/**/*.h", "ops/**/*.json"]
```

- [ ] **Step 3: Lint + commit**

```bash
git add slurm/tune_moe_int4_beverin.sbatch pyproject.toml
git commit -m "chore(moe): SLURM tuner job + package tuned-config JSON (issue #16)"
```

---

### Task 7: Run on beverin, bake in results, update README, report

**Files:**
- Create (on device, then committed): `src/xkernels/ops/moe/triton/tuned_configs/*.json`
- Modify: `README.md` (perf note)

This task runs on the cluster; no TDD. Each step is a real command.

- [ ] **Step 1: Sync the working branch to beverin scratch**

Run (from repo root):
```bash
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.ruff_cache' --exclude '.pytest_cache' \
  ./ beverin:/capstor/scratch/cscs/xyao/kernels/
```
Expected: completes without error. (If the CSCS SSH cert expired, renew at https://sshservice.cscs.ch/ first.)

- [ ] **Step 2: Submit the tuning job and capture the job id**

Run:
```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch slurm/tune_moe_int4_beverin.sbatch'
```
Expected: `Submitted batch job <JOBID>`.

- [ ] **Step 3: Wait for completion and read the log**

Poll until the job leaves the queue, then print the log:
```bash
ssh beverin 'squeue -u $USER; echo ---; tail -n 80 /capstor/scratch/cscs/xyao/kernels/tune-moe-int4-<JOBID>.out'
```
Expected: per-M winner lines for both `gate_up` and `down`, and two `wrote .../tuned_configs/E=...json` lines. Sanity-check the winning ms are finite and decode (M≤16) ms are in the sub-millisecond-to-few-ms range.

- [ ] **Step 4: Pull the produced JSON back into the local repo**

Run:
```bash
rsync -az beverin:/capstor/scratch/cscs/xyao/kernels/src/xkernels/ops/moe/triton/tuned_configs/ \
  src/xkernels/ops/moe/triton/tuned_configs/
ls -l src/xkernels/ops/moe/triton/tuned_configs/
.venv/bin/python -c "import json,glob;[print(f, sorted(k for k in json.load(open(f)) if not k.startswith('_'))) for f in glob.glob('src/xkernels/ops/moe/triton/tuned_configs/*.json')]"
```
Expected: two JSON files (gate_up N=4096,K=7168 and down N=7168,K=2048), each with buckets covering 1,2,4,8,16,32,512,4096.

- [ ] **Step 5: Confirm the tuned path is exercised on device**

Run the GPU correctness test on beverin to confirm the tuned configs run and match the reference for a production-ish shape (relies on the synced JSON):
```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch --wrap "srun --environment=tokenspeed-rocm-aiter-myofi --cpu-bind=none bash -lc \"export PYTHONPATH=$PWD/src; python -m pytest tests/test_moe_int4_w4a16.py tests/test_moe_int4_tuned_config.py -q\"" --partition=mi300 --account=a-infra02 --gpus-per-node=1 --gpu-bind=none --time=00:15:00 -o moe-test-%j.out'
```
Then read `moe-test-<JOBID>.out`. Expected: all tests pass on the GPU (bf16, atol/rtol 2e-2).

- [ ] **Step 6: Re-run `bench_all` on beverin for the updated `moe_int4_w4a16` number**

Run:
```bash
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels && sbatch slurm/bench_all_beverin.sbatch'
```
Read the resulting `bench-all-<JOBID>.out`. The `moe_int4_w4a16` row (M=64, gate_up shape) now uses the tuned path (closest bucket ≤ 64 = 32). Record its optimized ms + speedup.

- [ ] **Step 7: Update the README perf note**

In `README.md`, update the `moe_int4_w4a16` table row's optimized ms/speedup with the bench_all number from Step 6, and append a sentence to the existing `moe_int4_w4a16` notes (or add a note) stating that the production shapes are now driven by a checked-in tuned config (link the JSON) covering decode buckets M∈{1,2,4,8,16}, tuned on MI300A — and that this removes the default-config path in production. Use the actual measured numbers; do not invent.

- [ ] **Step 8: Commit the JSON + README**

```bash
git add src/xkernels/ops/moe/triton/tuned_configs/*.json README.md
git commit -m "feat(moe): checked-in tuned INT4 MoE configs for Kimi-K2.6 shapes on MI300A (issue #16)"
```

- [ ] **Step 9: Push, open PR, and report on the issue**

```bash
git push -u origin issue-16-tuned-moe-int4-config
gh pr create --repo ResearchComputer/kernels --base main \
  --title "feat(moe): tuned INT4 W4A16 fused-MoE configs for Kimi-K2.6 shapes (issue #16)" \
  --body "<summary: loader + direct launch + tuner + checked-in MI300A configs; closes #16; include the decode-bucket ms table>"
```
Then add a comment to issue #16 summarizing the tuned decode-bucket ms and that the default-config warning is resolved by the checked-in JSON. (Squash-merge per repo convention once reviewed.)

---

## Self-review

- **Spec coverage:** JSON store (Task 5 writes, Task 7 checks in) ✓; loader `load_tuned_config`/`get_moe_int4_config` (Task 1) ✓; direct launch + autotune fallback (Task 3) ✓; tuner (Task 5) ✓; SLURM (Task 6) ✓; loader unit tests (Task 1) + tuned-path integration test (Task 4) ✓; on-device run + bake-in + README + #16 report (Task 7) ✓. Alignment invariant (spec edge case) handled in Tasks 2–3 ✓. package-data for JSON (spec deliverable shipping) ✓.
- **Placeholders:** none — all steps carry concrete code/commands. `<JOBID>` / PR body are runtime values, intentionally filled at execution.
- **Type/name consistency:** `get_moe_int4_config`, `_select_config`, `_config_dir`, `_config_filename`, `_device_name`, `load_tuned_config`, `align_block_m`, `_TUNED_CACHE`, `_fused_moe_int4_kernel`, `fused_moe_int4_kernel`, `int4_w4a16_moe_gemm(..., config=)` are used identically across Tasks 1–6. Config dict keys (`BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`, `num_warps`, `num_stages`, `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`) match between the tuner's writer, the JSON, the loader, and the launch reader.
