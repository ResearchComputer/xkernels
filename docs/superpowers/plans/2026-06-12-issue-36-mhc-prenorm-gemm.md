# MHC Hidden-Compression Prenorm GEMM (#36) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the gfx942 (MI300A) portable replacement for `deep_gemm.tf32_hc_prenorm_gemm` — the GEMM + RMS-prenorm-squared-sum half of DeepSeek-V4's `mhc_pre` — as a clean xkernels-native op (`hc_prenorm_gemm`) re-exported under the upstream-faithful name (`tf32_hc_prenorm_gemm`) tokenspeed binds, so the V4 MHC layer stops raising on AMD.

**Architecture:** A new `mhc` kernel type in `src/xkernels/ops/mhc/`: a pure-torch oracle (`F.linear(A, fn)` + row squared-sum, written into the split-K layout), a Triton split-K kernel (one program per `(split, row-tile)`, each owning a contiguous K-range, fusing `tl.dot(A, fnᵀ)` and `Σ A²` from the same A loads), and a faithful-named in-place wrapper. The op writes `gemm_out_mul[n_splits, T, N]` and `gemm_out_sqrsum[n_splits, T]` such that summing over splits reproduces the full `F.linear`/sqsum — the only invariant the downstream TileLang post-fusion depends on.

**Tech Stack:** PyTorch, Triton (gfx942), pytest (GPU bf16 / `TRITON_INTERPRET=1` CPU fp32), SLURM on beverin (CSCS MI300A).

**Key facts (V4):** `A = residual_flat.view(T, K)` bf16, `K = hc_mult*hidden` (Flash: 4·4096=16384). `fn` is **`[N, K]` fp32** (Linear orientation, `hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))`), `N = hc_mult3 = 2·hc_mult + hc_mult² = 24` for hc_mult=4. Op = `F.linear(A, fn)` (= `A @ fnᵀ`) + per-row `Σ A²`. `n_splits` is passed in (tokenspeed's `_compute_num_split`, ~64 at decode); the split partition is numerically free since TileLang only sums across splits.

**Reference oracle = source of truth.** Every Triton result is asserted against the **summed** invariant (`out.sum(0)` vs `F.linear`/sqsum), never per-split (reference puts all in split 0; Triton genuinely distributes — only the sum matches).

**Env note:** all local commands use the repo venv: prefix with `VIRTUAL_ENV=.venv` and call `.venv/bin/python` / `.venv/bin/ruff` (matches the #32 plan).

---

### Task 1: Reference oracle (`F.linear` + row squared-sum, split layout)

**Files:**
- Create: `src/xkernels/ops/mhc/__init__.py`
- Create: `src/xkernels/ops/mhc/reference.py`
- Test: `tests/test_mhc_prenorm_gemm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mhc_prenorm_gemm.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
import os

import pytest
import torch
import torch.nn.functional as F

from xkernels.ops.mhc.reference import hc_prenorm_gemm_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _full(a, fn):
    """Independent oracle: full F.linear(A, fn) and per-row sum of squares (fp32)."""
    af = a.float()
    return F.linear(af, fn.float()), (af * af).sum(dim=-1)


def test_reference_split_sum_invariant():
    dev = _dev()
    torch.manual_seed(0)
    T, hc_mult, hidden = 5, 4, 32
    K = hc_mult * hidden
    N = 2 * hc_mult + hc_mult * hc_mult  # 24
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    for n_splits in (1, 4, 16):
        mul, sqr = hc_prenorm_gemm_ref(a, fn, n_splits=n_splits)
        assert mul.shape == (n_splits, T, N)
        assert sqr.shape == (n_splits, T)
        assert mul.dtype == torch.float32 and sqr.dtype == torch.float32
        fmul, fsqr = _full(a, fn)
        torch.testing.assert_close(mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(sqr.sum(0), fsqr, atol=1e-4, rtol=1e-4)


def test_reference_empty_tokens():
    dev = _dev()
    a = torch.zeros(0, 16, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(6, 16, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm_ref(a, fn, n_splits=4)
    assert mul.shape == (4, 0, 6) and sqr.shape == (4, 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'xkernels.ops.mhc'`.

- [ ] **Step 3: Write the package init + oracle**

```python
# src/xkernels/ops/mhc/__init__.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 MHC (multi-head hidden-compression) kernels.

Ships ``hc_prenorm_gemm`` (issue #36): the GEMM + RMS-prenorm squared-sum half of
V4's ``mhc_pre`` — a portable gfx942 replacement for the NVIDIA-only
``deep_gemm.tf32_hc_prenorm_gemm``. Re-exported under that faithful name so
tokenspeed binds it drop-in; the TileLang post-fusion that consumes its outputs
is already portable on AMD and is untouched.
"""
from .interface import hc_prenorm_gemm, tf32_hc_prenorm_gemm

# Import the Triton backend for its registration side effect (optional). Routed
# through the optional ``_triton_compat`` redirect so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed.
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import prenorm_gemm_kernel  # noqa: F401
except Exception:
    pass

__all__ = ["hc_prenorm_gemm", "tf32_hc_prenorm_gemm"]
```

```python
# src/xkernels/ops/mhc/reference.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 MHC hidden-compression prenorm GEMM
(issue #36) — numerical oracle and default (CPU / no-Triton) backend on gfx942.

This is the GEMM + RMS-prenorm-squared-sum half of V4's ``mhc_pre``. Upstream
computes it with the NVIDIA-only ``deep_gemm.tf32_hc_prenorm_gemm``; on AMD that
raises. The op takes the flattened residual ``A = residual.view(T, hc_mult*hidden)``
(bf16) and the fp32 hidden-compression weight ``fn`` of shape ``[N, K]`` (Linear
orientation, ``N = hc_mult3 = 2*hc_mult + hc_mult**2``), and produces, in a
split-K layout consumed by the TileLang post-fusion:

    gemm_out_mul[s, t, :]    partial of  F.linear(A, fn)[t]  ( = A @ fn.T )
    gemm_out_sqrsum[s, t]    partial of  (A.float()**2).sum(-1)[t]   (RMS prenorm)

summed over the split axis ``s``. The TileLang kernel only ever sums across
splits, so any complete disjoint K-partition is valid; the reference uses the
trivial one (full result in split 0, zeros elsewhere). All math in fp32 (CDNA3
has no TF32; the parity target is this reference, not NVIDIA bit-equality).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["hc_prenorm_gemm_ref"]


def hc_prenorm_gemm_ref(
    a: torch.Tensor,
    fn: torch.Tensor,
    *,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MHC prenorm GEMM reference. See module docstring.

    Args:
        a: ``[T, K]`` flattened residual (bf16; fp32 accepted). ``K = hc_mult*hidden``.
        fn: ``[N, K]`` fp32 hidden-compression weight (Linear orientation).
        n_splits: number of K-split partials to emit (``>= 1``).

    Returns:
        ``(gemm_out_mul [n_splits, T, N] fp32, gemm_out_sqrsum [n_splits, T] fp32)``
        with ``gemm_out_mul.sum(0) == F.linear(a.float(), fn.float())`` and
        ``gemm_out_sqrsum.sum(0) == (a.float()**2).sum(-1)``.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    T, K = a.shape
    N = fn.shape[0]
    if fn.shape[1] != K:
        raise ValueError(f"fn must be [N, K] with K={K}, got {tuple(fn.shape)}")
    af = a.float()
    gemm_out_mul = af.new_zeros(n_splits, T, N)
    gemm_out_sqrsum = af.new_zeros(n_splits, T)
    if T > 0:
        gemm_out_mul[0] = F.linear(af, fn.float())
        gemm_out_sqrsum[0] = (af * af).sum(dim=-1)
    return gemm_out_mul, gemm_out_sqrsum


register("hc_prenorm_gemm", Backend.REFERENCE)(hc_prenorm_gemm_ref)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/mhc/__init__.py src/xkernels/ops/mhc/reference.py tests/test_mhc_prenorm_gemm.py
git commit -m "feat(mhc): hc_prenorm_gemm reference oracle (split-K layout) (#36)"
```

> Note: Step 3 imports `from .interface import ...` in `__init__.py`, but `interface.py` is created in Task 2. Until then, importing the package top-level would fail — the Task 1 test imports `xkernels.ops.mhc.reference` directly (not the package `__init__`), so it passes. Task 2 adds `interface.py` and the package import resolves.

---

### Task 2: Native op + faithful in-place wrapper

**Files:**
- Create: `src/xkernels/ops/mhc/interface.py`
- Test: `tests/test_mhc_prenorm_gemm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mhc_prenorm_gemm.py
from xkernels.ops.mhc import hc_prenorm_gemm, tf32_hc_prenorm_gemm


def test_native_op_dispatches_to_reference():
    dev = _dev()
    torch.manual_seed(1)
    T, K, N = 3, 64, 8
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=4, backend="reference")
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sqr.sum(0), fsqr, atol=1e-4, rtol=1e-4)


def test_faithful_wrapper_writes_in_place():
    """tf32_hc_prenorm_gemm matches the deep_gemm signature: in-place, returns None."""
    dev = _dev()
    torch.manual_seed(2)
    T, K, N, n_splits = 4, 128, 24, 3
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    gemm_out_mul = torch.empty(n_splits, T, N, device=dev, dtype=torch.float32)
    gemm_out_sqrsum = torch.empty(n_splits, T, device=dev, dtype=torch.float32)
    ret = tf32_hc_prenorm_gemm(a, fn, gemm_out_mul, gemm_out_sqrsum, n_splits,
                               backend="reference")
    assert ret is None
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(gemm_out_mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gemm_out_sqrsum.sum(0), fsqr, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py -q -k "native_op or faithful"`
Expected: FAIL — `ModuleNotFoundError`/`ImportError` (no `interface`).

- [ ] **Step 3: Write the interface**

```python
# src/xkernels/ops/mhc/interface.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public MHC op (``hc_prenorm_gemm``) + faithful-named in-place wrapper
(``tf32_hc_prenorm_gemm``): dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import reference  # noqa: F401  (registers hc_prenorm_gemm REFERENCE)


def hc_prenorm_gemm(
    a: torch.Tensor,
    fn: torch.Tensor,
    *,
    n_splits: int,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 MHC hidden-compression prenorm GEMM (issue #36): the GEMM +
    RMS-prenorm squared-sum half of ``mhc_pre``. Portable gfx942 replacement for
    the NVIDIA-only ``deep_gemm.tf32_hc_prenorm_gemm``.

    Computes, in a split-K layout summed over the split axis by the downstream
    TileLang post-fusion:

        gemm_out_mul.sum(0)    == F.linear(a.float(), fn.float())   ( = a @ fn.T )
        gemm_out_sqrsum.sum(0) == (a.float()**2).sum(-1)

    Args:
        a: ``[T, K]`` flattened residual (bf16; fp32 accepted). ``K = hc_mult*hidden``.
        fn: ``[N, K]`` fp32 hidden-compression weight (Linear orientation,
            ``N = hc_mult3 = 2*hc_mult + hc_mult**2``).
        n_splits: number of K-split partials to emit (``>= 1``).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(gemm_out_mul [n_splits, T, N] fp32, gemm_out_sqrsum [n_splits, T] fp32)``.
    """
    return dispatch("hc_prenorm_gemm", a, fn, n_splits=n_splits, backend=backend)


def tf32_hc_prenorm_gemm(
    a: torch.Tensor,
    fn: torch.Tensor,
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    n_splits: int,
    *,
    backend: Backend | str = "auto",
) -> None:
    """Upstream-faithful in-place wrapper (the tokenspeed binding target).

    Exact ``deep_gemm.tf32_hc_prenorm_gemm`` positional signature: writes the
    pre-allocated ``gemm_out_mul [n_splits, T, N]`` / ``gemm_out_sqrsum
    [n_splits, T]`` fp32 buffers in place and returns ``None``.
    """
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=backend)
    if gemm_out_mul.shape != mul.shape or gemm_out_sqrsum.shape != sqr.shape:
        raise ValueError(
            f"out buffer shape mismatch: mul {tuple(gemm_out_mul.shape)} vs "
            f"{tuple(mul.shape)}, sqrsum {tuple(gemm_out_sqrsum.shape)} vs "
            f"{tuple(sqr.shape)}"
        )
    gemm_out_mul.copy_(mul)
    gemm_out_sqrsum.copy_(sqr)
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/mhc/interface.py tests/test_mhc_prenorm_gemm.py
git commit -m "feat(mhc): hc_prenorm_gemm native op + tf32_hc_prenorm_gemm in-place wrapper (#36)"
```

---

### Task 3: Triton split-K kernel (gfx942)

**Files:**
- Create: `src/xkernels/ops/mhc/triton/__init__.py`
- Create: `src/xkernels/ops/mhc/triton/prenorm_gemm_kernel.py`
- Test: `tests/test_mhc_prenorm_gemm.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mhc_prenorm_gemm.py
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends

_HAS_TRITON = Backend.TRITON in registered_backends("hc_prenorm_gemm")


@pytest.mark.parametrize("hc_mult,hidden", [(4, 64), (2, 48), (4, 70)])  # 70 -> K not /64
@pytest.mark.parametrize("n_splits", [1, 4, 16])
def test_triton_matches_reference(hc_mult, hidden, n_splits):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    torch.manual_seed(3)
    T = 7
    K = hc_mult * hidden
    N = 2 * hc_mult + hc_mult * hc_mult
    dt = torch.float32 if _INTERP else torch.bfloat16
    a = torch.randn(T, K, device=dev, dtype=dt)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    got_mul, got_sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)
    fmul, fsqr = _full(a, fn)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    # Only the SUM over splits is the invariant (Triton distributes K genuinely).
    torch.testing.assert_close(got_mul.sum(0), fmul, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_sqr.sum(0), fsqr, atol=atol, rtol=rtol)


def test_triton_v4_flash_shape():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    if _INTERP:
        pytest.skip("V4 K=16384 too slow under the CPU interpreter")
    torch.manual_seed(4)
    T, hc_mult, hidden = 8, 4, 4096
    K, N = hc_mult * hidden, 24
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=16, backend=Backend.TRITON)
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(mul.sum(0), fmul, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(sqr.sum(0), fsqr, atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py::test_triton_matches_reference -q`
Expected: FAIL — under `TRITON_INTERPRET=1` the triton import in `mhc/__init__.py` runs, but `prenorm_gemm_kernel` doesn't exist yet, so `_HAS_TRITON` is False and the test SKIPs. Treat a full skip as "not yet passing"; it turns into real PASS after Step 3/4. (To force a hard fail first, the kernel file simply being absent is the red state.)

- [ ] **Step 3: Write the Triton kernel**

```python
# src/xkernels/ops/mhc/triton/__init__.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton backends for the MHC kernels (gfx942)."""
```

```python
# src/xkernels/ops/mhc/triton/prenorm_gemm_kernel.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton split-K MHC prenorm GEMM for AMD MI300A (gfx942, CDNA3), issue #36.

One program per ``(split, row-tile)``. Split ``s`` owns a contiguous K-range (the
``ceil_div(K, BLOCK_K)`` K-blocks partitioned as evenly as possible across
``n_splits``), streams it in ``BLOCK_K`` chunks, and accumulates both
``A[:, krange] @ fn[:, krange].T`` (via ``tl.dot`` with a transposed ``fn`` tile —
``fn`` is stored ``[N, K]``) and the per-row ``Σ A²`` from the same A loads. The
downstream TileLang post-fusion sums the per-split partials, so the disjoint
K-partition reproduces the full ``F.linear``/sqsum exactly. Empty splits (when
``n_splits > num_kblocks``) still run and store zeros, keeping every
``torch.empty`` output slot defined. Compute is fp32 (CDNA3 has no TF32).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["hc_prenorm_gemm_triton", "hc_prenorm_gemm_kernel"]


@triton.jit
def hc_prenorm_gemm_kernel(
    a_ptr, fn_ptr, mul_ptr, sqr_ptr,
    T, K, N, n_splits, num_kblocks,
    stride_at, stride_ak,
    stride_fn, stride_fk,
    stride_ms, stride_mt, stride_mn,
    stride_ss, stride_st,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    s = tl.program_id(0)
    m = tl.program_id(1)
    rows = m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < T
    ns = tl.arange(0, BLOCK_N)
    n_mask = ns < N

    # Contiguous K-block range owned by this split (even partition).
    kb_lo = s * num_kblocks // n_splits
    kb_hi = (s + 1) * num_kblocks // n_splits
    k_lo = kb_lo * BLOCK_K
    k_hi = tl.minimum(kb_hi * BLOCK_K, K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k in range(k_lo, k_hi, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        a_tile = tl.load(
            a_ptr + rows[:, None] * stride_at + ks[None, :] * stride_ak,
            mask=row_mask[:, None] & k_mask[None, :], other=0.0,
        ).to(tl.float32)
        # fn is [N, K]; gather a [BLOCK_K, BLOCK_N] tile (K on axis0, N on axis1).
        fn_tile = tl.load(
            fn_ptr + ks[:, None] * stride_fk + ns[None, :] * stride_fn,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        ).to(tl.float32)
        acc += tl.dot(a_tile, fn_tile)
        sq += tl.sum(a_tile * a_tile, axis=1)

    tl.store(
        mul_ptr + s * stride_ms + rows[:, None] * stride_mt + ns[None, :] * stride_mn,
        acc, mask=row_mask[:, None] & n_mask[None, :],
    )
    tl.store(sqr_ptr + s * stride_ss + rows * stride_st, sq, mask=row_mask)


def hc_prenorm_gemm_triton(a, fn, *, n_splits):
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    a = a.contiguous()
    fn = fn.contiguous()
    T, K = a.shape
    N = fn.shape[0]
    if fn.shape[1] != K:
        raise ValueError(f"fn must be [N, K] with K={K}, got {tuple(fn.shape)}")
    mul = torch.empty(n_splits, T, N, device=a.device, dtype=torch.float32)
    sqr = torch.empty(n_splits, T, device=a.device, dtype=torch.float32)
    if T == 0:
        return mul, sqr  # no rows; nothing to write

    BLOCK_M = 64
    BLOCK_K = 64
    BLOCK_N = max(16, triton.next_power_of_2(N))
    num_kblocks = triton.cdiv(K, BLOCK_K)
    grid = (n_splits, triton.cdiv(T, BLOCK_M))
    hc_prenorm_gemm_kernel[grid](
        a, fn, mul, sqr,
        T, K, N, n_splits, num_kblocks,
        a.stride(0), a.stride(1),
        fn.stride(0), fn.stride(1),
        mul.stride(0), mul.stride(1), mul.stride(2),
        sqr.stride(0), sqr.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    return mul, sqr


register("hc_prenorm_gemm", Backend.TRITON)(hc_prenorm_gemm_triton)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py::test_triton_matches_reference -q`
Expected: PASS (9 params). (`test_triton_v4_flash_shape` skips under the interpreter; it runs on-device in Task 6.)

If `tl.dot` rejects fp32 operands on the host triton build, that surfaces here as a compile error — the on-device gfx942 path is the real target (Task 6); for the interpreter, fp32 `tl.dot` is supported. Do **not** switch to bf16 inputs (it would break the fp32 parity story); if a specific build fails, keep fp32 and note it for the on-device run.

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/mhc/triton/__init__.py src/xkernels/ops/mhc/triton/prenorm_gemm_kernel.py tests/test_mhc_prenorm_gemm.py
git commit -m "feat(mhc): Triton split-K prenorm GEMM kernel for gfx942 (#36)"
```

---

### Task 4: Top-level public surface

**Files:**
- Modify: `src/xkernels/__init__.py`
- Test: `tests/test_mhc_prenorm_gemm.py`

> `src/xkernels/ops/__init__.py` is a bare docstring (no re-exports); the
> top-level `__init__.py` imports directly from each `.ops.<type>` module. So
> only `src/xkernels/__init__.py` needs editing — match that established pattern.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_mhc_prenorm_gemm.py
def test_top_level_exports():
    import xkernels
    for name in ("hc_prenorm_gemm", "tf32_hc_prenorm_gemm"):
        assert hasattr(xkernels, name), name
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py::test_top_level_exports -q`
Expected: FAIL — `AttributeError: module 'xkernels' has no attribute 'hc_prenorm_gemm'`.

- [ ] **Step 3: Wire the exports**

In `src/xkernels/__init__.py`, add an import line alongside the other `from .ops.<type> import ...` lines (place it after the `.ops.gather` import to keep rough alpha order):

```python
from .ops.mhc import hc_prenorm_gemm, tf32_hc_prenorm_gemm
```

and add these two entries to the top-level `__all__` list (e.g. after `"mxfp4_paged_gather",`):

```python
    "hc_prenorm_gemm",
    "tf32_hc_prenorm_gemm",
```

- [ ] **Step 4: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_mhc_prenorm_gemm.py -q`
Expected: PASS (whole file; V4 shape skips under interpreter).

- [ ] **Step 5: Lint + commit**

Run: `VIRTUAL_ENV=.venv .venv/bin/ruff check src/xkernels/ops/mhc tests/test_mhc_prenorm_gemm.py src/xkernels/__init__.py`
Expected: clean (fix any import-order/unused warnings).

```bash
git add src/xkernels/__init__.py tests/test_mhc_prenorm_gemm.py
git commit -m "feat(mhc): export hc_prenorm_gemm / tf32_hc_prenorm_gemm at top level (#36)"
```

---

### Task 5: Benchmark + bench_all wiring + kernel doc

**Files:**
- Create: `benchmarks/bench_mhc_prenorm_gemm.py`
- Modify: `benchmarks/bench_all.py`
- Create: `docs/issue-36-mhc-prenorm-gemm.md`

- [ ] **Step 1: Write the standalone benchmark**

```python
# benchmarks/bench_mhc_prenorm_gemm.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Benchmark the MHC prenorm GEMM (#36) vs the naive torch baseline a
practitioner would write: F.linear(A, fn) + a separate per-row sum-of-squares.

Run on one gfx942 GPU (see slurm/test_mhc_prenorm_beverin.sbatch)::

    python benchmarks/bench_mhc_prenorm_gemm.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton

from xkernels import hc_prenorm_gemm
from xkernels._backends import Backend


def main():
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (or any CUDA/ROCm) GPU.")
        return
    dev = "cuda"
    # V4-Flash MHC: hc_mult=4, hidden=4096 -> K=16384, N=24. Decode T small.
    for T in (1, 8, 64):
        hc_mult, hidden = 4, 4096
        K, N = hc_mult * hidden, 24
        n_splits = 16
        a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        fn = torch.randn(N, K, device=dev, dtype=torch.float32)

        def naive():
            af = a.float()
            return F.linear(af, fn.float()), (af * af).sum(-1)

        def opt():
            return hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)

        t_naive = triton.testing.do_bench(naive)
        t_opt = triton.testing.do_bench(opt)
        print(
            f"| mhc_prenorm_gemm | T={T}, K={K}, N={N}, splits={n_splits} | "
            f"{t_naive:.3f} ms (F.linear+sqsum) | {t_opt:.3f} ms | "
            f"{t_naive / t_opt:.2f}x |"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Wire into `benchmarks/bench_all.py`** — add a `bench_mhc_prenorm` following the existing `bench_sparse_mla` pattern (a `_record(...)` call with a naive vs Triton closure), and add `bench_mhc_prenorm` to the tuple in `main()`:

```python
def bench_mhc_prenorm(dev):
    import torch.nn.functional as F

    from xkernels import hc_prenorm_gemm
    from xkernels._backends import Backend

    T, hc_mult, hidden = 8, 4, 4096
    K, N, n_splits = hc_mult * hidden, 24, 16
    a = torch.randn(T, K, device=dev, dtype=DT)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)

    def naive():
        af = a.float()
        return F.linear(af, fn.float()), (af * af).sum(-1)

    _record(
        "mhc_prenorm_gemm", f"T={T}, K={K}, N={N}, splits={n_splits}",
        "F.linear+sqsum",
        naive,
        lambda: hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON),
    )
```
Add `bench_mhc_prenorm,` to the `for fn in (...)` tuple in `main()` (after `bench_sparse_mla`).

- [ ] **Step 3: Verify bench imports cleanly (no GPU needed)**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -c "import ast,pathlib; ast.parse(pathlib.Path('benchmarks/bench_mhc_prenorm_gemm.py').read_text()); ast.parse(pathlib.Path('benchmarks/bench_all.py').read_text()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 4: Write `docs/issue-36-mhc-prenorm-gemm.md`** — the math (`F.linear(A, fn)` + RMS-prenorm sqsum, `fn` is `[N,K]`), the split-K-summed invariant (why the partition is free), the V4 dims (`K=16384`, `N=24`, hc_mult=4), the audit conclusion (only `deep_gemm` dep in the MHC path), and an on-device results table (filled in Task 6). Mirror `docs/issue-32-sparse-mla-attention.md` in structure.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/bench_mhc_prenorm_gemm.py benchmarks/bench_all.py docs/issue-36-mhc-prenorm-gemm.md
git commit -m "docs+bench(mhc): prenorm GEMM doc + benchmark + bench_all wiring (#36)"
```

---

### Task 6: On-device validation on beverin (MI300A / gfx942)

**Files:**
- Create: `slurm/test_mhc_prenorm_beverin.sbatch`
- Modify: `docs/issue-36-mhc-prenorm-gemm.md` (record results)

Mirrors `slurm/test_sparse_mla_beverin.sbatch` exactly (same account/partition/env).

- [ ] **Step 1: Write the sbatch**

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 MHC prenorm GEMM (issue #36) on
# beverin (gfx942 / MI300A): Triton split-K vs torch oracle, real bf16 A / fp32
# fn on the GPU (TRITON_INTERPRET unset so Triton actually compiles).
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/xkernels-issue-36 \
#          slurm/test_mhc_prenorm_beverin.sbatch
#
#SBATCH --job-name=xk-mhc-prenorm
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=mhc-prenorm-%j.out
#SBATCH --error=mhc-prenorm-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/xkernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"

echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  unset TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  cd "'"$REPO"'"

  python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, \"dev\", torch.cuda.get_device_name(0))"

  echo "######## pytest: MHC prenorm GEMM on GPU (Triton bf16/fp32 vs oracle) ########"
  python -m pytest tests/test_mhc_prenorm_gemm.py -q

  echo "######## V4-Flash MHC shape parity + max|err| ########"
  python -u - <<"PY"
import torch
import torch.nn.functional as F
from xkernels import hc_prenorm_gemm
from xkernels._backends import Backend

torch.manual_seed(0)
dev = "cuda"
# V4-Flash MHC: hc_mult=4, hidden=4096 -> K=16384, N=24. decode-ish T, 16 splits.
T, hc_mult, hidden = 8, 4, 4096
K, N, n_splits = hc_mult * hidden, 24, 16
a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
fn = torch.randn(N, K, device=dev, dtype=torch.float32)
mul, sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)
af = a.float()
fmul = F.linear(af, fn.float())
fsqr = (af * af).sum(-1)
emul = (mul.sum(0) - fmul).abs().max().item()
esqr = (sqr.sum(0) - fsqr).abs().max().item()
rmul = emul / fmul.abs().max().clamp_min(1e-6).item()
print(f"[V4 MHC T={T} K={K} N={N} splits={n_splits}] mul max|err|={emul:.4e} rel={rmul:.4e} | sqrsum max|err|={esqr:.4e}")
assert emul / fmul.abs().max().item() < 5e-2, (emul, rmul)
assert esqr / fsqr.abs().max().item() < 5e-2, esqr
print("PASS: MHC prenorm GEMM correct on gfx942")
PY
'
```

- [ ] **Step 2: Run on beverin** — sync the repo to scratch and submit (mirrors how #32 was validated):

```bash
ssh beverin 'mkdir -p /capstor/scratch/cscs/xyao/xkernels-issue-36'
rsync -az --delete --exclude .git --exclude '__pycache__' --exclude '.venv' \
  /home/xiayao/Documents/research/xkernels/ \
  beverin:/capstor/scratch/cscs/xyao/xkernels-issue-36/
ssh beverin 'cd /capstor/scratch/cscs/xyao/xkernels-issue-36 && \
  sbatch --export=ALL,REPO=$PWD slurm/test_mhc_prenorm_beverin.sbatch'
# then poll the job + tail mhc-prenorm-<jobid>.out for the PASS line.
```
Expected output: the pytest summary all-passed + `PASS: MHC prenorm GEMM correct on gfx942` with `mul max|err|` ~1e-2 or below. **If the cluster is unreachable** (no `beverin` SSH alias / no allocation), STOP here, leave this step unchecked, and report that the sbatch is ready for the user to submit — do not fabricate results.

- [ ] **Step 3: Record results in `docs/issue-36-mhc-prenorm-gemm.md`, commit**

Fill the on-device table with the real `torch.__version__`, `hip`, device name, pytest pass count, and the `max|err|`/`rel` numbers from the job output.

```bash
git add slurm/test_mhc_prenorm_beverin.sbatch docs/issue-36-mhc-prenorm-gemm.md
git commit -m "test(mhc): on-device prenorm GEMM validation on MI300A (#36)"
```

---

### Task 7: README perf row + PR

**Files:**
- Modify: `README.md` (add the `mhc_prenorm_gemm` row to the Performance table, with the Task-6 numbers)

- [ ] **Step 1: Add the README Performance row** using the measured `bench_all.py` / standalone numbers from Task 6 (only if the on-device run produced them; otherwise leave a note and skip the row rather than inventing one). Follow the existing table format and add a short bullet under "Notes" describing the op (memory-bound tall-skinny GEMM, K=16384/N=24, split-K for decode occupancy).

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(mhc): README perf row for hc_prenorm_gemm (#36)"
```

- [ ] **Step 3: Push + open draft PR** against `ResearchComputer/xkernels`:

```bash
git push -u origin feat/issue-36-mhc-prenorm-gemm
gh pr create --repo ResearchComputer/xkernels --draft \
  --title "feat(mhc): DeepSeek-V4 MHC hidden-compression prenorm GEMM on gfx942 (issue #36)" \
  --body "$(cat <<'EOF'
## What & why

The next gating kernel for serving DeepSeek-V4-Flash on AMD MI300A (gfx942) after
sparse-MLA (#32/#33). With sparse-MLA bound, the V4 forward reaches the MHC layer
and dies in `mhc_pre` on `deep_gemm.tf32_hc_prenorm_gemm` (NVIDIA-only). This
ships a portable Triton replacement as a clean xkernels-native op
(`hc_prenorm_gemm`) re-exported under the faithful name `tf32_hc_prenorm_gemm`
tokenspeed binds drop-in.

## What ships

- **`hc_prenorm_gemm(a, fn, *, n_splits)`** — native op (oracle + Triton gfx942
  split-K backend). `a [T,K]` bf16 residual, `fn [N,K]` fp32 weight (Linear
  orientation); produces `gemm_out_mul [n_splits,T,N]` + `gemm_out_sqrsum
  [n_splits,T]` whose split-sums are `F.linear(a, fn)` and `Σ a²` — the RMS
  prenorm + projection the TileLang post-fusion consumes.
- **`tf32_hc_prenorm_gemm(a, fn, out_mul, out_sqrsum, n_splits)`** — faithful
  in-place wrapper matching the deep_gemm signature.
- Split-K (each split owns a disjoint K-range) parallelizes the K=16384 reduction
  for small decode T; the partition is numerically free (TileLang only sums splits).

## Audit

The only NVIDIA-only dep in the whole MHC path is this one GEMM
(`deepseek_v4_mhc.py:284`); `mhc_post` and the `mhc_pre` post-fusion are pure
TileLang, available on AMD. Replacing this unblocks the MHC layer.

## Test plan

- Offline (`TRITON_INTERPRET=1` CPU fp32 + GPU bf16): Triton vs torch oracle on
  the split-summed invariant, `n_splits∈{1,4,16}`, `hc_mult∈{2,4}`, K not
  divisible by BLOCK_K, T=0 edge, faithful-wrapper in-place write.
- On-device (beverin, MI300A / gfx942): `tests/test_mhc_prenorm_gemm.py` + a
  V4-Flash-shape (K=16384, N=24) parity max|err| check. [results in the PR body /
  issue doc]

Refs #36, umbrella #28. Out of scope: tokenspeed-side binding (a tokenspeed
change), the TileLang post-fusion (already portable).
EOF
)"
```

- [ ] **Step 4: Comment on issue #36** linking the PR (or, if the cluster was unreachable, note that the kernel + offline validation are done and on-device is pending the sbatch run).

---

## Self-Review

**Spec coverage:** reference oracle + split layout (T1), native op + faithful in-place wrapper (T2), Triton split-K kernel (T3), top-level exports (T4), bench + bench_all + doc (T5), beverin on-device validation (T6), README row + PR (T7). Every spec section (purpose, the op, the audit, API, kernel strategy, testing, public surface) maps to a task. ✅

**Placeholder scan:** code is concrete in every code step; the only deferred items are explicit *measurement* fills (on-device numbers in T6/T7) that cannot exist before the run, and they are gated with "do not fabricate / skip the row" instructions. ✅

**Type consistency:** `hc_prenorm_gemm(a, fn, *, n_splits, backend) -> (mul [n_splits,T,N], sqrsum [n_splits,T])` everywhere; `tf32_hc_prenorm_gemm(a, fn, gemm_out_mul, gemm_out_sqrsum, n_splits, *, backend) -> None`; `hc_prenorm_gemm_ref` / `hc_prenorm_gemm_triton` share the `(a, fn, *, n_splits)` signature and the registry key `"hc_prenorm_gemm"` (REFERENCE in T1, TRITON in T3). `fn` is `[N, K]` consistently; the kernel loads it transposed for `A @ fnᵀ`. Test helper `_full(a, fn)` returns `(F.linear, sqsum)` and is reused across tasks. ✅

**Invariant discipline:** every Triton assertion is on `out.sum(0)` vs the full `F.linear`/sqsum — never per-split — because the reference (all in split 0) and Triton (genuine split-K) differ per-split by construction; only the sum is the contract. ✅
</content>
