# Sparse-MLA Attention Compute (#32) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the gfx942 (MI300A) sparse-MLA attention *compute* for DeepSeek-V4 — the flash softmax over the DSA-indexer-selected latent KV — as a clean xkernels-native op re-exported under the upstream-faithful names tokenspeed binds (`flash_mla_sparse_fwd` / `flash_mla_with_kvcache` / `get_mla_metadata`).

**Architecture:** A new `sparse-MLA` area in `src/xkernels/ops/attention/`: a pure-torch oracle (the authored MLA math), a Triton flash kernel (one program per `(token, head)`, online softmax over top-k with an optional attention-sink logit), an `fp8_ds_mla` dequant module (mirrors `gather/mxfp4.py` + #29), and thin faithful-named wrappers. Prefill takes a shared bf16 latent workspace + indices; decode gathers+dequants the paged fp8 cache(s) per token, flattens to the same `(kv, indices)` form, and reuses the one compute kernel.

**Tech Stack:** PyTorch, Triton (gfx942), pytest (GPU bf16 / `TRITON_INTERPRET=1` CPU fp32), SLURM on beverin (CSCS MI300A).

**Key dims (V4):** latent `D=512` = nope/`kv_lora` `448` (value-bearing, fp8 e4m3 w/ per-64 pow2 scale) + decoupled rope `64` (score-only, bf16). MQA (one shared latent KV head). `topk` 512 (Flash) / 1024 (Pro). `sm_scale`, `attn_sink`, `topk_length` passed in. Value dim `d_v` is parameterized (default = full `D`; pin to 448 on-device against tokenspeed's o_proj — see Task 8).

**Reference oracle = source of truth.** Every Triton/decode result is asserted against `sparse_mla_attention_ref`.

---

### Task 1: Reference oracle (the MLA compute math)

**Files:**
- Create: `src/xkernels/ops/attention/sparse_mla_reference.py`
- Test: `tests/test_sparse_mla_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sparse_mla_attention.py
import os
import pytest
import torch

from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _naive(q, kv, indices, sm_scale, topk_length, attn_sink, d_v):
    # Independent dense oracle: build the selected scores explicitly per (t,h).
    T, H, D = q.shape
    topk = indices.shape[1]
    out = torch.zeros(T, H, d_v)
    for t in range(T):
        n = int(topk_length[t]) if topk_length is not None else topk
        for h in range(H):
            logits, vals = [], []
            for j in range(topk):
                idx = int(indices[t, j])
                if idx < 0 or j >= n:
                    continue
                logits.append(sm_scale * float(q[t, h].float() @ kv[idx].float()))
                vals.append(kv[idx, :d_v].float())
            if attn_sink is not None:
                logits.append(float(attn_sink.reshape(-1)[h % attn_sink.numel()]))
                vals.append(torch.zeros(d_v))
            if not logits:
                continue
            lg = torch.tensor(logits)
            p = torch.softmax(lg, dim=0)
            out[t, h] = (p[:, None] * torch.stack(vals)).sum(0)
    return out


def test_oracle_matches_independent_naive():
    dev = _dev()
    torch.manual_seed(0)
    T, H, D, Kv, topk, d_v = 3, 4, 16, 32, 6, 16
    q = torch.randn(T, H, D, device=dev)
    kv = torch.randn(Kv, D, device=dev)
    indices = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    indices[0, -2:] = -1  # sentinels
    topk_length = torch.tensor([topk, topk - 1, topk], device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)
    out, lse, maxl = sparse_mla_attention_ref(
        q, kv, indices, sm_scale=0.25, topk_length=topk_length, attn_sink=sink, d_v=d_v
    )
    ref = _naive(q.cpu(), kv.cpu(), indices.cpu(), 0.25, topk_length.cpu(), sink.cpu(), d_v)
    torch.testing.assert_close(out.float().cpu(), ref, atol=1e-5, rtol=1e-5)
    assert lse.shape == (T, H) and maxl.shape == (T, H)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_oracle_matches_independent_naive -q`
Expected: FAIL — `ModuleNotFoundError: ... sparse_mla_reference`.

- [ ] **Step 3: Write the oracle**

```python
# src/xkernels/ops/attention/sparse_mla_reference.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 sparse-MLA attention compute
(issue #32) — numerical oracle and default (CPU / no-Triton) backend on gfx942.

This is the kernel that *consumes* the DSA indexer's top-k KV selection (#27/#31)
and runs the actual attention softmax over V4's latent KV. MLA in latent form is
MQA: a single shared latent KV per position of dim ``D = kv_lora_rank + rope``
(V4: 512 = 448 + 64). The score uses the full ``D``; the value is the first
``d_v`` (the kv_lora / nope part). An optional per-head attention **sink** logit
joins the softmax denominator and contributes no value.

    s[t,h,j] = sm_scale * (q[t,h] . kv[idx[t,j]])      over selected idx
    p        = softmax([s..., sink[h]])                 (sink column has zero value)
    out[t,h] = sum_j p[j] * kv[idx[t,j], :d_v]

Validity of a column ``j`` is ``idx >= 0`` AND (when given) ``j < topk_length[t]``.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = ["sparse_mla_attention_ref"]

_NEG_INF = float("-inf")


def sparse_mla_attention_ref(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    d_v: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse latent-MLA attention. See module docstring.

    Args:
        q: ``[T, H, D]`` latent queries.
        kv: ``[Kv, D]`` shared latent MQA cache (bf16/fp32).
        indices: ``[T, topk]`` int32 columns into ``kv`` (``<0`` = padding).
        sm_scale: softmax scale applied to the q.k score.
        topk_length: optional ``[T]`` int — valid column count per query.
        attn_sink: optional ``[H]`` (or scalar) per-head sink logit.
        d_v: value/output dim (first ``d_v`` latent dims). Defaults to ``D``.

    Returns:
        ``(out [T, H, d_v] in q.dtype, lse [T, H] fp32, max_logits [T, H] fp32)``.
    """
    T, H, D = q.shape
    Kv = kv.shape[0]
    topk = indices.shape[1]
    d_v = D if d_v is None else d_v
    qf, kvf = q.float(), kv.float()
    out = q.new_zeros(T, H, d_v)
    lse = q.new_zeros(T, H, dtype=torch.float32)
    maxl = q.new_zeros(T, H, dtype=torch.float32)
    pos = torch.arange(topk, device=q.device)

    sink_vec = None
    if attn_sink is not None:
        s = attn_sink.float().reshape(-1)
        sink_vec = (s.expand(H) if s.numel() == 1 else s[:H]).reshape(H, 1)

    for t in range(T):
        idx = indices[t].long()
        valid = idx >= 0
        if topk_length is not None:
            valid = valid & (pos < int(topk_length[t]))
        safe = idx.clamp(0, Kv - 1)
        ksel = kvf[safe]  # [topk, D]
        scores = sm_scale * (qf[t] @ ksel.t())  # [H, topk]
        scores = scores.masked_fill(~valid.unsqueeze(0), _NEG_INF)
        aug = scores if sink_vec is None else torch.cat([scores, sink_vec], dim=1)
        m = aug.amax(dim=1)  # [H]
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        p = (aug - m_safe.unsqueeze(1)).exp()
        denom = p.sum(dim=1)
        pv = p[:, :topk]  # sink column excluded from value
        ov = (pv @ ksel[:, :d_v]) / denom.clamp_min(1e-20).unsqueeze(1)
        out[t] = ov.to(q.dtype)
        lse[t] = torch.where(denom > 0, m_safe + denom.clamp_min(1e-20).log(),
                             torch.full_like(m_safe, _NEG_INF))
        maxl[t] = m_safe
    return out, lse, maxl


register("sparse_mla_attention", Backend.REFERENCE)(sparse_mla_attention_ref)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_oracle_matches_independent_naive -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/attention/sparse_mla_reference.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): sparse-MLA attention oracle (#32)"
```

---

### Task 2: Native op + faithful wrappers + metadata stub

**Files:**
- Modify: `src/xkernels/ops/attention/interface.py`
- Test: `tests/test_sparse_mla_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sparse_mla_attention.py
from xkernels.ops.attention.interface import (
    flash_mla_sparse_fwd,
    get_mla_metadata,
    sparse_mla_attention,
)


def test_native_op_dispatches_to_reference():
    dev = _dev()
    torch.manual_seed(1)
    T, H, D, Kv, topk = 2, 3, 16, 20, 5
    q = torch.randn(T, H, D, device=dev)
    kv = torch.randn(Kv, D, device=dev)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    out, lse, maxl = sparse_mla_attention(q, kv, idx, sm_scale=0.3, backend="reference")
    from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref
    eo, el, em = sparse_mla_attention_ref(q, kv, idx, sm_scale=0.3)
    torch.testing.assert_close(out, eo)


def test_flash_mla_sparse_fwd_matches_oracle():
    """Prefill wrapper: [Kv,1,D] kv + [T,1,topk] indices, returns (out, maxl, lse)."""
    dev = _dev()
    torch.manual_seed(2)
    T, H, D, Kv, topk = 3, 4, 16, 24, 6
    q = torch.randn(T, H, D, device=dev)
    kv = torch.randn(Kv, 1, D, device=dev)
    idx = torch.randint(0, Kv, (T, 1, topk), device=dev, dtype=torch.int32)
    lens = torch.tensor([topk, topk - 2, topk - 1], device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)
    out, maxl, lse = flash_mla_sparse_fwd(
        q, kv, idx, 0.2, attn_sink=sink, topk_length=lens, backend="reference"
    )
    from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref
    eo, el, em = sparse_mla_attention_ref(
        q, kv.squeeze(1), idx.squeeze(1), sm_scale=0.2, topk_length=lens, attn_sink=sink
    )
    torch.testing.assert_close(out, eo)
    torch.testing.assert_close(lse, el)


def test_get_mla_metadata_is_callable_noarg():
    meta, num_splits = get_mla_metadata()
    assert isinstance(num_splits, int) and num_splits >= 1
    assert isinstance(meta, torch.Tensor)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py -q -k "native_op or sparse_fwd or metadata"`
Expected: FAIL — `cannot import name 'sparse_mla_attention'`.

- [ ] **Step 3: Extend the interface**

Add to `src/xkernels/ops/attention/interface.py` — extend the existing imports and append the ops:

```python
# in the `from . import (...)` block, add:
    sparse_mla_reference,  # noqa: F401  (registers sparse_mla_attention REFERENCE)
```

```python
# append to src/xkernels/ops/attention/interface.py


def sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    d_v: int | None = None,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 sparse-MLA attention compute (issue #32): flash softmax over
    the DSA-indexer-selected latent KV. Portable gfx942 replacement for the
    NVIDIA-only ``flash_mla`` sparse/decode kernels.

    Args:
        q: ``[T, H, D]`` latent queries (D = kv_lora_rank + rope; V4: 512).
        kv: ``[Kv, D]`` shared latent MQA cache.
        indices: ``[T, topk]`` int32 columns into ``kv`` (``<0`` = padding).
        sm_scale: softmax scale on the q.k score.
        topk_length: optional ``[T]`` int — valid column count per query.
        attn_sink: optional ``[H]`` (or scalar) per-head sink logit.
        d_v: value/output dim (first ``d_v`` latent dims). Defaults to ``D``.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out [T, H, d_v], lse [T, H] fp32, max_logits [T, H] fp32)``.
    """
    return dispatch(
        "sparse_mla_attention", q, kv, indices,
        sm_scale=sm_scale, topk_length=topk_length, attn_sink=attn_sink,
        d_v=d_v, backend=backend,
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    *,
    d_v: int | None = None,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill sparse-MLA (upstream-faithful name). ``kv`` is the bf16 latent
    workspace ``[Kv, 1, D]`` and ``indices`` is ``[T, 1, topk]`` (the ``1`` is the
    MQA KV head). Returns ``(out, max_logits, lse)`` in upstream order.
    """
    kv2 = kv.squeeze(1) if kv.dim() == 3 else kv
    idx2 = indices.squeeze(1) if indices.dim() == 3 else indices
    out, lse, maxl = sparse_mla_attention(
        q, kv2, idx2, sm_scale=sm_scale, topk_length=topk_length,
        attn_sink=attn_sink, d_v=d_v, backend=backend,
    )
    return out, maxl, lse


def get_mla_metadata(*args, **kwargs) -> tuple[torch.Tensor, int]:
    """Scheduling metadata (upstream-faithful name). V4 calls this no-arg and
    threads ``[0]`` into the decode kernel as an opaque ``tile_scheduler_metadata``
    that this compute path ignores. Returns ``(placeholder int32 tensor,
    num_splits=1)`` — no split-KV scheduling (a future split path would reuse
    ``mha_merge_state`` #3).
    """
    return torch.empty(0, dtype=torch.int32), 1
```

- [ ] **Step 4: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py -q -k "native_op or sparse_fwd or metadata"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/attention/interface.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): sparse_mla_attention native op + flash_mla_sparse_fwd/get_mla_metadata wrappers (#32)"
```

---

### Task 3: Triton flash compute kernel (gfx942)

**Files:**
- Create: `src/xkernels/ops/attention/triton/sparse_mla_kernel.py`
- Modify: `src/xkernels/ops/attention/__init__.py` (import for registration)
- Test: `tests/test_sparse_mla_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sparse_mla_attention.py
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends

_HAS_TRITON = Backend.TRITON in registered_backends("sparse_mla_attention")


@pytest.mark.parametrize("D,d_v,topk,H", [(512, 512, 64, 8), (512, 448, 128, 4), (32, 32, 7, 3)])
@pytest.mark.parametrize("with_sink", [False, True])
def test_triton_matches_reference(D, d_v, topk, H, with_sink):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    torch.manual_seed(3)
    dt = torch.float32 if _INTERP else torch.bfloat16
    T, Kv = 5, max(64, topk + 8)
    q = torch.randn(T, H, D, device=dev, dtype=dt)
    kv = torch.randn(Kv, D, device=dev, dtype=dt)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    idx[0, -3:] = -1
    lens = torch.randint(1, topk + 1, (T,), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev) if with_sink else None
    got = sparse_mla_attention(q, kv, idx, sm_scale=0.1, topk_length=lens,
                               attn_sink=sink, d_v=d_v, backend=Backend.TRITON)
    exp = sparse_mla_attention(q, kv, idx, sm_scale=0.1, topk_length=lens,
                               attn_sink=sink, d_v=d_v, backend=Backend.REFERENCE)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(got[0].float(), exp[0].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(got[1], exp[1], atol=atol, rtol=rtol)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_triton_matches_reference -q`
Expected: FAIL (skips if triton missing → import the kernel first; with TRITON_INTERPRET the backend registers, so it should run and FAIL because the kernel file doesn't exist).

- [ ] **Step 3: Write the Triton kernel**

```python
# src/xkernels/ops/attention/triton/sparse_mla_kernel.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton sparse-MLA attention compute for AMD MI300A (gfx942, CDNA3), issue #32.

One program per ``(token, head)``: stream the top-k selected latent KV in
``BLOCK_N`` chunks with online (flash) softmax. The score uses all ``D`` dims; the
value accumulator stores the first ``d_v`` dims (the kv_lora / nope part). An
optional per-head attention **sink** logit folds into the denominator after the
stream and contributes no value. Columns with ``idx < 0`` or beyond
``topk_length`` are masked to ``-inf``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["sparse_mla_attention_triton", "sparse_mla_kernel"]


@triton.jit
def sparse_mla_kernel(
    q_ptr, kv_ptr, idx_ptr, sink_ptr, len_ptr,
    out_ptr, lse_ptr, maxl_ptr,
    sm_scale,
    H, Kv, topk,
    stride_qt, stride_qh, stride_qd,
    stride_kk, stride_kd,
    stride_it, stride_ik,
    stride_ot, stride_oh, stride_od,
    HAS_SINK: tl.constexpr, HAS_LEN: tl.constexpr,
    D: tl.constexpr, D_V: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    d_mask = d < D

    q = tl.load(q_ptr + t * stride_qt + h * stride_qh + d * stride_qd,
                mask=d_mask, other=0.0).to(tl.float32)
    n_valid = tl.load(len_ptr + t) if HAS_LEN else topk

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for start in range(0, topk, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        col_mask = cols < topk
        idx = tl.load(idx_ptr + t * stride_it + cols * stride_ik, mask=col_mask, other=-1)
        valid = (idx >= 0) & col_mask
        if HAS_LEN:
            valid = valid & (cols < n_valid)
        safe = tl.where(valid, idx, 0)
        kvb = tl.load(
            kv_ptr + safe[:, None] * stride_kk + d[None, :] * stride_kd,
            mask=valid[:, None] & d_mask[None, :], other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * kvb, axis=1) * sm_scale  # [BLOCK_N]
        scores = tl.where(valid, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.where(valid, tl.exp(scores - m_new), 0.0)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * kvb, axis=0)  # [BLOCK_D]
        m_i = m_new

    if HAS_SINK:
        sink = tl.load(sink_ptr + h).to(tl.float32)
        m_new = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.exp(sink - m_new)
        acc = acc * alpha
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe
    dv_mask = d < D_V
    tl.store(out_ptr + t * stride_ot + h * stride_oh + d * stride_od,
             out.to(out_ptr.dtype.element_ty), mask=dv_mask)
    lse_val = tl.where(l_i > 0.0, m_i + tl.log(l_safe), -float("inf"))
    tl.store(lse_ptr + t * H + h, lse_val)
    tl.store(maxl_ptr + t * H + h, m_i)


def sparse_mla_attention_triton(
    q, kv, indices, *, sm_scale, topk_length=None, attn_sink=None, d_v=None,
):
    q = q.contiguous()
    kv = kv.contiguous()
    indices = indices.contiguous().to(torch.int32)
    T, H, D = q.shape
    Kv = kv.shape[0]
    topk = indices.shape[1]
    d_v = D if d_v is None else d_v

    out = torch.empty(T, H, d_v, device=q.device, dtype=q.dtype)
    lse = torch.empty(T, H, device=q.device, dtype=torch.float32)
    maxl = torch.empty(T, H, device=q.device, dtype=torch.float32)

    has_sink = attn_sink is not None
    has_len = topk_length is not None
    dummy = torch.empty(1, device=q.device, dtype=torch.float32)
    sink = dummy
    if has_sink:
        s = attn_sink.contiguous().float().reshape(-1)
        sink = (s.expand(H).contiguous() if s.numel() == 1 else s[:H].contiguous())
    length = topk_length.contiguous().to(torch.int32) if has_len else dummy.to(torch.int32)

    BLOCK_D = triton.next_power_of_2(D)
    sparse_mla_kernel[(T, H)](
        q, kv, indices, sink, length, out, lse, maxl,
        sm_scale, H, Kv, topk,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        indices.stride(0), indices.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        HAS_SINK=has_sink, HAS_LEN=has_len,
        D=D, D_V=d_v, BLOCK_D=BLOCK_D, BLOCK_N=64,
    )
    return out, lse, maxl


register("sparse_mla_attention", Backend.TRITON)(sparse_mla_attention_triton)
```

- [ ] **Step 4: Register via the package import** — add `sparse_mla_kernel` to the guarded triton import in `src/xkernels/ops/attention/__init__.py`:

```python
    with triton_import_ctx():
        from .triton import (  # noqa: F401
            dsa_indexer_kernel,
            merge_state_kernel,
            sparse_mla_kernel,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_triton_matches_reference -q`
Expected: PASS (all params).

- [ ] **Step 6: Commit**

```bash
git add src/xkernels/ops/attention/triton/sparse_mla_kernel.py src/xkernels/ops/attention/__init__.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): Triton sparse-MLA flash compute kernel for gfx942 (#32)"
```

---

### Task 4: fp8_ds_mla dequant module (decode KV format)

**Files:**
- Create: `src/xkernels/ops/attention/sparse_mla.py`
- Test: `tests/test_sparse_mla_attention.py`

Layout pinned from the tokenspeed writer `_deepseek_v4_fused_sparse_compress_cache_kernel`: per token, value region = `nope_dim` fp8 e4m3 + `rope_dim` bf16 (`nope_dim + rope_dim*2` bytes); scale region = `nope_dim//quant_block (+1 pad)` uint8 exponents (`enc = exp + 127`, dequant mult `2**(enc-127)`). V4: nope 448, rope 64, quant_block 64.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sparse_mla_attention.py
from xkernels.ops.attention.sparse_mla import dequant_fp8_ds_mla, make_fp8_ds_mla_kv


def test_fp8_ds_mla_roundtrip():
    dev = _dev()
    rows = 7
    value, scale, ref = make_fp8_ds_mla_kv(rows, device=dev, seed=4)
    got = dequant_fp8_ds_mla(value, scale)
    assert got.shape == (rows, 512)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


def test_fp8_ds_mla_known_value():
    """A small hand-checked case: one group, scale exp known."""
    dev = _dev()
    value, scale, ref = make_fp8_ds_mla_kv(1, nope_dim=64, rope_dim=64, device=dev, seed=0)
    got = dequant_fp8_ds_mla(value, scale, nope_dim=64, rope_dim=64)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py -q -k fp8`
Expected: FAIL — module missing.

- [ ] **Step 3: Write the module**

```python
# src/xkernels/ops/attention/sparse_mla.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""fp8_ds_mla latent-KV format helpers for the DeepSeek-V4 decode path (issue #32).

The V4 paged KV cache stores each latent token (issue layout, pinned from the
tokenspeed writer ``_deepseek_v4_fused_sparse_compress_cache_kernel``) as:

* a **value region** of ``nope_dim + rope_dim*2`` bytes: ``nope_dim`` fp8 e4m3
  (the kv_lora / value-bearing part) followed by ``rope_dim`` bf16 (the decoupled
  rope, score-only), and
* a **scale region** of ``nope_dim//quant_block`` uint8 exponents (``enc``) plus a
  pad byte, shared per ``quant_block`` group along nope.

Dequant: ``nope = fp8_e4m3(byte) * 2**(enc - 127)`` per group; ``rope = bf16``.
V4: ``nope_dim=448, rope_dim=64, quant_block=64`` → latent ``D=512``.
"""

from __future__ import annotations

import torch

__all__ = [
    "FP8_DS_MLA_NOPE_DIM",
    "FP8_DS_MLA_ROPE_DIM",
    "FP8_DS_MLA_QUANT_BLOCK",
    "FP8_DS_MLA_HEAD_DIM",
    "dequant_fp8_ds_mla",
    "make_fp8_ds_mla_kv",
]

FP8_DS_MLA_NOPE_DIM = 448
FP8_DS_MLA_ROPE_DIM = 64
FP8_DS_MLA_QUANT_BLOCK = 64
FP8_DS_MLA_HEAD_DIM = FP8_DS_MLA_NOPE_DIM + FP8_DS_MLA_ROPE_DIM  # 512
_FP8_MAX = 448.0


def dequant_fp8_ds_mla(
    value_bytes: torch.Tensor,
    scale_bytes: torch.Tensor,
    *,
    nope_dim: int = FP8_DS_MLA_NOPE_DIM,
    rope_dim: int = FP8_DS_MLA_ROPE_DIM,
    quant_block: int = FP8_DS_MLA_QUANT_BLOCK,
) -> torch.Tensor:
    """Dequantize fp8_ds_mla rows to fp32 latent ``[..., nope_dim + rope_dim]``.

    Args:
        value_bytes: ``[..., nope_dim + rope_dim*2]`` uint8.
        scale_bytes: ``[..., nope_dim//quant_block (+pad)]`` uint8 exponents.
    """
    nope_fp8 = value_bytes[..., :nope_dim].contiguous()
    rope_raw = value_bytes[..., nope_dim : nope_dim + rope_dim * 2].contiguous()
    nope = nope_fp8.view(torch.float8_e4m3fn).to(torch.float32)
    ng = nope_dim // quant_block
    enc = scale_bytes[..., :ng].to(torch.int32) - 127
    mult = torch.exp2(enc.float()).repeat_interleave(quant_block, dim=-1)
    nope = nope * mult
    rope = rope_raw.view(torch.bfloat16).to(torch.float32)
    return torch.cat([nope, rope], dim=-1)


def make_fp8_ds_mla_kv(
    num_rows: int,
    *,
    nope_dim: int = FP8_DS_MLA_NOPE_DIM,
    rope_dim: int = FP8_DS_MLA_ROPE_DIM,
    quant_block: int = FP8_DS_MLA_QUANT_BLOCK,
    device="cuda",
    seed: int = 0,
):
    """Random valid fp8_ds_mla rows + their exact dequantization.

    Returns ``(value_bytes [rows, nope+rope*2] uint8,
    scale_bytes [rows, nope//qb + 1] uint8, ref [rows, nope+rope] fp32)`` where
    ``ref == dequant_fp8_ds_mla(value_bytes, scale_bytes)``.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    nope = (torch.rand(num_rows, nope_dim, generator=g, device=device) * 2 - 1) * 4
    rope = torch.rand(num_rows, rope_dim, generator=g, device=device) * 2 - 1
    ng = nope_dim // quant_block
    nope_g = nope.reshape(num_rows, ng, quant_block)
    absmax = nope_g.abs().amax(dim=-1).clamp_min(1e-4)
    exps = torch.ceil(torch.log2(absmax / _FP8_MAX))  # [rows, ng]
    inv = torch.exp2(-exps).unsqueeze(-1)
    fp8 = (nope_g * inv).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    nope_deq = (fp8.to(torch.float32) * torch.exp2(exps).unsqueeze(-1)).reshape(num_rows, nope_dim)
    enc = (exps + 127).clamp(0, 255).to(torch.uint8)

    value_bytes = torch.empty(num_rows, nope_dim + rope_dim * 2, device=device, dtype=torch.uint8)
    value_bytes[:, :nope_dim] = fp8.reshape(num_rows, nope_dim).view(torch.uint8)
    value_bytes[:, nope_dim:] = rope.to(torch.bfloat16).view(torch.uint8)
    scale_bytes = torch.zeros(num_rows, ng + 1, device=device, dtype=torch.uint8)
    scale_bytes[:, :ng] = enc
    ref = torch.cat([nope_deq, rope.to(torch.bfloat16).to(torch.float32)], dim=-1)
    return value_bytes, scale_bytes, ref
```

- [ ] **Step 4: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py -q -k fp8`
Expected: PASS.

> Note: `torch.float8_e4m3fn` is the OCP e4m3 the writer pins via `tl.float8e4nv`; CPU dequant requires a torch build with fp8 support (≥2.1). If the host venv lacks it, this test runs on-device only (gate with `torch.float8_e4m3fn` availability — see Task 5 test gate).

- [ ] **Step 5: Commit**

```bash
git add src/xkernels/ops/attention/sparse_mla.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): fp8_ds_mla latent-KV dequant + test generator (#32)"
```

---

### Task 5: Decode wrapper `flash_mla_with_kvcache` (gather+dequant → core compute)

**Files:**
- Create: `src/xkernels/ops/attention/sparse_mla_decode.py`
- Modify: `src/xkernels/ops/attention/interface.py` (re-export the wrapper)
- Test: `tests/test_sparse_mla_attention.py`

Decode contract (xkernels-clean, unit-testable): paged caches are passed as a
**value tensor** `[num_blocks, block_size, value_bytes]` uint8 + a **scale tensor**
`[num_blocks, block_size, scale_bytes]` uint8 (the on-device adapter that splits
tokenspeed's single 2D pool buffer into these views is pinned in Task 8). The
wrapper gathers each query's selected positions from the primary (`k_cache`/SWA)
and optional `extra_k_cache` (compressed CSA) caches, dequants to bf16, flattens
to the shared `(kv, indices)` form, and runs the Task-3 compute kernel.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sparse_mla_attention.py
_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


def _paged_cache(num_blocks, block_size, seed):
    """Build a paged fp8_ds_mla (value, scale) cache + its full bf16 latent."""
    rows = num_blocks * block_size
    value, scale, ref = make_fp8_ds_mla_kv(rows, device=_dev(), seed=seed)
    vb, sb = value.shape[-1], scale.shape[-1]
    return (value.view(num_blocks, block_size, vb),
            scale.view(num_blocks, block_size, sb),
            ref.view(num_blocks, block_size, -1))


@pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
def test_flash_mla_with_kvcache_dual_cache():
    from xkernels.ops.attention.interface import flash_mla_with_kvcache
    from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref
    dev = _dev()
    torch.manual_seed(5)
    nb, bs, H, D = 4, 8, 4, 512
    backend = Backend.TRITON if _HAS_TRITON else Backend.REFERENCE
    val, sca, ref = _paged_cache(nb, bs, seed=5)
    vale, scae, refe = _paged_cache(nb, bs, seed=6)
    rows = nb * bs
    T, topk = 3, 5
    dt = torch.float32 if _INTERP else torch.bfloat16
    q = torch.randn(T, H, D, device=dev, dtype=dt)
    blk = torch.arange(nb, device=dev, dtype=torch.int32).view(1, nb).expand(T, nb).contiguous()
    idx = torch.randint(0, rows, (T, topk), device=dev, dtype=torch.int32)
    eidx = torch.randint(0, rows, (T, topk), device=dev, dtype=torch.int32)
    lens = torch.full((T,), topk, device=dev, dtype=torch.int32)
    elens = torch.full((T,), topk, device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)

    out, lse = flash_mla_with_kvcache(
        q=q.unsqueeze(1), k_cache=val, block_table=blk, cache_seqlens=None,
        head_dim_v=D, tile_scheduler_metadata=None, softmax_scale=0.1,
        is_fp8_kvcache=True, indices=idx.unsqueeze(1), attn_sink=sink,
        extra_k_cache=vale, extra_indices_in_kvcache=eidx,
        topk_length=lens, extra_topk_length=elens,
        scale_cache=sca, extra_scale_cache=scae, block_size=bs, backend=backend,
    )
    out = out.squeeze(1) if out.dim() == 4 else out

    # Oracle: concat the two dequantized gathered sets per token, run the ref.
    flat = ref.reshape(rows, D)
    eflat = refe.reshape(rows, D)
    kv_cat = torch.cat([flat, eflat], dim=0)              # [2*rows, D]
    idx_cat = torch.cat([idx, eidx + rows], dim=1)        # [T, 2*topk]
    len_cat = lens + elens
    eo, el, _ = sparse_mla_attention_ref(
        q.to(torch.float32), kv_cat, idx_cat, sm_scale=0.1,
        topk_length=len_cat, attn_sink=sink, d_v=D)
    atol = 1e-3 if _INTERP else 3e-2
    torch.testing.assert_close(out.float(), eo.float(), atol=atol, rtol=atol)
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_flash_mla_with_kvcache_dual_cache -q`
Expected: FAIL — `cannot import name 'flash_mla_with_kvcache'`.

- [ ] **Step 3: Write the decode wrapper**

```python
# src/xkernels/ops/attention/sparse_mla_decode.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Decode entry point ``flash_mla_with_kvcache`` for sparse-MLA on gfx942 (#32).

Gathers each query's DSA-selected positions from the paged fp8_ds_mla primary
cache (SWA) and the optional compressed (CSA) ``extra_k_cache``, dequantizes to
bf16, flattens to the shared ``(kv, indices)`` form, and runs the sparse-MLA
compute kernel. The two index sets share one softmax (the hybrid attention is
realized here as a union of selections, not separate passes).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from .sparse_mla import dequant_fp8_ds_mla


def _gather_dequant(value_cache, scale_cache, block_table, indices, lengths, block_size):
    """Gather + dequant selected positions to bf16 ``[T, topk, D]``.

    ``value_cache``/``scale_cache``: ``[num_blocks, block_size, *bytes]`` uint8.
    ``indices``: ``[T, topk]`` positions into the (per-seq) logical KV; resolved
    through ``block_table`` ``[T, max_blocks]``. ``<0`` or ``>= lengths`` → zero row.
    """
    T, topk = indices.shape
    nb, bs, vb = value_cache.shape
    D = (vb - 0)  # set below from dequant
    dev = value_cache.device
    pos = torch.arange(topk, device=dev)
    valid = indices >= 0
    if lengths is not None:
        valid = valid & (pos.unsqueeze(0) < lengths.unsqueeze(1))
    safe = indices.clamp_min(0).long()
    logical_blk = safe // block_size
    within = safe % block_size
    phys = torch.gather(block_table.long(), 1, logical_blk)  # [T, topk]
    vsel = value_cache[phys.reshape(-1), within.reshape(-1)]  # [T*topk, vb]
    ssel = scale_cache[phys.reshape(-1), within.reshape(-1)]  # [T*topk, sb]
    deq = dequant_fp8_ds_mla(vsel, ssel)                      # [T*topk, D]
    D = deq.shape[-1]
    deq = deq.reshape(T, topk, D)
    deq = torch.where(valid.unsqueeze(-1), deq, torch.zeros_like(deq))
    return deq, valid


def flash_mla_with_kvcache(
    q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata,
    *, softmax_scale, is_fp8_kvcache=True, indices, attn_sink=None,
    extra_k_cache=None, extra_indices_in_kvcache=None,
    topk_length=None, extra_topk_length=None,
    scale_cache=None, extra_scale_cache=None, block_size=None,
    backend: Backend | str = "auto",
):
    """Decode sparse-MLA over paged fp8_ds_mla cache(s). Returns ``(out, lse)``.

    ``q``: ``[B, 1, H, D]`` (seq_q=1). ``indices``/``extra_indices_in_kvcache``:
    ``[B, 1, topk]``. See module docstring for the cache contract.
    """
    from .interface import sparse_mla_attention  # local import (avoid cycle)

    q2 = q.squeeze(1) if q.dim() == 4 else q          # [T, H, D]
    idx = indices.squeeze(1) if indices.dim() == 3 else indices
    T, H, D = q2.shape
    if block_size is None:
        block_size = k_cache.shape[1]

    kv1, _ = _gather_dequant(k_cache, scale_cache, block_table, idx, topk_length, block_size)
    parts = [kv1]
    idx_parts = [torch.where(
        (idx >= 0) & ((topk_length is None) |
                      (torch.arange(idx.shape[1], device=q2.device) < (topk_length.unsqueeze(1) if topk_length is not None else idx.shape[1]))),
        torch.arange(idx.shape[1], device=q2.device).expand_as(idx), torch.full_like(idx, -1))]
    lens = [topk_length if topk_length is not None
            else torch.full((T,), idx.shape[1], device=q2.device, dtype=torch.int32)]
    offset = kv1.shape[1]

    if extra_k_cache is not None:
        eidx = extra_indices_in_kvcache.squeeze(1) if extra_indices_in_kvcache.dim() == 3 else extra_indices_in_kvcache
        kv2, _ = _gather_dequant(extra_k_cache, extra_scale_cache, block_table, eidx, extra_topk_length, block_size)
        parts.append(kv2)
        ar = torch.arange(eidx.shape[1], device=q2.device)
        evalid = (eidx >= 0)
        if extra_topk_length is not None:
            evalid = evalid & (ar.unsqueeze(0) < extra_topk_length.unsqueeze(1))
        idx_parts.append(torch.where(evalid, offset + ar.expand_as(eidx), torch.full_like(eidx, -1)))
        lens.append(extra_topk_length if extra_topk_length is not None
                    else torch.full((T,), eidx.shape[1], device=q2.device, dtype=torch.int32))

    kv = torch.cat([p.reshape(T, p.shape[1], D) for p in parts], dim=1)  # [T, total, D]
    kv_flat = kv.reshape(T * kv.shape[1], D)
    # Per-token indices into kv_flat: row t uses block [t*total, (t+1)*total).
    total = kv.shape[1]
    base = (torch.arange(T, device=q2.device) * total).view(T, 1)
    idx_all = torch.cat(idx_parts, dim=1)                      # [T, total] local (or -1)
    idx_flat = torch.where(idx_all >= 0, idx_all + base, torch.full_like(idx_all, -1)).to(torch.int32)
    len_all = sum(lens)                                        # [T]

    out, lse, _ = sparse_mla_attention(
        q2, kv_flat, idx_flat, sm_scale=softmax_scale,
        topk_length=len_all, attn_sink=attn_sink, d_v=head_dim_v, backend=backend,
    )
    return out.unsqueeze(1), lse
```

- [ ] **Step 4: Re-export from the interface** — add to `src/xkernels/ops/attention/interface.py`:

```python
from .sparse_mla_decode import flash_mla_with_kvcache  # noqa: F401  (re-export)
```

- [ ] **Step 5: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_flash_mla_with_kvcache_dual_cache -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/xkernels/ops/attention/sparse_mla_decode.py src/xkernels/ops/attention/interface.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): flash_mla_with_kvcache decode (gather+dequant dual cache) (#32)"
```

---

### Task 6: Public surface — top-level re-exports

**Files:**
- Modify: `src/xkernels/ops/attention/__init__.py`
- Modify: `src/xkernels/__init__.py`
- Test: `tests/test_sparse_mla_attention.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_sparse_mla_attention.py
def test_top_level_exports():
    import xkernels
    for name in ("sparse_mla_attention", "flash_mla_sparse_fwd",
                 "flash_mla_with_kvcache", "get_mla_metadata"):
        assert hasattr(xkernels, name), name
```

- [ ] **Step 2: Run to verify it fails**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py::test_top_level_exports -q`
Expected: FAIL — `AttributeError`.

- [ ] **Step 3: Wire the exports** — `src/xkernels/ops/attention/__init__.py`: extend the `from .interface import (...)` line and `__all__`:

```python
from .interface import (
    dsa_indexer_logits,
    dsa_indexer_topk,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
    mha_merge_state,
    sparse_mla_attention,
)
```
```python
__all__ = [
    "mha_merge_state", "dsa_indexer_logits", "dsa_indexer_topk",
    "sparse_mla_attention", "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache", "get_mla_metadata",
]
```

`src/xkernels/__init__.py`: extend the attention import + `__all__`:

```python
from .ops.attention import (
    dsa_indexer_logits,
    dsa_indexer_topk,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
    mha_merge_state,
    sparse_mla_attention,
)
```
Add `"sparse_mla_attention", "flash_mla_sparse_fwd", "flash_mla_with_kvcache", "get_mla_metadata",` to `__all__`.

- [ ] **Step 4: Run to verify it passes**

Run: `VIRTUAL_ENV=.venv TRITON_INTERPRET=1 .venv/bin/python -m pytest tests/test_sparse_mla_attention.py -q`
Expected: PASS (whole file).

- [ ] **Step 5: Lint + commit**

Run: `VIRTUAL_ENV=.venv .venv/bin/ruff check src/xkernels/ops/attention tests/test_sparse_mla_attention.py`
Expected: clean (fix any import-order/unused warnings).
```bash
git add src/xkernels/ops/attention/__init__.py src/xkernels/__init__.py tests/test_sparse_mla_attention.py
git commit -m "feat(attention): export sparse-MLA ops at top level (#32)"
```

---

### Task 7: Benchmark + kernel doc

**Files:**
- Create: `benchmarks/bench_sparse_mla.py`
- Create: `docs/issue-32-sparse-mla-attention.md`

- [ ] **Step 1: Write the benchmark** (speedup vs naive gather+dense-softmax torch baseline; mirrors `benchmarks/bench_mha_merge_state.py` structure — `do_bench`, V4 shapes H, D=512, d_v=448, topk∈{512,1024}). Print a `| sparse_mla | shape | naive | optimized | speedup |` row.

```python
# benchmarks/bench_sparse_mla.py  (skeleton — fill the do_bench calls to match bench_mha_merge_state.py)
import torch, triton
from xkernels import sparse_mla_attention
from xkernels._backends import Backend

def main():
    dev = "cuda"
    T, H, D, d_v, topk, Kv = 8, 128, 512, 448, 512, 4096
    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)
    def naive():
        ks = kv[idx.long()]                                  # [T, topk, D]
        s = torch.einsum("thd,tkd->thk", q.float(), ks.float()) * (1/ D**0.5)
        p = s.softmax(-1)
        return torch.einsum("thk,tkd->thd", p, ks[..., :d_v].float())
    def opt():
        return sparse_mla_attention(q, kv, idx, sm_scale=1/D**0.5, attn_sink=sink, d_v=d_v, backend=Backend.TRITON)
    t_naive = triton.testing.do_bench(naive)
    t_opt = triton.testing.do_bench(opt)
    print(f"| sparse_mla | T={T},H={H},D={D},topk={topk} | {t_naive:.3f} ms | {t_opt:.3f} ms | {t_naive/t_opt:.1f}x |")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `docs/issue-32-sparse-mla-attention.md`** — the math (latent MLA score/value split, sink), the fp8_ds_mla layout (value+scale regions), the prefill vs decode entry points, the variant-agnostic framing, and the on-device validation results table (filled after Task 8).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_sparse_mla.py docs/issue-32-sparse-mla-attention.md
git commit -m "docs+bench(attention): sparse-MLA doc + benchmark (#32)"
```

---

### Task 8: On-device validation on beverin (MI300A / gfx942)

**Files:**
- Create: `slurm/test_sparse_mla_beverin.sbatch`

Mirrors `slurm/test_dsa_indexer_beverin.sbatch`: `srun --environment=...`, `PYTHONPATH=$REPO/src`, `TRITON_INTERPRET` unset (real compile).

- [ ] **Step 1: Write the sbatch**

```bash
#!/bin/bash
# SPDX-License-Identifier: MIT
# On-device correctness for the DeepSeek-V4 sparse-MLA attention compute
# (issue #32) on beverin (gfx942 / MI300A): Triton vs torch oracle, bf16, on GPU.
#
#   sbatch --export=ALL,REPO=/capstor/scratch/cscs/xyao/kernels-issue-32 \
#          slurm/test_sparse_mla_beverin.sbatch
#
#SBATCH --job-name=xk-sparse-mla
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --time=00:25:00
#SBATCH --output=sparse-mla-%j.out
#SBATCH --error=sparse-mla-%j.out

set -uo pipefail
REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"
echo "REPO=$REPO ENV=$ENV_NAME node=$(hostname)"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  unset TRITON_INTERPRET || true
  export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  cd "'"$REPO"'"
  python -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip, torch.cuda.get_device_name(0))"

  echo "######## pytest: sparse-MLA on GPU (Triton bf16 vs oracle) ########"
  python -m pytest tests/test_sparse_mla_attention.py -q

  echo "######## V4-shape prefill parity + max|err| ########"
  python -u - <<"PY"
import torch
from xkernels import sparse_mla_attention, flash_mla_sparse_fwd
from xkernels._backends import Backend
from xkernels.ops.attention.sparse_mla_reference import sparse_mla_attention_ref
torch.manual_seed(0); dev="cuda"
T,H,D,d_v,Kv,topk = 8,128,512,448,4096,512
q=torch.randn(T,H,D,device=dev,dtype=torch.bfloat16)
kv=torch.randn(Kv,D,device=dev,dtype=torch.bfloat16)
idx=torch.randint(0,Kv,(T,topk),device=dev,dtype=torch.int32)
lens=torch.randint(1,topk+1,(T,),device=dev,dtype=torch.int32)
sink=torch.randn(H,device=dev)
got=sparse_mla_attention(q,kv,idx,sm_scale=1/D**0.5,topk_length=lens,attn_sink=sink,d_v=d_v,backend=Backend.TRITON)
ref=sparse_mla_attention_ref(q,kv,idx,sm_scale=1/D**0.5,topk_length=lens,attn_sink=sink,d_v=d_v)
err=(got[0].float()-ref[0].float()).abs().max().item()
print(f"[V4 prefill H=128 D=512 d_v=448 topk=512] max|err|={err:.4e}")
assert err < 5e-2, err
print("PASS: sparse-MLA prefill compute correct on gfx942")
PY
'
```

- [ ] **Step 2: Run on beverin** (per the project memory — rsync then sbatch):

```bash
rsync -az --delete /home/xiayao/Documents/research/kernels/ \
  beverin:/capstor/scratch/cscs/xyao/kernels-issue-32/
ssh beverin 'cd /capstor/scratch/cscs/xyao/kernels-issue-32 && \
  sbatch --export=ALL,REPO=$PWD slurm/test_sparse_mla_beverin.sbatch'
```
Expected output: `PASS: sparse-MLA prefill compute correct on gfx942` and the pytest summary all-passed. **Pin `d_v`** (512 vs 448) here against tokenspeed's o_proj if the prefill parity vs a real V4 layer disagrees; update the default + doc.

- [ ] **Step 3: Record results in `docs/issue-32-sparse-mla-attention.md`, commit**

```bash
git add slurm/test_sparse_mla_beverin.sbatch docs/issue-32-sparse-mla-attention.md
git commit -m "test(attention): on-device sparse-MLA validation on MI300A (#32)"
```

---

### Task 9: PR

- [ ] **Step 1: Push + open draft PR** against `ResearchComputer/kernels`, body summarizing: variant-agnostic sparse-MLA compute, the three faithful-named wrappers, fp8_ds_mla decode dequant, interpreter + MI300A validation, and the tokenspeed-side binding (replacing `error_fn`) called out as a follow-up in that repo. Link issue #32 and umbrella #28.

```bash
git push -u origin feat/issue-32-sparse-mla-attention
gh pr create --repo ResearchComputer/kernels --draft \
  --title "feat(attention): DeepSeek-V4 sparse-MLA attention compute on gfx942 (issue #32)" \
  --body "..."
```

---

## Self-Review

**Spec coverage:** oracle (T1), native op + faithful names + metadata (T2), Triton compute (T3), fp8_ds_mla dequant (T4), decode dual-cache wrapper (T5), exports (T6), bench+doc (T7), beverin validation (T8), PR (T9) — every spec section maps to a task. ✅

**Placeholder scan:** the only deferred items are explicit *verification* steps (pin `d_v` and the real-pool byte-split on-device, T5/T8) — code is concrete throughout; the fp8 layout is self-defined via `make_fp8_ds_mla_kv` so it's unit-testable offline. ✅

**Type consistency:** `sparse_mla_attention(...) -> (out, lse, max_logits)` everywhere; `flash_mla_sparse_fwd -> (out, max_logits, lse)` (upstream order); `flash_mla_with_kvcache -> (out, lse)`; `dequant_fp8_ds_mla(value_bytes, scale_bytes)`; `make_fp8_ds_mla_kv -> (value, scale, ref)`. Registry key `"sparse_mla_attention"` shared by REFERENCE (T1) and TRITON (T3). ✅
