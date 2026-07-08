# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: fused DSA indexer top-k (issue #54) — dsa_indexer_topk vs the
canonical-argsort oracle ``dsa_indexer_topk_ref``.

The output is ``[T, topk]`` int32 indices, so correctness is judged by SET
EQUALITY (not element-wise float tolerance): a different KV position is
abs_err >= 1 and fails any tolerance, but for bf16/fp16 inputs a 1-ULP fp32
accumulation-order flip can swap a near-tie position in/out of the top-k —
the selected SET is still correct. For fp32 inputs with well-separated scores,
the selection is element-wise exact. See registry/ops/dsa_indexer_topk.spec.json
numerics.notes.
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import dsa_indexer_topk
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention.dsa_reference import (
    dsa_indexer_topk_ref,
)
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("dsa_indexer_topk")


def _inputs(T, H, D, K, dtype, dev, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, H, D, device=dev, dtype=dtype)
    k = torch.randn(K, D, device=dev, dtype=dtype)
    weights = torch.rand(T, H, device=dev, dtype=torch.float32) + 0.1
    return q, k, weights


def _to_sets(idx: torch.Tensor) -> list[set[int]]:
    return [set(row.tolist()) for row in idx.cpu()]


def test_reference_matches_oracle():
    """The reference backend (dsa_indexer_topk_ref dispatched) must be bit-exact
    vs the direct oracle call — same computation, same canonical argsort."""
    dev = "cpu"
    dtype = torch.float32
    q, k, weights = _inputs(4, 16, 64, 96, dtype, dev, seed=3)
    topk = 8
    got = dsa_indexer_topk(q, k, weights, topk=topk, backend=Backend.REFERENCE)
    ref = dsa_indexer_topk_ref(q, k, weights, topk=topk)
    torch.testing.assert_close(got, ref)


def test_reference_uses_canonical_argsort_not_torch_topk():
    """The reference must use a stable descending argsort (ties by ascending KV
    id), NOT torch.topk(sorted=False) whose tie-break is unspecified. With exact
    ties (all-zero logits), the reference must return ascending KV ids."""
    dev = "cpu"
    q = torch.zeros(2, 8, 16, device=dev, dtype=torch.float32)
    k = torch.zeros(32, 16, device=dev, dtype=torch.float32)
    weights = torch.ones(2, 8, device=dev, dtype=torch.float32)
    topk = 4
    got = dsa_indexer_topk(q, k, weights, topk=topk, backend=Backend.REFERENCE)
    # All logits are 0 (relu(0) = 0), so ties broken by ascending KV id.
    expected = torch.arange(topk, device=dev, dtype=torch.int32).unsqueeze(0).repeat(2, 1)
    torch.testing.assert_close(got, expected)


def test_topk_respects_causal_mask():
    """Columns outside [row_starts, row_starts+lengths) are masked to -inf and
    never selected — even when they would have the highest score."""
    dev = "cpu"
    dtype = torch.float32
    T, H, D, K = 2, 8, 16, 32
    q, k, weights = _inputs(T, H, D, K, dtype, dev, seed=5)
    # Query 0 sees only columns [0, 4); query 1 sees [16, 24).
    row_starts = torch.tensor([0, 16], device=dev, dtype=torch.int32)
    lengths = torch.tensor([4, 8], device=dev, dtype=torch.int32)
    topk = 3
    got = dsa_indexer_topk(
        q, k, weights, topk=topk, lengths=lengths, row_starts=row_starts,
        backend=Backend.REFERENCE,
    )
    ref = dsa_indexer_topk_ref(q, k, weights, topk=topk, lengths=lengths, row_starts=row_starts)
    torch.testing.assert_close(got, ref)
    # Every selected index falls within the valid window.
    for t in range(T):
        lo, hi = int(row_starts[t]), int(row_starts[t]) + int(lengths[t])
        for idx in got[t].tolist():
            assert lo <= idx < hi, f"row {t}: index {idx} outside [{lo}, {hi})"


def test_triton_matches_reference_fp32():
    """For fp32 inputs with well-separated scores, the Triton fused kernel must
    select the exact same indices (element-wise) as the reference."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.float32
    T, H, D, K, topk = 4, 16, 64, 96, 8
    q, k, weights = _inputs(T, H, D, K, dtype, dev, seed=11)
    got = dsa_indexer_topk(q, k, weights, topk=topk, backend=Backend.TRITON)
    ref = dsa_indexer_topk_ref(q, k, weights, topk=topk)
    torch.testing.assert_close(got, ref)


def test_triton_matches_reference_set_equality():
    """For bf16 inputs, the Triton fused kernel may select a different KV
    position than the reference on a near-tie (fp32 accumulation order), but the
    selected SET must match (the real correctness gate for index outputs)."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, K, topk = 3, 32, 64, 256, 16
    q, k, weights = _inputs(T, H, D, K, dtype, dev, seed=7)
    got = dsa_indexer_topk(q, k, weights, topk=topk, backend=Backend.TRITON)
    ref = dsa_indexer_topk_ref(q, k, weights, topk=topk)
    got_sets = _to_sets(got)
    ref_sets = _to_sets(ref)
    assert got_sets == ref_sets


def test_triton_matches_reference_with_causal_mask():
    """Fused top-k with causal mask: set-equality on bf16, same window."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, K, topk = 4, 16, 64, 128, 8
    q, k, weights = _inputs(T, H, D, K, dtype, dev, seed=13)
    row_starts = torch.tensor([0, 8, 0, 16], device=dev, dtype=torch.int32)
    lengths = torch.tensor([64, 32, 128, 80], device=dev, dtype=torch.int32)
    got = dsa_indexer_topk(
        q, k, weights, topk=topk, lengths=lengths, row_starts=row_starts,
        backend=Backend.TRITON,
    )
    ref = dsa_indexer_topk_ref(q, k, weights, topk=topk, lengths=lengths, row_starts=row_starts)
    # fp32 in interpreter mode -> element-wise; bf16 on GPU -> set-equality.
    if _INTERP:
        torch.testing.assert_close(got, ref)
    else:
        assert _to_sets(got) == _to_sets(ref)
