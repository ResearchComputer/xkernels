# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: DeepSeek-V4 DSA indexer logits (issue #27) backends vs the
weighted-ReLU MQA torch oracle, plus end-to-end top-k selection parity.

Runs on GPU (bf16 q/k) or CPU via ``TRITON_INTERPRET=1`` (fp32). The oracle is
``dsa_indexer_logits_ref`` (identical math to the upstream
``_indexer_topk_reference`` in tokenspeed's deepseek_v4 attention-ops test).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import dsa_indexer_logits, dsa_indexer_topk
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention.dsa_reference import dsa_indexer_logits_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("dsa_indexer_logits")


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _inputs(T, H, D, K, dtype, dev, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(T, H, D, device=dev, dtype=dtype)
    k = torch.randn(K, D, device=dev, dtype=dtype)
    # Non-negative per-head weights, as produced by the V4 indexer (softplus-ish).
    weights = torch.rand(T, H, device=dev, dtype=torch.float32) + 0.1
    return q, k, weights


# Includes the real V4 indexer shape (H=64 index heads, D=128 index_head_dim).
@pytest.mark.parametrize(
    "T,H,D,K",
    [(4, 16, 32, 96), (2, 64, 128, 128), (3, 32, 64, 200)],
)
def test_triton_logits_match_reference(T, H, D, K):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    q, k, weights = _inputs(T, H, D, K, dtype, dev)
    got = dsa_indexer_logits(q, k, weights, backend=Backend.TRITON)
    ref = dsa_indexer_logits_ref(q, k, weights)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got, ref, atol=atol, rtol=rtol)


def test_triton_logits_match_reference_with_causal_mask():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, K = 4, 16, 64, 128
    q, k, weights = _inputs(T, H, D, K, dtype, dev)
    # Each query sees a distinct causal window [start, start+length).
    row_starts = torch.tensor([0, 8, 0, 16], device=dev, dtype=torch.int32)
    lengths = torch.tensor([64, 32, 128, 80], device=dev, dtype=torch.int32)
    got = dsa_indexer_logits(
        q, k, weights, lengths=lengths, row_starts=row_starts, backend=Backend.TRITON
    )
    ref = dsa_indexer_logits_ref(q, k, weights, lengths=lengths, row_starts=row_starts)
    # -inf entries must match exactly; finite entries within tolerance.
    masked = torch.isinf(ref) & (ref < 0)
    assert torch.equal(torch.isinf(got) & (got < 0), masked)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got[~masked], ref[~masked], atol=atol, rtol=rtol)


def test_topk_selection_matches_reference():
    """End-to-end: Triton logits + top-k select the same KV as the oracle."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, K, topk = 3, 32, 64, 256, 16
    q, k, weights = _inputs(T, H, D, K, dtype, dev, seed=7)
    got_logits = dsa_indexer_logits(q, k, weights, backend=Backend.TRITON)
    ref_logits = dsa_indexer_logits_ref(q, k, weights)
    got_idx = dsa_indexer_topk(got_logits, topk)
    ref_idx = dsa_indexer_topk(ref_logits, topk)
    # top-k is order-independent and robust to small logit noise: compare as sets.
    got_sets = [set(row.tolist()) for row in got_idx.cpu()]
    ref_sets = [set(row.tolist()) for row in ref_idx.cpu()]
    assert got_sets == ref_sets


def test_reference_backend_matches_oracle():
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    q, k, weights = _inputs(2, 16, 32, 64, dtype, dev)
    got = dsa_indexer_logits(q, k, weights, backend=Backend.REFERENCE)
    ref = dsa_indexer_logits_ref(q, k, weights)
    torch.testing.assert_close(got, ref)


def test_relu_gate_is_applied():
    """Negative q.k contributions must be clamped to zero (ReLU), not summed."""
    dev = _device()
    # One head, one KV: q.k < 0 -> relu -> 0 -> logit 0, not negative.
    q = torch.full((1, 16, 16), -1.0, device=dev, dtype=torch.float32)
    k = torch.full((1, 16), 1.0, device=dev, dtype=torch.float32)
    weights = torch.ones(1, 16, device=dev, dtype=torch.float32)
    ref = dsa_indexer_logits_ref(q, k, weights)
    assert torch.all(ref == 0.0)
