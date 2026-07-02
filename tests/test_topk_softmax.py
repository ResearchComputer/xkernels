# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: topk_softmax (issue #70) — fused MoE gating.

Covers the reference oracle, the Triton device kernel, the canonical tie-break
contract, and the runtime precondition. Runs on GPU (bf16/fp16) or CPU via
``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import topk_softmax
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe.topk_softmax import topk_softmax_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("topk_softmax")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Reference oracle — semantics + the canonical tie-break contract
# ═══════════════════════════════════════════════════════════════════════════════


def test_reference_semantics_descending_and_renorm():
    """Weights are descending per row; renormalize=True => rows sum to 1.0;
    renormalize=False => the raw softmax of the selected experts (< 1.0)."""
    dev = _device()
    torch.manual_seed(0)
    g = torch.randn(4, 16, device=dev, dtype=torch.float32)
    w, ids = topk_softmax_ref(g, 4, renormalize=True)
    assert w.shape == (4, 4) and ids.shape == (4, 4)
    assert w.dtype == torch.float32 and ids.dtype == torch.int32
    # descending within each row
    assert bool(((w[:, :-1] - w[:, 1:]) >= -1e-6).all())
    # renormalized -> rows sum to 1
    torch.testing.assert_close(w.sum(dim=1), torch.ones(4, device=dev), atol=1e-5, rtol=1e-5)
    # ids are valid expert indices in [0, E)
    assert bool(((ids >= 0) & (ids < 16)).all())
    # renormalize=False -> weights ARE the raw softmax of the selected experts
    # (gather consistency); sums < 1. Renormalize=True is just these divided by
    # their row sum, so we check gather-consistency on the False branch.
    w2, ids2 = topk_softmax_ref(g, 4, renormalize=False)
    probs = torch.softmax(g, dim=1)
    torch.testing.assert_close(w2, probs.gather(1, ids2.long()), atol=1e-6, rtol=1e-6)
    assert bool((w2.sum(dim=1) < 1.0).all())
    # and renormalize=True == renormalize=False / its row sum
    torch.testing.assert_close(w, w2 / w2.sum(dim=1, keepdim=True), atol=1e-6, rtol=1e-6)


def test_reference_tie_break_is_ascending_id():
    """CONTRACT: descending prob, ties broken by ASCENDING expert id. With two
    experts sharing an identical logit (=> identical prob), the lower id must
    come first. This is load-bearing for bf16 inputs (bf16 quantizes many experts
    to the same prob; the tie-break is NOT measure-zero there)."""
    dev = _device()
    # expert 2 and expert 5 have the SAME logit (0.7); both should be selected,
    # and expert 2 (lower id) must precede expert 5.
    g = torch.tensor([[0.1, 0.2, 0.7, 0.3, 0.0, 0.7, -1.0]], device=dev, dtype=torch.float32)
    w, ids = topk_softmax_ref(g, 3, renormalize=False)
    assert ids[0].tolist() == [2, 5, 3], ids[0].tolist()  # 2 before 5 (tie -> asc id), then 3
    assert w[0, 0].item() == w[0, 1].item()  # the tied pair has equal weight


def test_reference_rejects_bad_topk():
    g = torch.randn(2, 8, dtype=torch.float32)
    with pytest.raises(ValueError):
        topk_softmax_ref(g, 0)  # topk < 1
    with pytest.raises(ValueError):
        topk_softmax_ref(g, 9)  # topk > E


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Triton device kernel vs the reference (incl. the bf16 tie-break parity)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("M,E,topk", [(8, 16, 4), (64, 256, 8), (37, 8, 2), (1, 256, 8)])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16, torch.float32])
def test_triton_matches_reference(M, E, topk, dt):
    """The fused Triton kernel matches the reference: weights within fp32 softmax
    tolerance AND ids EXACT (the integer top-k selection must agree, including
    the ascending-id tie-break on bf16-induced ties)."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if dt != torch.float32 and _INTERP:
        pytest.skip("interpreter runs fp32 only")
    dev = _device()
    torch.manual_seed(0)
    g = torch.randn(M, E, device=dev, dtype=dt)
    for renorm in (True, False):
        wt, it = topk_softmax(g, topk, renormalize=renorm, backend=Backend.TRITON)
        wr, ir = topk_softmax_ref(g, topk, renorm)
        # ids must match EXACTLY (integer selection + canonical tie-break).
        assert bool((it == ir).all()), f"ids mismatch (renorm={renorm})"
        # weights within the softmax fp32 tolerance (tl.exp vs torch.exp).
        torch.testing.assert_close(
            wt.float(), wr.float(), atol=2e-2, rtol=2e-2,
        )


def test_public_auto_backend_works():
    """``backend="auto"`` routes to Triton (or reference fallback) and returns a
    valid (weights, ids) pair."""
    dev = _device()
    dt = torch.float32 if _INTERP else torch.bfloat16
    g = torch.randn(16, 64, device=dev, dtype=dt)
    w, ids = topk_softmax(g, 8)
    assert w.shape == (16, 8) and ids.shape == (16, 8)
    assert torch.isfinite(w).all()
