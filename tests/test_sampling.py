# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: decoding-time sampling ops (issue #69).

Covers the two deterministic-given-inputs sampling ops:

  * ``sampling_from_probs``        -- inverse-CDF multinomial draw
  * ``top_k_sampling_from_probs``  -- mask top-k, renormalize, inverse-CDF draw

The load-bearing design: the RNG is EXTERNAL (``uniform_samples`` is an input
tensor), so each op is a DETERMINISTIC function of its inputs and the device
kernel must land the SAME token as the reference -- bit-exact (rtol=atol=0). The
inverse-CDF runs on a FIXED-POINT int64 representation so the cumulative sum is
order-INDEPENDENT (integer + is associative/exact), making the crossing
bit-exact across scan orders/backends at any vocab size.
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import sampling_from_probs, top_k_sampling_from_probs
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.sampling.sampling import (
    sampling_from_probs_ref,
    top_k_sampling_from_probs_ref,
)
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("sampling_from_probs")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  sampling_from_probs -- the inverse-CDF multinomial core
# ═══════════════════════════════════════════════════════════════════════════════


def test_sampling_is_deterministic_given_uniform():
    """CONTRACT: the same (probs, uniform_samples) always yields the same token.
    This is what makes the op ``verify``-able -- the RNG is external, so the
    kernel is a pure deterministic function of its inputs."""
    dev = _device()
    logits = torch.randn(64, 256, device=dev, dtype=torch.float32) * 3.0
    probs = torch.softmax(logits, dim=1)
    u = torch.rand(64, device=dev) * 0.999
    t1 = sampling_from_probs_ref(probs, u)
    t2 = sampling_from_probs_ref(probs, u)
    assert bool((t1 == t2).all())
    # output is [B] int32, valid token indices
    assert t1.shape == (64,) and t1.dtype == torch.int32
    assert bool(((t1 >= 0) & (t1 < 256)).all())


def test_sampling_inverse_cdf_matches_manual():
    """The reference's token is the leftmost j with cumsum(probs)[j] > u."""
    dev = _device()
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(8, 128, device=dev) * 3.0, dim=1)
    u = torch.rand(8, device=dev) * 0.999
    tok = sampling_from_probs_ref(probs, u)
    cdf = torch.cumsum(probs, dim=1)
    manual = torch.tensor([int((cdf[b] > u[b]).float().argmax().item()) for b in range(8)],
                          device=dev, dtype=torch.int32)
    assert bool((tok == manual).all())


def test_sampling_fallback_when_draw_exceeds_total():
    """If u >= the row total (probs summing to < 1), the LAST index V-1 is
    returned (not argmax-of-empty which would wrongly give 0)."""
    dev = _device()
    probs = torch.tensor([[0.2, 0.2, 0.2]], device=dev)  # sums to 0.6, not 1
    u = torch.tensor([0.9], device=dev)  # exceeds total -> fallback
    tok = sampling_from_probs_ref(probs, u)
    assert tok.item() == 2  # V-1


# ═══════════════════════════════════════════════════════════════════════════════
# §2  top_k_sampling_from_probs -- mask + renorm + inverse-CDF
# ═══════════════════════════════════════════════════════════════════════════════


def test_top_k_sampling_only_draws_kept_tokens():
    """The sampled token must be one of the top-k highest-probability tokens --
    the masked-out tail (prob 0 after renorm) can never be drawn."""
    dev = _device()
    probs = torch.softmax(torch.randn(32, 256, device=dev) * 3.0, dim=1)
    u = torch.rand(32, device=dev) * 0.999
    top_k = 10
    tok = top_k_sampling_from_probs_ref(probs, u, top_k)
    # the top-k indices per row
    order = torch.argsort(probs, dim=1, descending=True, stable=True)[:, :top_k]
    kept = [set(order[b].tolist()) for b in range(32)]
    assert all(int(tok[b].item()) in kept[b] for b in range(32))


def test_top_k_sampling_rejects_bad_top_k():
    dev = _device()
    probs = torch.softmax(torch.randn(2, 8, device=dev), dim=1)
    u = torch.rand(2, device=dev)
    with pytest.raises(ValueError):
        top_k_sampling_from_probs_ref(probs, u, 0)
    with pytest.raises(ValueError):
        top_k_sampling_from_probs_ref(probs, u, 9)


def test_top_k_equals_one_is_argmax():
    """top_k=1 keeps only the single highest-prob token -> the draw always lands
    on the argmax token regardless of u."""
    dev = _device()
    probs = torch.softmax(torch.randn(16, 64, device=dev) * 3.0, dim=1)
    u = torch.rand(16, device=dev) * 0.999
    tok = top_k_sampling_from_probs_ref(probs, u, 1)
    assert bool((tok == probs.argmax(dim=1)).all())


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Triton device kernel vs reference -- bit-exact at every size
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("B,V", [(64, 256), (37, 1024), (256, 4096), (32, 16384)])
@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_sampling_triton_matches_reference(B, V, dt):
    """The device kernel lands the SAME token as the reference -- bit-exact at
    every size (incl. the B=256,V=4096 regime where a naive floating cumsum
    diverges between scan orders). The integer fixed-point cumsum is
    order-independent, so this holds regardless of vocab size."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if dt != torch.float32 and _INTERP:
        pytest.skip("interpreter runs fp32 only")
    dev = _device()
    torch.manual_seed(0)
    logits = torch.randn(B, V, device=dev, dtype=dt) * 3.0
    probs = torch.softmax(logits.float(), dim=1).to(dt)
    u = torch.rand(B, device=dev) * 0.999
    kt = sampling_from_probs(probs, u, backend=Backend.TRITON)
    ref = sampling_from_probs_ref(probs, u)
    assert bool((kt == ref).all()), f"token mismatch at B={B},V={V},{dt}"


@pytest.mark.parametrize("B,V,top_k", [(64, 256, 8), (37, 1024, 16), (256, 4096, 40)])
def test_top_k_sampling_triton_matches_reference(B, V, top_k):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16):
        if dt != torch.float32 and _INTERP:
            continue
        logits = torch.randn(B, V, device=dev, dtype=dt) * 3.0
        probs = torch.softmax(logits.float(), dim=1).to(dt)
        u = torch.rand(B, device=dev) * 0.999
        kt = top_k_sampling_from_probs(probs, u, top_k, backend=Backend.TRITON)
        ref = top_k_sampling_from_probs_ref(probs, u, top_k)
        assert bool((kt == ref).all()), f"token mismatch at B={B},V={V},top_k={top_k},{dt}"


def test_public_auto_backend_works():
    """``backend="auto"`` routes to Triton (or reference fallback)."""
    dev = _device()
    dt = torch.float32 if _INTERP else torch.bfloat16
    probs = torch.softmax(torch.randn(32, 512, device=dev, dtype=dt) * 3.0, dim=1).to(dt)
    u = torch.rand(32, device=dev) * 0.999
    t = sampling_from_probs(probs, u)
    assert t.shape == (32,) and t.dtype == torch.int32
    tk = top_k_sampling_from_probs(probs, u, 20)
    assert tk.shape == (32,) and tk.dtype == torch.int32
