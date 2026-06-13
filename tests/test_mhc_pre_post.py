# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DeepSeek-V4 MHC ``mhc_pre`` / ``mhc_post`` full-fusion parity (issue #44).

The TileLang ``mhc_pre`` fusion mislowers the ``layer_input`` (pre-weighted
residual combine) branch on gfx942 (~97% wrong -> incoherent generation). These
tests pin a correct, portable gfx942 implementation against an independent torch
oracle, with the ``layer_input`` combine front and centre so the defect cannot
regress silently.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Independent torch oracle (mirrors the TileLang fusion math, not the impl)
# ---------------------------------------------------------------------------
def _sinkhorn_ref(mixes, iters, eps):
    mixes = torch.softmax(mixes, dim=-1) + eps
    mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        mixes = mixes / (mixes.sum(dim=-1, keepdim=True) + eps)
        mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    return mixes


def _mhc_pre_oracle(residual, fn, hc_scale, hc_base, rms_eps, hc_eps, iters):
    num_tokens, hc_mult, _ = residual.shape
    x = residual.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_base, post_base, comb_base = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * hc_scale[1].float() + post_base) * 2.0).unsqueeze(-1)
    comb = _sinkhorn_ref(
        comb_raw.reshape(num_tokens, hc_mult, hc_mult) * hc_scale[2].float()
        + comb_base.reshape(1, hc_mult, hc_mult),
        iters,
        hc_eps,
    )
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(residual.dtype), post, comb


def _mhc_post_oracle(hidden_states, residual, post, comb):
    if post.dim() == 2:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = post.float() * hidden_states.float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)


def _inputs(num_tokens, hc_mult, hidden, dev, dt):
    torch.manual_seed(0)
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    residual = torch.randn(num_tokens, hc_mult, hidden, device=dev, dtype=dt)
    # small fn so the prenorm'd mixes land in sigmoid's responsive range
    fn = (torch.randn(hc_mult3, hc_mult * hidden, device=dev) * 0.02).float()
    hc_scale = torch.tensor([0.7, 1.1, 0.5], device=dev, dtype=torch.float32)
    hc_base = torch.randn(hc_mult3, device=dev, dtype=torch.float32)
    return residual, fn, hc_scale, hc_base


# ---------------------------------------------------------------------------
# reference backend (the numerical oracle is also the CPU default backend)
# ---------------------------------------------------------------------------
from xkernels.ops.mhc import mhc_post, mhc_pre  # noqa: E402

RMS_EPS = HC_EPS = 1e-6
ITERS = 20


def test_reference_pre_matches_oracle():
    dev = _dev()
    hc_mult, hidden = 4, 64
    dt = torch.float32 if _INTERP else torch.bfloat16
    residual, fn, hc_scale, hc_base = _inputs(8, hc_mult, hidden, dev, dt)
    li_o, post_o, comb_o = _mhc_pre_oracle(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS
    )
    li, post, comb = mhc_pre(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS, backend="reference"
    )
    torch.testing.assert_close(li.float(), li_o.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(post.float(), post_o.float(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(comb.float(), comb_o.float(), atol=1e-4, rtol=1e-4)


def test_reference_post_matches_oracle():
    dev = _dev()
    hc_mult, hidden = 4, 64
    dt = torch.float32 if _INTERP else torch.bfloat16
    residual, fn, hc_scale, hc_base = _inputs(8, hc_mult, hidden, dev, dt)
    _, post_o, comb_o = _mhc_pre_oracle(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS
    )
    hidden_states = torch.randn(8, hidden, device=dev, dtype=dt)
    out_o = _mhc_post_oracle(hidden_states, residual, post_o, comb_o)
    out = mhc_post(hidden_states, residual, post_o, comb_o, backend="reference")
    torch.testing.assert_close(out.float(), out_o.float(), atol=1e-4, rtol=1e-4)


def test_reference_empty_tokens():
    dev = _dev()
    hc_mult, hidden = 4, 32
    residual, fn, hc_scale, hc_base = _inputs(0, hc_mult, hidden, dev, torch.float32)
    li, post, comb = mhc_pre(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS, backend="reference"
    )
    assert li.shape == (0, hidden)
    assert post.shape == (0, hc_mult, 1)
    assert comb.shape == (0, hc_mult, hc_mult)
    hidden_states = torch.zeros(0, hidden, device=dev, dtype=torch.float32)
    out = mhc_post(hidden_states, residual, post, comb, backend="reference")
    assert out.shape == (0, hc_mult, hidden)


# ---------------------------------------------------------------------------
# Triton backend (gfx942 path; the regression target)
# ---------------------------------------------------------------------------
from xkernels._backends import Backend  # noqa: E402
from xkernels._dispatch import registered_backends  # noqa: E402

_HAS_TRITON_PRE = Backend.TRITON in registered_backends("mhc_pre")
_HAS_TRITON_POST = Backend.TRITON in registered_backends("mhc_post")


@pytest.mark.parametrize("hc_mult,hidden", [(4, 64), (2, 48), (4, 128)])
@pytest.mark.parametrize("num_tokens", [1, 8, 33])
def test_triton_pre_matches_oracle(hc_mult, hidden, num_tokens):
    if not _HAS_TRITON_PRE:
        pytest.skip("triton mhc_pre backend not registered")
    dev = _dev()
    dt = torch.float32 if _INTERP else torch.bfloat16
    residual, fn, hc_scale, hc_base = _inputs(num_tokens, hc_mult, hidden, dev, dt)
    li_o, post_o, comb_o = _mhc_pre_oracle(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS
    )
    li, post, comb = mhc_pre(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS, backend=Backend.TRITON
    )
    atol = rtol = 1e-3 if _INTERP else 2e-2
    # layer_input is the defect branch -- assert it explicitly and tightly.
    assert torch.isfinite(li).all()
    torch.testing.assert_close(li.float(), li_o.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(post.float(), post_o.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(comb.float(), comb_o.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("hc_mult,hidden", [(4, 64), (2, 48), (4, 128)])
@pytest.mark.parametrize("num_tokens", [1, 8, 33])
def test_triton_post_matches_oracle(hc_mult, hidden, num_tokens):
    if not _HAS_TRITON_POST:
        pytest.skip("triton mhc_post backend not registered")
    dev = _dev()
    dt = torch.float32 if _INTERP else torch.bfloat16
    residual, fn, hc_scale, hc_base = _inputs(num_tokens, hc_mult, hidden, dev, dt)
    _, post_o, comb_o = _mhc_pre_oracle(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS
    )
    hidden_states = torch.randn(num_tokens, hidden, device=dev, dtype=dt)
    out_o = _mhc_post_oracle(hidden_states, residual, post_o, comb_o)
    out = mhc_post(hidden_states, residual, post_o, comb_o, backend=Backend.TRITON)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), out_o.float(), atol=atol, rtol=rtol)


def test_triton_v4_flash_shape():
    if not _HAS_TRITON_PRE or not _HAS_TRITON_POST:
        pytest.skip("triton mhc backends not registered")
    if _INTERP:
        pytest.skip("V4 hidden=4096 too slow under the CPU interpreter")
    dev = _dev()
    hc_mult, hidden = 4, 4096
    residual, fn, hc_scale, hc_base = _inputs(8, hc_mult, hidden, dev, torch.bfloat16)
    li_o, post_o, comb_o = _mhc_pre_oracle(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS
    )
    li, post, comb = mhc_pre(
        residual, fn, hc_scale, hc_base, RMS_EPS, HC_EPS, ITERS, backend=Backend.TRITON
    )
    torch.testing.assert_close(li.float(), li_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(post.float(), post_o.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(comb.float(), comb_o.float(), atol=2e-2, rtol=2e-2)
    hidden_states = torch.randn(8, hidden, device=dev, dtype=torch.bfloat16)
    out_o = _mhc_post_oracle(hidden_states, residual, post_o, comb_o)
    out = mhc_post(hidden_states, residual, post_o, comb_o, backend=Backend.TRITON)
    torch.testing.assert_close(out.float(), out_o.float(), atol=2e-2, rtol=2e-2)


def test_top_level_exports():
    import xkernels

    for name in ("mhc_pre", "mhc_post"):
        assert hasattr(xkernels, name), name
