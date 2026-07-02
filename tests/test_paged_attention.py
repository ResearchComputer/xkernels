# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: batched paged GQA DECODE attention (issue #71).

Covers the reference oracle, the fused Triton device kernel, the GQA head
mapping (MHA/GQA/MQA), the page indirection, and the runtime divisibility
assert. Runs on GPU (bf16/fp16) or CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import paged_attention
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention.paged_attention import paged_attention_decode_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("paged_attention")


def _make_paged(B, Hq, Hkv, D, bs, msl, dt, seed=0, device="cuda"):
    """Build a self-consistent paged KV setup (q, k_cache, v_cache, block_table,
    seq_lens, scale). Each request gets a disjoint page range; seq_lens ragged."""
    maxb = (msl + bs - 1) // bs
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(B, Hq, D, generator=g, device=device, dtype=dt)
    kc = torch.randn(B * maxb, bs, Hkv, D, generator=g, device=device, dtype=dt)
    vc = torch.randn(B * maxb, bs, Hkv, D, generator=g, device=device, dtype=dt)
    bt = torch.arange(B * maxb, device=device, dtype=torch.int32).reshape(B, maxb)
    sl = torch.randint(1, msl + 1, (B,), generator=g, device=device, dtype=torch.int32)
    return q, kc, vc, bt, sl, float(D ** -0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Reference oracle -- semantics + the GQA head mapping
# ═══════════════════════════════════════════════════════════════════════════════


def test_reference_matches_manual_sdpa():
    """The reference matches a manual per-request SDPA with GQA expansion."""
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(4, 8, 2, 64, 1, 37, torch.float32, device=dev)
    out = paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)
    group = 8 // 2
    for b in range(4):
        slb = int(sl[b])
        nb = slb  # block_size=1
        kv = kc[bt[b, :nb].long()].reshape(-1, 2, 64)[:slb].repeat_interleave(group, dim=1)
        vv = vc[bt[b, :nb].long()].reshape(-1, 2, 64)[:slb].repeat_interleave(group, dim=1)
        scores = scale * torch.einsum("hd,shd->hs", q[b], kv)
        p = torch.softmax(scores, dim=-1)
        manual = torch.einsum("hs,shd->hd", p, vv)
        torch.testing.assert_close(out[b], manual, atol=1e-5, rtol=1e-5)


def test_reference_mha_gqa_mqa_all_run():
    """MHA (Hkv==Hq), GQA (Hkv<Hq), and MQA (Hkv==1) all run and produce the
    right output shape."""
    dev = _device()
    for Hq, Hkv in [(8, 8), (32, 8), (8, 1)]:
        q, kc, vc, bt, sl, scale = _make_paged(4, Hq, Hkv, 64, 1, 16, torch.float32, device=dev)
        out = paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)
        assert out.shape == (4, Hq, 64), f"shape wrong for MHA/GQA/MQA {Hq}/{Hkv}"


def test_reference_rejects_non_gqa_divisor():
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(2, 7, 2, 64, 1, 16, torch.float32, device=dev)
    # H_q=7 not divisible by H_kv=2 -> not a valid GQA config
    with pytest.raises(ValueError):
        paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)


def test_reference_block_size_generalization():
    """block_size > 1 (vLLM-style 16) maps positions to pages correctly."""
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(4, 8, 8, 64, 16, 100, torch.float32, device=dev)
    out = paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)
    assert out.shape == (4, 8, 64)
    assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Triton device kernel vs reference
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("B,Hq,Hkv,D,bs,msl", [
    (4, 32, 8, 128, 1, 37),    # Qwen3-4B GQA, page_size=1
    (8, 32, 8, 128, 16, 256),  # vLLM-style block_size=16
    (1, 32, 8, 128, 1, 16),    # single request
    (4, 64, 8, 64, 1, 128),    # Llama-70B GQA, D=64
    (4, 8, 1, 128, 1, 48),     # MQA
])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_triton_matches_reference(B, Hq, Hkv, D, bs, msl, dt):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if dt != torch.float32 and _INTERP:
        pytest.skip("interpreter runs fp32 only")
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(B, Hq, Hkv, D, bs, msl, dt, device=dev)
    kt = paged_attention(q, kc, vc, bt, sl, scale=scale, backend=Backend.TRITON)
    ref = paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)
    # attention agreement: combined criterion (atol absorbs the near-zero regime;
    # see spec numerics.notes for why rel-only is ill-conditioned for attention).
    torch.testing.assert_close(kt.float(), ref.float(), atol=0.1, rtol=0.01)


def test_triton_fp32_machine_precision():
    """With fp32 inputs, the kernel matches the reference to ~machine precision
    in ABSOLUTE terms (proving the flash-online-softmax is essentially exact --
    the higher relative error on bf16 is quantization + near-zero denominators)."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(8, 32, 8, 128, 1, 256, torch.float32, device=dev)
    kt = paged_attention(q, kc, vc, bt, sl, scale=scale, backend=Backend.TRITON)
    ref = paged_attention_decode_ref(q, kc, vc, bt, sl, scale=scale)
    assert (kt - ref).abs().max().item() < 1e-4, "fp32 abs error not near machine precision"


def test_triton_rejects_non_gqa_divisor():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    q, kc, vc, bt, sl, scale = _make_paged(2, 7, 2, 64, 1, 16,
                                           torch.bfloat16 if not _INTERP else torch.float32,
                                           device=dev)
    with pytest.raises(ValueError):
        paged_attention(q, kc, vc, bt, sl, scale=scale, backend=Backend.TRITON)


def test_public_auto_backend_works():
    """``backend="auto"`` routes to Triton (or reference fallback)."""
    dev = _device()
    dt = torch.float32 if _INTERP else torch.bfloat16
    q, kc, vc, bt, sl, scale = _make_paged(8, 32, 8, 128, 1, 64, dt, device=dev)
    out = paged_attention(q, kc, vc, bt, sl, scale=scale)
    assert out.shape == (8, 32, 128)
    assert torch.isfinite(out).all()
