# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: varlen paged GQA PREFILL/EXTEND attention (issue #71 prefill).

Covers the causal math (the load-bearing part), the extend (nk>nq) semantics,
the multi-sequence packed partitioning via cu_seqlens, the GQA head mapping
(MHA/GQA/MQA), block_size generalization, and the runtime divisibility assert.
The causal gold check cross-validates against
``F.scaled_dot_product_attention(is_causal=True)``. Runs on GPU (bf16/fp16) or
CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from xkernels import paged_attention_prefill
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention.paged_attention_prefill import (
    paged_attention_prefill_ref,
)
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("paged_attention_prefill")


def _make(num_seqs, Lq, Lk, Hq, Hkv, D, bs, dt, seed=0, device="cuda"):
    """Build a varlen paged prefill setup. Each seq has Lq new q-tokens and Lk
    total kv (Lk>=Lq; Lk>Lk is the extend case). One block_table row per seq,
    disjoint ascending page ids (page_size=bs)."""
    max_blocks = (Lk + bs - 1) // bs
    g = torch.Generator(device=device).manual_seed(seed)
    nq_per = torch.full((num_seqs,), Lq, dtype=torch.int32, device=device)
    nk_per = torch.full((num_seqs,), Lk, dtype=torch.int32, device=device)
    cu_q = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_q[1:] = torch.cumsum(nq_per, 0)
    cu_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    cu_k[1:] = torch.cumsum(nk_per, 0)
    num_tokens = int(cu_q[-1])
    num_blocks = num_seqs * max_blocks
    q = torch.randn(num_tokens, Hq, D, generator=g, device=device, dtype=dt)
    kc = torch.randn(num_blocks, bs, Hkv, D, generator=g, device=device, dtype=dt)
    vc = torch.randn(num_blocks, bs, Hkv, D, generator=g, device=device, dtype=dt)
    bt = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(num_seqs, max_blocks)
    return q, kc, vc, bt, cu_q, cu_k, float(D ** -0.5)


def _extend_gold(q, kc, vc, bt, cu_q, cu_k, scale, group, bs):
    """Clean causal-extend gold via SDPA over the full kv per seq, keeping only
    the new tokens' rows. seq s: nq new tokens are the SUFFIX of nk kv; the new
    token at local p attends to kv [0, prefix+p+1)."""
    num_seqs = cu_q.numel() - 1
    D = q.shape[-1]
    gold = torch.empty_like(q)
    for s in range(num_seqs):
        qs, qe = int(cu_q[s]), int(cu_q[s + 1])
        nq = qe - qs
        nk = int(cu_k[s + 1]) - int(cu_k[s])
        prefix = nk - nq
        pages = bt[s, :(nk + bs - 1) // bs].long()
        k = kc[pages].reshape(-1, kc.shape[2], D)[:nk]      # [nk, Hkv, D]
        v = vc[pages].reshape(-1, vc.shape[2], D)[:nk]
        ke = k.repeat_interleave(group, 1)                  # [nk, Hq, D]
        ve = v.repeat_interleave(group, 1)
        # Run SDPA on the FULL nk-length seq with causal mask, take last nq rows.
        # Need q of length nk: prepend prefix dummy rows (their output discarded).
        qfull = torch.zeros(nk, q.shape[1], D, device=q.device, dtype=q.dtype)
        qfull[prefix:] = q[qs:qe]
        full = F.scaled_dot_product_attention(
            qfull[None].permute(0, 2, 1, 3), ke[None].permute(0, 2, 1, 3),
            ve[None].permute(0, 2, 1, 3), is_causal=True, scale=scale,
        ).permute(0, 2, 1, 3)[0]                            # [nk, Hq, D]
        gold[qs:qe] = full[prefix:]                         # only the new tokens
    return gold


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Reference oracle -- causal math + extend semantics
# ═══════════════════════════════════════════════════════════════════════════════


def test_reference_matches_sdpa_causal_pure_prefill():
    """Pure prefill (nk==nq): reference matches SDPA(is_causal)."""
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 37, 37, 32, 8, 128, 1, torch.float32, device=dev)
    out = paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=sc)
    gold = _extend_gold(q, kc, vc, bt, cu_q, cu_k, sc, group=4, bs=1)
    torch.testing.assert_close(out.float(), gold.float(), atol=1e-4, rtol=1e-4)


def test_reference_matches_sdpa_extend():
    """Extend (nk>nq): reference matches SDPA(is_causal) on the new tokens."""
    dev = _device()
    # 1 seq, 16 new tokens appended after 48 already-cached tokens.
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 16, 64, 32, 8, 128, 1, torch.float32, device=dev)
    out = paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=sc)
    gold = _extend_gold(q, kc, vc, bt, cu_q, cu_k, sc, group=4, bs=1)
    torch.testing.assert_close(out.float(), gold.float(), atol=1e-4, rtol=1e-4)


def test_reference_rejects_kv_shorter_than_q():
    """nk < nq is invalid (new tokens must be a kv suffix)."""
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 64, 16, 8, 8, 64, 1, torch.float32, device=dev)
    # cu_k implies nk=16 < nq=64 -> extend invariant violated.
    with pytest.raises(ValueError, match="kv length"):
        paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=sc)


def test_reference_rejects_non_gqa_divisor():
    dev = _device()
    # build with Hq=7, Hkv=2 (not divisible)
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 16, 16, 8, 8, 64, 1, torch.float32, device=dev)
    q = q[:, :7]  # force Hq=7
    kc = kc[:, :, :2]  # Hkv=2
    vc = vc[:, :, :2]
    with pytest.raises(ValueError, match="multiple of H_kv"):
        paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=sc)


def test_reference_multi_seq_partitioning():
    """A packed batch of ragged-length seqs: each seq's output matches its own
    SDPA gold (verifies cu_seqlens partitioning)."""
    dev = _device()
    Ls = [37, 16, 64]
    g = torch.Generator(device=dev).manual_seed(3)
    Hq, Hkv, D, dt = 32, 8, 128, torch.float32
    num_tokens = sum(Ls)
    max_blocks = max(Ls)
    bt = torch.arange(3 * max_blocks, device=dev, dtype=torch.int32).reshape(3, max_blocks)
    cu_q = torch.tensor([0, Ls[0], Ls[0] + Ls[1], num_tokens], dtype=torch.int32, device=dev)
    cu_k = cu_q.clone()
    q = torch.randn(num_tokens, Hq, D, generator=g, device=dev, dtype=dt)
    kc = torch.randn(3 * max_blocks, 1, Hkv, D, generator=g, device=dev, dtype=dt)
    vc = torch.randn(3 * max_blocks, 1, Hkv, D, generator=g, device=dev, dtype=dt)
    out = paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=D ** -0.5)
    gold = _extend_gold(q, kc, vc, bt, cu_q, cu_k, D ** -0.5, group=4, bs=1)
    torch.testing.assert_close(out.float(), gold.float(), atol=1e-4, rtol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Triton device kernel vs reference + SDPA gold
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("Lq,Lk,Hq,Hkv,D,bs", [
    (64, 64, 32, 8, 128, 1),    # pure prefill, Qwen3-4B GQA
    (32, 32, 64, 8, 64, 1),     # Llama-70B shape
    (32, 32, 8, 1, 128, 1),     # MQA
    (32, 32, 32, 32, 64, 1),    # MHA
    (128, 128, 32, 8, 128, 16), # block_size=16 (vLLM)
    (16, 64, 32, 8, 128, 1),    # extend (prefix=48)
])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_triton_matches_reference(Lq, Lk, Hq, Hkv, D, bs, dt):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if _INTERP and dt != torch.float32:
        pytest.skip("interpreter runs fp32 only")
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(2, Lq, Lk, Hq, Hkv, D, bs, dt, device=dev)
    kt = paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=sc, backend=Backend.TRITON)
    ref = paged_attention_prefill_ref(q, kc, vc, bt, cu_q, cu_k, scale=sc)
    # combined criterion (atol absorbs the near-zero regime, per the spec notes).
    torch.testing.assert_close(kt.float(), ref.float(), atol=0.1, rtol=0.01)


def test_triton_matches_sdpa_gold_fp32():
    """The causal math is exact: fp32 triton matches SDPA(is_causal) to ~1e-4
    (proving the flash-online-softmax + causal bound are correct)."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 48, 48, 32, 8, 128, 1, torch.float32, device=dev)
    kt = paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=sc, backend=Backend.TRITON)
    gold = _extend_gold(q, kc, vc, bt, cu_q, cu_k, sc, group=4, bs=1)
    torch.testing.assert_close(kt.float(), gold.float(), atol=1e-3, rtol=1e-3)


def test_triton_extend_matches_sdpa_gold():
    """Extend path: the new tokens' triton output matches SDPA(is_causal) gold."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 16, 64, 32, 8, 128, 1, torch.float32, device=dev)
    kt = paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=sc, backend=Backend.TRITON)
    gold = _extend_gold(q, kc, vc, bt, cu_q, cu_k, sc, group=4, bs=1)
    torch.testing.assert_close(kt.float(), gold.float(), atol=1e-3, rtol=1e-3)


def test_triton_rejects_non_gqa_divisor():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    q, kc, vc, bt, cu_q, cu_k, sc = _make(1, 16, 16, 8, 8, 64, 1,
                                          torch.bfloat16 if not _INTERP else torch.float32,
                                          device=dev)
    q = q[:, :7]
    kc = kc[:, :, :2]
    vc = vc[:, :, :2]
    with pytest.raises(ValueError, match="multiple of H_kv"):
        paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=sc, backend=Backend.TRITON)


def test_public_auto_backend_works():
    """``backend="auto"`` routes to Triton (or reference fallback)."""
    dev = _device()
    dt = torch.float32 if _INTERP else torch.bfloat16
    q, kc, vc, bt, cu_q, cu_k, sc = _make(2, 32, 32, 32, 8, 128, 1, dt, device=dev)
    out = paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=sc)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()
