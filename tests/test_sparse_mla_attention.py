# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: DeepSeek-V4 sparse-MLA attention compute (issue #32) on gfx942.

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32). The pure-torch
oracle ``sparse_mla_attention_ref`` is the source of truth for the Triton kernel
and the decode wrapper.
"""

from __future__ import annotations

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


from xkernels.ops.attention.interface import (  # noqa: E402
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
    eo, el, em = sparse_mla_attention_ref(
        q, kv.squeeze(1), idx.squeeze(1), sm_scale=0.2, topk_length=lens, attn_sink=sink
    )
    torch.testing.assert_close(out, eo)
    torch.testing.assert_close(lse, el)


def test_get_mla_metadata_is_callable_noarg():
    meta, num_splits = get_mla_metadata()
    assert isinstance(num_splits, int) and num_splits >= 1
    assert isinstance(meta, torch.Tensor)


from xkernels._backends import Backend  # noqa: E402
from xkernels._dispatch import registered_backends  # noqa: E402

_HAS_TRITON = Backend.TRITON in registered_backends("sparse_mla_attention")


@pytest.mark.parametrize(
    "D,d_v,topk,H", [(512, 512, 64, 8), (512, 448, 128, 4), (32, 32, 7, 3)]
)
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
    got = sparse_mla_attention(
        q, kv, idx, sm_scale=0.1, topk_length=lens, attn_sink=sink,
        d_v=d_v, backend=Backend.TRITON,
    )
    exp = sparse_mla_attention(
        q, kv, idx, sm_scale=0.1, topk_length=lens, attn_sink=sink,
        d_v=d_v, backend=Backend.REFERENCE,
    )
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(got[0].float(), exp[0].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(got[1], exp[1], atol=atol, rtol=rtol)


def test_triton_empty_window_row_no_nan():
    """A token with zero selected KV (topk_length=0) must not NaN; with a sink it
    attends to the sink only (out=0), without a sink it is a zero row."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    torch.manual_seed(7)
    dt = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, Kv, topk = 2, 4, 32, 16, 96  # topk > BLOCK_N=64 -> multi-chunk
    q = torch.randn(T, H, D, device=dev, dtype=dt)
    kv = torch.randn(Kv, D, device=dev, dtype=dt)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    lens = torch.tensor([0, topk], device=dev, dtype=torch.int32)  # row 0 empty
    for sink in (None, torch.randn(H, device=dev)):
        got = sparse_mla_attention(
            q, kv, idx, sm_scale=0.1, topk_length=lens, attn_sink=sink,
            backend=Backend.TRITON,
        )
        assert torch.isfinite(got[0]).all(), "NaN/Inf in output"
        torch.testing.assert_close(
            got[0][0].float(), torch.zeros_like(got[0][0]).float(),
            atol=1e-4, rtol=1e-4,
        )


from xkernels.ops.attention.sparse_mla import (  # noqa: E402
    dequant_fp8_ds_mla,
    make_fp8_ds_mla_kv,
)

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


@pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
def test_fp8_ds_mla_roundtrip():
    dev = _dev()
    rows = 7
    value, scale, ref = make_fp8_ds_mla_kv(rows, device=dev, seed=4)
    got = dequant_fp8_ds_mla(value, scale)
    assert got.shape == (rows, 512)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
def test_fp8_ds_mla_small_dims():
    """A small case (one nope group) exercises the same encode/decode inverse."""
    dev = _dev()
    value, scale, ref = make_fp8_ds_mla_kv(1, nope_dim=64, rope_dim=64, device=dev, seed=0)
    got = dequant_fp8_ds_mla(value, scale, nope_dim=64, rope_dim=64)
    torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


def _paged_cache(num_blocks, block_size, seed):
    """Build a paged fp8_ds_mla (value, scale) cache + its full bf16 latent."""
    rows = num_blocks * block_size
    value, scale, ref = make_fp8_ds_mla_kv(rows, device=_dev(), seed=seed)
    vb, sb = value.shape[-1], scale.shape[-1]
    return (
        value.view(num_blocks, block_size, vb),
        scale.view(num_blocks, block_size, sb),
        ref.view(num_blocks, block_size, -1),
    )


@pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
def test_flash_mla_with_kvcache_dual_cache():
    from xkernels.ops.attention.interface import flash_mla_with_kvcache

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
    kv_cat = torch.cat([flat, eflat], dim=0)  # [2*rows, D]
    idx_cat = torch.cat([idx, eidx + rows], dim=1)  # [T, 2*topk]
    len_cat = lens + elens
    eo, el, _ = sparse_mla_attention_ref(
        q.to(torch.float32), kv_cat, idx_cat, sm_scale=0.1,
        topk_length=len_cat, attn_sink=sink, d_v=D,
    )
    atol = 1e-3 if _INTERP else 3e-2
    torch.testing.assert_close(out.float(), eo.float(), atol=atol, rtol=atol)


def test_top_level_exports():
    import xkernels

    for name in (
        "sparse_mla_attention",
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
    ):
        assert hasattr(xkernels, name), name


@pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
def test_flash_mla_with_kvcache_single_cache():
    """Degenerate single-cache decode (no extra compressed cache)."""
    from xkernels.ops.attention.interface import flash_mla_with_kvcache

    dev = _dev()
    torch.manual_seed(8)
    nb, bs, H, D = 3, 8, 4, 512
    backend = Backend.TRITON if _HAS_TRITON else Backend.REFERENCE
    val, sca, ref = _paged_cache(nb, bs, seed=8)
    rows = nb * bs
    T, topk = 2, 6
    dt = torch.float32 if _INTERP else torch.bfloat16
    q = torch.randn(T, H, D, device=dev, dtype=dt)
    blk = torch.arange(nb, device=dev, dtype=torch.int32).view(1, nb).expand(T, nb).contiguous()
    idx = torch.randint(0, rows, (T, topk), device=dev, dtype=torch.int32)
    idx[0, -2:] = -1  # padding sentinels
    out, lse = flash_mla_with_kvcache(
        q=q.unsqueeze(1), k_cache=val, block_table=blk, cache_seqlens=None,
        head_dim_v=D, tile_scheduler_metadata=None, softmax_scale=0.1,
        is_fp8_kvcache=True, indices=idx.unsqueeze(1),
        scale_cache=sca, block_size=bs, backend=backend,
    )
    out = out.squeeze(1)
    eo, _, _ = sparse_mla_attention_ref(
        q.to(torch.float32), ref.reshape(rows, D), idx, sm_scale=0.1, d_v=D
    )
    atol = 1e-3 if _INTERP else 3e-2
    torch.testing.assert_close(out.float(), eo.float(), atol=atol, rtol=atol)
