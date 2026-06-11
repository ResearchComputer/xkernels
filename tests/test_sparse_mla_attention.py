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
