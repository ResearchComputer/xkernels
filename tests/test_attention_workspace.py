# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Reusable output-buffer workspaces for hot decode kernels (issue #52).

Validates: (a) workspace path == allocation path (same bits), (b) buffer-address
stability across calls (the graph-capture enabler), (c) larger-workspace reuse
for smaller buckets without stale-data leakage, (d) shape/device/dtype
mismatch rejection, and (e) an end-to-end CUDA graph capture + replay.
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import (
    paged_attention,
    paged_attention_prefill,
)
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention import (
    PagedAttentionPrefillWorkspace,
    PagedAttentionWorkspace,
    SparseMlaAttentionWorkspace,
)
from xkernels.registry.input_gen import generate_inputs
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("paged_attention")
pytestmark = pytest.mark.skipif(
    not _HAS_TRITON, reason="triton backend not registered"
)


def _decode_inputs(B, dt, dev, seed=0):
    pt = {"dtype": "bf16" if dt == torch.bfloat16 else "fp16" if dt == torch.float16 else "fp32",
          "B": B, "H_q": 32, "H_kv": 8, "D": 128, "block_size": 1, "max_seq_len": 64}
    return generate_inputs("paged_attention@1.0.0", pt, seed=seed, device=dev)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  paged_attention (decode) workspace
# ═══════════════════════════════════════════════════════════════════════════════


def test_decode_workspace_matches_allocation():
    dev = _device()
    dt = torch.bfloat16
    inp = _decode_inputs(8, dt, dev)
    o_alloc = paged_attention(backend="triton", **inp)
    ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=dev, dtype=dt)
    o_ws = paged_attention(backend="triton", workspace=ws, **inp)
    assert torch.equal(o_ws, o_alloc)
    assert torch.equal(o_ws, ws.out[:8])  # wrote into the workspace buffer


def test_decode_workspace_address_stable():
    """Same buffer address across calls (the graph-capture requirement)."""
    dev = _device()
    inp = _decode_inputs(8, torch.bfloat16, dev)
    ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=dev, dtype=torch.bfloat16)
    o1 = paged_attention(backend="triton", workspace=ws, **inp)
    o2 = paged_attention(backend="triton", workspace=ws, **inp)
    assert o1.data_ptr() == o2.data_ptr() == ws.out.data_ptr()


def test_decode_workspace_smaller_m_reuse_no_stale_leak():
    """Reuse a B_max=16 workspace for B=8 -- [:8] is correct with no stale data
    even after a prior B=16 fill (issue validation bullet 3)."""
    dev = _device()
    dt = torch.bfloat16
    inp8 = _decode_inputs(8, dt, dev, seed=0)
    inp16 = _decode_inputs(16, dt, dev, seed=1)
    alloc8 = paged_attention(backend="triton", **inp8)
    ws = PagedAttentionWorkspace.allocate(16, 32, 128, device=dev, dtype=dt)
    paged_attention(backend="triton", workspace=ws, **inp16)  # fill 16
    o8 = paged_attention(backend="triton", workspace=ws, **inp8)  # B=8
    assert torch.equal(o8, alloc8)  # [:8] correct, no leak from the B=16 fill
    assert o8.shape[0] == 8


def test_decode_workspace_rejects_too_small():
    dev = _device()
    inp = _decode_inputs(8, torch.bfloat16, dev)
    ws = PagedAttentionWorkspace.allocate(4, 32, 128, device=dev, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="workspace buffer"):
        paged_attention(backend="triton", workspace=ws, **inp)


def test_decode_workspace_rejects_wrong_dtype():
    dev = _device()
    inp = _decode_inputs(8, torch.bfloat16, dev)
    ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=dev, dtype=torch.float16)
    with pytest.raises(ValueError):
        paged_attention(backend="triton", workspace=ws, **inp)


def test_decode_workspace_ignored_by_reference():
    """The reference backend ignores workspace (returns a fresh tensor)."""
    dev = _device()
    inp = _decode_inputs(8, torch.float32, dev)
    ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=dev, dtype=torch.float32)
    o_ref = paged_attention(backend="reference", workspace=ws, **inp)
    o_ref2 = paged_attention(backend="reference", **inp)
    assert torch.equal(o_ref, o_ref2)
    assert o_ref.data_ptr() != ws.out.data_ptr()  # reference didn't use the buffer


# ═══════════════════════════════════════════════════════════════════════════════
# §2  paged_attention_prefill workspace
# ═══════════════════════════════════════════════════════════════════════════════


def _prefill_inputs(dev, seed=0):
    pt = {"dtype": "bf16", "num_seqs": 4, "max_seq_len_q": 64, "max_seq_len_k": 64,
          "H_q": 32, "H_kv": 8, "D": 128, "block_size": 1, "prefix_frac": 0.0}
    return generate_inputs("paged_attention_prefill@1.0.0", pt, seed=seed, device=dev)


def test_prefill_workspace_matches_allocation():
    dev = _device()
    inp = _prefill_inputs(dev)
    nt = inp["q"].shape[0]
    o_alloc = paged_attention_prefill(backend="triton", **inp)
    ws = PagedAttentionPrefillWorkspace.allocate(nt, 32, 128, device=dev, dtype=torch.bfloat16)
    o_ws = paged_attention_prefill(backend="triton", workspace=ws, **inp)
    assert torch.equal(o_ws, o_alloc)
    assert torch.equal(o_ws, ws.out[:nt])


def test_prefill_workspace_address_stable():
    dev = _device()
    inp = _prefill_inputs(dev)
    nt = inp["q"].shape[0]
    ws = PagedAttentionPrefillWorkspace.allocate(nt, 32, 128, device=dev, dtype=torch.bfloat16)
    o1 = paged_attention_prefill(backend="triton", workspace=ws, **inp)
    o2 = paged_attention_prefill(backend="triton", workspace=ws, **inp)
    assert o1.data_ptr() == o2.data_ptr()


def test_prefill_workspace_larger_reuse():
    dev = _device()
    inp = _prefill_inputs(dev)
    nt = inp["q"].shape[0]
    alloc = paged_attention_prefill(backend="triton", **inp)
    ws = PagedAttentionPrefillWorkspace.allocate(nt * 4, 32, 128, device=dev, dtype=torch.bfloat16)
    o = paged_attention_prefill(backend="triton", workspace=ws, **inp)
    assert torch.equal(o, alloc)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  dataclass logic (allocate / matches / validation)
# ═══════════════════════════════════════════════════════════════════════════════


def test_sparse_mla_workspace_dataclass():
    dev = _device()
    ws = SparseMlaAttentionWorkspace.allocate(8, 4, 64, device=dev, dtype=torch.bfloat16)
    assert ws.out.shape == (8, 4, 64)
    assert ws.lse.shape == (8, 4) and ws.maxl.shape == (8, 4)
    assert ws.lse.dtype == torch.float32 and ws.maxl.dtype == torch.float32
    assert ws.matches(8, 4, 64, device=dev, dtype=torch.bfloat16)
    assert ws.matches(4, 4, 64, device=dev, dtype=torch.bfloat16)  # smaller T
    assert not ws.matches(16, 4, 64, device=dev, dtype=torch.bfloat16)  # larger T
    assert not ws.matches(8, 8, 64, device=dev, dtype=torch.bfloat16)  # wrong H


def test_device_string_matches_concrete():
    """device='cuda' (string) matches a buffer on cuda:0 (the ambiguity the
    _same_device helper resolves)."""
    dev = _device()
    ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=dev, dtype=torch.bfloat16)
    assert ws.matches(8, 32, 128, device="cuda", dtype=torch.bfloat16)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CUDA graph capture end-to-end (the #52 payoff)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(_INTERP, reason="interpreter can't capture a CUDA graph")
def test_decode_graph_capture_and_replay():
    """The workspace keeps buffer addresses stable so a CUDA graph captures once
    and replays on new inputs -- impossible with per-call allocation."""
    dev = _device()
    dt = torch.bfloat16
    inp = _decode_inputs(4, dt, dev, seed=5)
    ws = PagedAttentionWorkspace.allocate(4, 32, 128, device=dev, dtype=dt)
    # warmup
    for _ in range(3):
        paged_attention(backend="triton", workspace=ws, **inp)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    paged_attention(backend="triton", workspace=ws, **inp)
    with torch.cuda.graph(g):
        o_capture = paged_attention(backend="triton", workspace=ws, **inp)
    # mutate inputs in place, replay, compare to a fresh-alloc run on new inputs
    inp["q"].add_(0.05)
    inp["k_cache"].mul_(0.99)
    g.replay()
    fresh = paged_attention(backend="triton", **inp)
    torch.testing.assert_close(o_capture.float(), fresh.float(), atol=0.1, rtol=0.01)
    # the captured output lives in the stable workspace buffer
    assert o_capture.data_ptr() == ws.out.data_ptr()
