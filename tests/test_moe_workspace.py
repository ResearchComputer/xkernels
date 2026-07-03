# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Reusable workspace buffers for the hot MoE decode kernels (issue #52).

Validates ``moe_align_block_size`` (5 scratch buffers), ``fused_moe_int4_w4a16``
(combine / scratch output), and ``fused_moe_mxfp4`` (activation / combine
output): workspace path == allocation path, buffer-address stability across
calls (graph-capture enabler), smaller-bucket reuse without stale-data leak,
and the re-zero discipline for atomic-add combine outputs.

NOTE: the int4 ``fused_combine=True`` path is NOT exercised here -- it relies on
the RESOLVED-config launch path for soundness (see the SOUNDNESS GUARD in
``_moe_int4_w4a16_triton``). With ``config=None`` (no tuned config for the test
M on GB10), ``@triton.autotune`` accumulates atomic-adds across candidate
configs, which breaks the ALLOC path equally. A serving stack reuses a cached
config (config != None), where the workspace IS active and correct; verifying
that needs a populated config cache.
"""

from __future__ import annotations

import pytest
import torch

from xkernels import fused_moe_int4_w4a16, fused_moe_mxfp4, moe_align_block_size
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe import (
    MoeAlignWorkspace,
    MoeInt4Workspace,
    MoeMxfp4Workspace,
    make_mxfp4_moe_weights,
    make_w4a16_weights,
)
from xkernels.utils.testing import gpu_device_or_skip as _device

_HAS_TRITON_ALIGN = Backend.TRITON in registered_backends("moe_align_block_size")
_HAS_TRITON_INT4 = Backend.TRITON in registered_backends("moe_int4_w4a16")
_HAS_TRITON_MXFP4 = Backend.TRITON in registered_backends("moe_mxfp4")
_DEV = pytest.mark.skipif(
    not (_HAS_TRITON_ALIGN and _HAS_TRITON_INT4 and _HAS_TRITON_MXFP4),
    reason="triton MoE backends not registered",
)
_DTYPE = torch.bfloat16


def _int4_inputs(M, E, N, K, top_k, dev, seed=1, gs=32):
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=seed)
    A = (torch.randn(M, K, device=dev) * 0.1).to(_DTYPE)
    tids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, packed, scale, tids, tw


def _mxfp4_inputs(M, E, hidden, ispp, top_k, dev, seed=1, gs=32):
    d = make_mxfp4_moe_weights(E, hidden, ispp, group_size=gs, device=dev, seed=seed)
    A = (torch.randn(M, hidden, device=dev) * 0.1).to(_DTYPE)
    tids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, d["w13"], d["w13_scale"], d["w2"], d["w2_scale"], tids, tw


# ═══════════════════════════════════════════════════════════════════════════════
# §1  moe_align_block_size
# ═══════════════════════════════════════════════════════════════════════════════


@_DEV
def test_align_workspace_matches_allocation():
    dev = _device()
    E, BS = 8, 64
    tids = torch.randint(0, E, (8, 4), device=dev, dtype=torch.int32)
    s_a, e_a, n_a = moe_align_block_size(tids, BS, E, backend="triton", truncate=False)
    ws = MoeAlignWorkspace.allocate(8, 4, E, BS, device=dev)
    s_w, e_w, n_w = moe_align_block_size(
        tids, BS, E, backend="triton", truncate=False, workspace=ws
    )
    assert torch.equal(s_w, s_a)
    assert torch.equal(e_w, e_a)
    assert torch.equal(n_w, n_a)
    assert torch.equal(s_w, ws.sorted_ids)  # wrote into the workspace buffer
    assert torch.equal(e_w, ws.expert_ids)


@_DEV
def test_align_workspace_address_stable():
    dev = _device()
    E, BS = 8, 64
    tids = torch.randint(0, E, (8, 4), device=dev, dtype=torch.int32)
    ws = MoeAlignWorkspace.allocate(8, 4, E, BS, device=dev)
    s1, _, _ = moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    s2, _, _ = moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    assert s1.data_ptr() == s2.data_ptr() == ws.sorted_ids.data_ptr()


@_DEV
def test_align_workspace_counters_reinit():
    """The histogram counters / pad-fill are re-initialized each call, so a
    third call into the same workspace equals the first (no stale accumulate)."""
    dev = _device()
    E, BS = 8, 64
    tids = torch.randint(0, E, (8, 4), device=dev, dtype=torch.int32)
    ws = MoeAlignWorkspace.allocate(8, 4, E, BS, device=dev)
    s1, _, n1 = moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    s3, _, n3 = moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    assert torch.equal(s1, s3)
    assert torch.equal(n1, n3)


@_DEV
def test_align_workspace_smaller_m_reuse():
    """Reuse a max_M=16 workspace for M=4 -- output is the M-shaped slice."""
    dev = _device()
    E, BS = 8, 64
    tids4 = torch.randint(0, E, (4, 4), device=dev, dtype=torch.int32)
    s_a, _, n_a = moe_align_block_size(tids4, BS, E, backend="triton", truncate=False)
    ws_big = MoeAlignWorkspace.allocate(16, 4, E, BS, device=dev)
    s_w, _, n_w = moe_align_block_size(
        tids4, BS, E, backend="triton", truncate=False, workspace=ws_big
    )
    assert torch.equal(s_w, s_a)
    assert torch.equal(n_w, n_a)
    assert s_w.shape[0] == s_a.shape[0]  # M-shaped, not max_M-shaped


@_DEV
def test_align_workspace_ignored_by_reference():
    dev = _device()
    E, BS = 8, 64
    tids = torch.randint(0, E, (8, 4), device=dev, dtype=torch.int32)
    ws = MoeAlignWorkspace.allocate(8, 4, E, BS, device=dev)
    o_ref = moe_align_block_size(tids, BS, E, backend="reference", workspace=ws)
    o_ref2 = moe_align_block_size(tids, BS, E, backend="reference")
    assert torch.equal(o_ref[0], o_ref2[0])


# ═══════════════════════════════════════════════════════════════════════════════
# §2  fused_moe_int4_w4a16  (fused_combine=False, the sound path on GB10)
# ═══════════════════════════════════════════════════════════════════════════════


@_DEV
def test_int4_workspace_scratch_matches_allocation():
    dev = _device()
    A, packed, scale, tids, tw = _int4_inputs(8, 8, 256, 256, 4, dev, seed=2)
    N = packed.shape[1]
    o_alloc = fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="triton"
    )
    ws = MoeInt4Workspace.allocate(8, 4, N, dtype=_DTYPE, device=dev)
    o_ws = fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="triton", workspace=ws
    )
    torch.testing.assert_close(o_ws.float(), o_alloc.float(), atol=1e-3, rtol=1e-3)


@_DEV
def test_int4_workspace_scratch_address_stable_and_reused():
    dev = _device()
    A, packed, scale, tids, tw = _int4_inputs(8, 8, 256, 256, 4, dev, seed=2)
    N = packed.shape[1]
    ws = MoeInt4Workspace.allocate(8, 4, N, dtype=_DTYPE, device=dev)
    fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="triton", workspace=ws
    )
    p1 = ws.scratch.data_ptr()
    fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="triton", workspace=ws
    )
    assert ws.scratch.data_ptr() == p1  # buffer address stable


@_DEV
def test_int4_workspace_smaller_m_reuse():
    dev = _device()
    A4, packed4, scale4, tids4, tw4 = _int4_inputs(4, 8, 256, 256, 4, dev, seed=3)
    N = packed4.shape[1]
    o_alloc = fused_moe_int4_w4a16(
        A4, packed4, scale4, tids4, tw4, fused_combine=False, backend="triton"
    )
    ws_big = MoeInt4Workspace.allocate(16, 4, N, dtype=_DTYPE, device=dev)
    o_ws = fused_moe_int4_w4a16(
        A4, packed4, scale4, tids4, tw4,
        fused_combine=False, backend="triton", workspace=ws_big,
    )
    torch.testing.assert_close(o_ws.float(), o_alloc.float(), atol=1e-3, rtol=1e-3)


@_DEV
def test_int4_workspace_ignored_by_reference():
    dev = _device()
    A, packed, scale, tids, tw = _int4_inputs(4, 8, 96, 64, 2, dev, seed=5)
    N = packed.shape[1]
    ws = MoeInt4Workspace.allocate(4, 2, N, dtype=_DTYPE, device=dev)
    o_ref = fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="reference", workspace=ws
    )
    o_ref2 = fused_moe_int4_w4a16(
        A, packed, scale, tids, tw, fused_combine=False, backend="reference"
    )
    assert torch.equal(o_ref, o_ref2)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  fused_moe_mxfp4
# ═══════════════════════════════════════════════════════════════════════════════


@_DEV
def test_mxfp4_workspace_matches_allocation():
    dev = _device()
    A, w13, w13s, w2, w2s, tids, tw = _mxfp4_inputs(8, 8, 256, 128, 4, dev, seed=2)
    hidden, ispp, E = 256, 128, 8
    o_alloc = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton")
    from xkernels.ops.moe.triton.moe_mxfp4_kernel import get_default_config

    block_m = get_default_config(8)["BLOCK_SIZE_M"]
    ws = MoeMxfp4Workspace.allocate(8, 4, E, block_m, ispp, hidden, dtype=_DTYPE, device=dev)
    o_ws = fused_moe_mxfp4(
        A, w13, w13s, w2, w2s, tids, tw, backend="triton", workspace=ws
    )
    torch.testing.assert_close(o_ws.float(), o_alloc.float(), atol=1e-2, rtol=1e-2)


@_DEV
def test_mxfp4_workspace_combine_rezeroed_on_reuse():
    """The down-stage combine is atomic-add -> the workspace MUST re-zero the
    combine buffer each call (a stale nonzero would corrupt the result)."""
    dev = _device()
    A, w13, w13s, w2, w2s, tids, tw = _mxfp4_inputs(8, 8, 256, 128, 4, dev, seed=2)
    hidden, ispp, E = 256, 128, 8
    from xkernels.ops.moe.triton.moe_mxfp4_kernel import get_default_config

    block_m = get_default_config(8)["BLOCK_SIZE_M"]
    ws = MoeMxfp4Workspace.allocate(8, 4, E, block_m, ispp, hidden, dtype=_DTYPE, device=dev)
    o1 = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton", workspace=ws)
    o2 = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton", workspace=ws)
    torch.testing.assert_close(o1.float(), o2.float(), atol=1e-2, rtol=1e-2)


@_DEV
def test_mxfp4_workspace_ignored_by_reference():
    dev = _device()
    A, w13, w13s, w2, w2s, tids, tw = _mxfp4_inputs(4, 8, 256, 128, 4, dev, seed=6)
    hidden, ispp, E = 256, 128, 8
    from xkernels.ops.moe.triton.moe_mxfp4_kernel import get_default_config

    block_m = get_default_config(4)["BLOCK_SIZE_M"]
    ws = MoeMxfp4Workspace.allocate(4, 4, E, block_m, ispp, hidden, dtype=_DTYPE, device=dev)
    o_ref = fused_moe_mxfp4(
        A, w13, w13s, w2, w2s, tids, tw, backend="reference", workspace=ws
    )
    o_ref2 = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="reference")
    torch.testing.assert_close(o_ref.float(), o_ref2.float(), atol=1e-4, rtol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  CUDA graph capture (moe_align -- the cleanest, fully graph-capturable)
# ═══════════════════════════════════════════════════════════════════════════════


@_DEV
def test_align_graph_capture_and_replay():
    """The workspace keeps the 5 scratch buffer addresses stable, so a CUDA graph
    captures the whole align call once and replays on new inputs (impossible with
    per-call allocation)."""
    dev = _device()
    E, BS = 8, 64
    tids = torch.randint(0, E, (4, 4), device=dev, dtype=torch.int32)
    ws = MoeAlignWorkspace.allocate(4, 4, E, BS, device=dev)
    for _ in range(3):
        moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        s_g, e_g, n_g = moe_align_block_size(
            tids, BS, E, backend="triton", truncate=False, workspace=ws
        )
    # mutate inputs in place, replay, compare to a fresh-alloc run
    tids.add_(1)
    tids.clamp_(max=E - 1)
    g.replay()
    s_ref, e_ref, _ = moe_align_block_size(
        tids, BS, E, backend="triton", truncate=False
    )
    assert torch.equal(s_g, s_ref)
    assert torch.equal(e_g, e_ref)
    assert s_g.data_ptr() == ws.sorted_ids.data_ptr()  # output in the workspace buffer
