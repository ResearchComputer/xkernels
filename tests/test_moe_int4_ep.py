# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Expert-parallel (``ep_size > 1``) correctness for the INT4 W4A16 fused-MoE GEMM
(issue #26: lift the ``ep_size <= 1`` gate on the AMD quantized fused-MoE).

Acceptance: under expert parallelism each rank holds a contiguous subset of the
global experts and computes a **partial** MoE output for the tokens routed to its
local experts; tokens routed to other ranks' experts contribute zero on this rank.
Summing the per-rank partials reproduces the full (non-EP) MoE output exactly — so
the production all-reduce over the partials is numerically equivalent to the dense
single-rank kernel. We simulate ``ep_size`` ranks in one process by slicing the
expert dimension and an ``expert_map`` per rank.

Runs on:

* GPU (NVIDIA or AMD gfx942) with a real Triton install -> bf16, ``atol/rtol=2e-2``.
* CPU via ``TRITON_INTERPRET=1`` (no GPU) -> fp32, ``atol/rtol=3e-3`` (the kernel
  path is identical; ``b_deq`` is always cast to ``a.dtype`` before ``tl.dot``).

Usage::

    pytest tests/test_moe_int4_ep.py
    TRITON_INTERPRET=1 pytest tests/test_moe_int4_ep.py
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe import (
    dequant_w4a16,
    fused_moe_int4_w4a16,
    make_w4a16_weights,
    moe_align_block_size_ep,
)

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("moe_int4_w4a16")


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _pin_single_config():
    from xkernels.ops.moe.triton.moe_int4_kernel import fused_moe_int4_kernel

    node = fused_moe_int4_kernel
    while node is not None and not hasattr(node, "configs"):
        node = getattr(node, "fn", None)
    if node is not None:
        node.configs = node.configs[:1]


def _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed):
    """Full (non-EP) grouped-MoE oracle reduced to ``[M, N]`` (fp32)."""
    W = dequant_w4a16(packed, scale, group_size).to(A.dtype)
    M, topk = topk_ids.shape
    out = torch.zeros(M, W.shape[1], dtype=torch.float32, device=A.device)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j])
            contrib = A[m].float() @ W[e].float().T
            if mul_routed:
                contrib = topk_w[m, j].float() * contrib
            out[m] += contrib
    return out


def _inputs(M, E, N, K, top_k, dev, group_size=32):
    torch.manual_seed(0)
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    dtype = torch.float32 if _INTERP else torch.bfloat16
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return packed, scale, A, topk_ids, topk_w


def _build_expert_map(E, ep_size, rank, dev):
    """Contiguous-block EP partition: rank ``r`` owns experts
    ``[r * E//ep, (r+1) * E//ep)``. Returns ``[E]`` int32 mapping each global
    expert id to its local row index on ``rank`` (``-1`` if not owned)."""
    assert E % ep_size == 0
    per = E // ep_size
    lo, hi = rank * per, (rank + 1) * per
    emap = torch.full((E,), -1, dtype=torch.int32, device=dev)
    emap[lo:hi] = torch.arange(per, dtype=torch.int32, device=dev)
    return emap, lo, hi


def _ep_params():
    if _INTERP:
        return [(4, 8, 64, 128, 2, 2), (2, 8, 128, 256, 4, 4), (3, 6, 96, 64, 2, 3)]
    return [
        (1, 48, 256, 512, 8, 4),  # decode, Kimi-ish E/top_k, ep=4
        (4, 16, 512, 1024, 4, 2),
        (8, 16, 1024, 2048, 4, 4),
    ]


# --------------------------------------------------------------------------- #
# 1. Pure host-side dispatch builder: EP partition reproduces full routing.    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,E,N,K,top_k,ep_size", _ep_params())
def test_align_ep_partitions_routing(M, E, N, K, top_k, ep_size):
    """The union of per-rank EP dispatches covers every routed slot exactly once,
    each remapped to the right local expert row — no slot dropped or double-counted."""
    dev = _device()
    _, _, _, topk_ids, _ = _inputs(M, E, N, K, top_k, dev)
    block_size = 16
    seen = torch.zeros(M * top_k, dtype=torch.int64, device=dev)
    for rank in range(ep_size):
        emap, lo, hi = _build_expert_map(E, ep_size, rank, dev)
        sorted_ids, expert_ids, num_post = moe_align_block_size_ep(
            topk_ids, block_size, E, emap
        )
        npost = int(num_post.item())
        # Every non-pad slot must belong to a local expert, and its block's local
        # expert id must equal the slot's remapped global expert id.
        flat_e = topk_ids.flatten().long()
        for b, e_local in enumerate(expert_ids.tolist()):
            blk = sorted_ids[b * block_size : (b + 1) * block_size]
            for slot in blk.tolist():
                if slot >= M * top_k:  # padding
                    continue
                g = int(flat_e[slot])
                assert lo <= g < hi, f"rank {rank} got non-local expert {g}"
                assert emap[g].item() == e_local
                seen[slot] += 1
        assert npost <= sorted_ids.numel()
    # Each global slot is owned by exactly one rank.
    assert torch.all(seen == 1), f"slot coverage broken: {seen.tolist()}"


# --------------------------------------------------------------------------- #
# 2. End-to-end: sum of per-rank partials == full non-EP MoE output.           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,E,N,K,top_k,ep_size", _ep_params())
@pytest.mark.parametrize("mul_routed", [False, True])
def test_ep_partials_sum_to_full(M, E, N, K, top_k, ep_size, mul_routed):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered (triton not installed)")
    dev = _device()
    group_size = 32
    _pin_single_config()
    packed, scale, A, topk_ids, topk_w = _inputs(M, E, N, K, top_k, dev, group_size)

    acc = torch.zeros(M, N, dtype=torch.float32, device=dev)
    for rank in range(ep_size):
        emap, lo, hi = _build_expert_map(E, ep_size, rank, dev)
        local_packed = packed[lo:hi].contiguous()
        local_scale = scale[lo:hi].contiguous()
        partial = fused_moe_int4_w4a16(
            A, local_packed, local_scale, topk_ids, topk_w,
            group_size=group_size, mul_routed_weight=mul_routed,
            backend=Backend.TRITON, expert_map=emap,
        )
        assert partial.shape == (M, N)
        acc += partial.float()

    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed)
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(acc, ref, atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# 3. ep_size == 1 (full local) is a no-op equivalent to the non-EP path.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mul_routed", [False, True])
def test_ep_identity_map_matches_nonep(mul_routed):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered (triton not installed)")
    dev = _device()
    group_size = 32
    _pin_single_config()
    M, E, N, K, top_k = (2, 8, 128, 256, 4) if _INTERP else (4, 16, 512, 1024, 4)
    packed, scale, A, topk_ids, topk_w = _inputs(M, E, N, K, top_k, dev, group_size)

    emap = torch.arange(E, dtype=torch.int32, device=dev)  # identity: all local
    got_ep = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=mul_routed,
        backend=Backend.TRITON, expert_map=emap,
    )
    got_plain = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=mul_routed,
        backend=Backend.TRITON,
    )
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got_ep.float(), got_plain.float(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
