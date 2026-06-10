# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness for moe_align_block_size (issue #4): verify the spec invariants.

Reference backend only for now (a Triton perf kernel is the tracked follow-up),
so this runs on plain CPU — no GPU / interpreter needed.
"""

from __future__ import annotations

import pytest
import torch

from xkernels import moe_align_block_size
from xkernels._backends import Backend


def _make_topk_ids(M, top_k, num_experts, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, num_experts, (M, top_k), generator=g, dtype=torch.int32)


@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (32, 4, 8, 8)],
)
def test_align_invariants(M, top_k, num_experts, block_size):
    topk_ids = _make_topk_ids(M, top_k, num_experts)
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, block_size, num_experts, backend=Backend.REFERENCE
    )

    total = M * top_k
    pad_id = total
    n = int(num_post.item())

    # num_post is a multiple of block_size and within the bound.
    assert n % block_size == 0
    max_pad = total + (num_experts + 1) * (block_size - 1)
    assert sorted_ids.numel() == max_pad
    assert n <= max_pad

    # One expert id per block of `block_size`.
    assert expert_ids.numel() == n // block_size

    # Every real token-slot appears exactly once in the padded region.
    used = sorted_ids[:n]
    real = used[used != pad_id]
    assert real.numel() == total
    assert torch.equal(torch.sort(real).values, torch.arange(total, dtype=torch.int32))

    # Each block holds only tokens whose expert == that block's expert_id; pads ok.
    flat_e = topk_ids.flatten()
    for b in range(n // block_size):
        e = int(expert_ids[b])
        block = sorted_ids[b * block_size : (b + 1) * block_size]
        toks = block[block != pad_id]
        if toks.numel():
            assert torch.all(flat_e[toks.long()] == e)


def test_auto_backend_resolves_to_reference():
    # No Triton backend registered for this op yet -> auto picks reference.
    topk_ids = _make_topk_ids(8, 2, 4)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 4, 4)  # backend="auto"
    assert sorted_ids.dtype == torch.int32 and num_post.numel() == 1
