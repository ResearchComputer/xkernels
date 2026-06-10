# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness for moe_align_block_size (issue #4): verify the spec invariants.

The reference backend runs on plain CPU. The Triton perf backend (the vLLM/
SGLang-style 4-stage histogram + padded prefix-sum + scatter) is exercised on
GPU, or on CPU under ``TRITON_INTERPRET=1`` — both against the torch oracle.
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import moe_align_block_size
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe.w4a16 import moe_align_block_size_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("moe_align_block_size")


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _make_topk_ids(M, top_k, num_experts, seed=0, device="cpu"):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, num_experts, (M, top_k), generator=g, dtype=torch.int32, device=device)


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


def test_auto_backend_resolves_to_reference_on_cpu():
    # On a CPU/none-vendor build, "auto" resolves to reference even once the
    # Triton backend is registered (Triton needs a GPU at runtime).
    topk_ids = _make_topk_ids(8, 2, 4)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 4, 4)  # backend="auto"
    assert sorted_ids.dtype == torch.int32 and num_post.numel() == 1


# --- Triton perf backend: exact match against the torch reference ------------
# The 4-stage kernel walks contiguous token chunks in order, so its within-expert
# ordering is the same stable order as the reference's argsort(stable=True) — the
# full (sorted_token_ids, expert_ids, num_tokens_post_padded) triple matches.


@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (32, 4, 8, 8), (7, 3, 5, 4), (64, 2, 16, 32)],
)
def test_triton_matches_reference(M, top_k, num_experts, block_size):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    topk_ids = _make_topk_ids(M, top_k, num_experts, device=dev)
    got = moe_align_block_size(topk_ids, block_size, num_experts, backend=Backend.TRITON)
    ref = moe_align_block_size_ref(topk_ids, block_size, num_experts)
    names = ("sorted_token_ids", "expert_ids", "num_tokens_post_padded")
    for name, g, r in zip(names, got, ref, strict=True):
        assert g.dtype == torch.int32, name
        torch.testing.assert_close(g, r, rtol=0, atol=0, msg=name)


def test_triton_single_expert_all_tokens():
    # Degenerate routing: every token-slot hits expert 0. One contiguous run.
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    topk_ids = torch.zeros((10, 4), dtype=torch.int32, device=dev)
    got = moe_align_block_size(topk_ids, 8, 6, backend=Backend.TRITON)
    ref = moe_align_block_size_ref(topk_ids, 8, 6)
    for g, r in zip(got, ref, strict=True):
        torch.testing.assert_close(g, r, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(8))
def test_triton_matches_reference_randomized(seed):
    # Random shapes/routings exercise the per-expert offset + padding index math
    # (skewed counts, empty experts, partial last blocks) against the oracle.
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    g = torch.Generator(device=dev).manual_seed(seed)
    M = int(torch.randint(1, 40, (1,), generator=g).item())
    top_k = int(torch.randint(1, 9, (1,), generator=g).item())
    num_experts = int(torch.randint(2, 33, (1,), generator=g).item())
    block_size = int(2 ** torch.randint(2, 6, (1,), generator=g).item())
    # Skew the routing so some experts get many slots and some get none.
    logits = torch.randn(M, top_k, num_experts, generator=g, device=dev)
    topk_ids = logits.argmax(dim=-1).to(torch.int32)
    got = moe_align_block_size(topk_ids, block_size, num_experts, backend=Backend.TRITON)
    ref = moe_align_block_size_ref(topk_ids, block_size, num_experts)
    for g_out, r_out in zip(got, ref, strict=True):
        torch.testing.assert_close(g_out, r_out, rtol=0, atol=0)
