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
    # Triton backend is registered (Triton needs a GPU at runtime). On a GPU box
    # auto resolves to Triton, so this CPU-vendor assertion only applies there.
    if torch.cuda.is_available() and not _INTERP:
        pytest.skip("auto resolves to the Triton backend on a GPU vendor")
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


@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (7, 3, 5, 4)],
)
def test_reference_truncate_false_fixed_shape(M, top_k, num_experts, block_size):
    topk_ids = _make_topk_ids(M, top_k, num_experts)
    s_t, e_t, n_t = moe_align_block_size_ref(topk_ids, block_size, num_experts)  # truncate=True
    s_f, e_f, n_f = moe_align_block_size_ref(topk_ids, block_size, num_experts, truncate=False)
    total = M * top_k
    max_pad = total + (num_experts + 1) * (block_size - 1)
    max_blocks = (max_pad + block_size - 1) // block_size
    used = int(n_f.item()) // block_size
    assert e_f.numel() == max_blocks                       # fixed shape
    torch.testing.assert_close(s_f, s_t, rtol=0, atol=0)   # sorted_ids unchanged
    torch.testing.assert_close(n_f, n_t, rtol=0, atol=0)   # num_post unchanged
    torch.testing.assert_close(e_f[:used], e_t, rtol=0, atol=0)  # used prefix matches
    assert torch.all(e_f[used:] == 0)                      # tail sentinel


@pytest.mark.parametrize(
    "M,top_k,num_experts,block_size",
    [(8, 2, 4, 4), (16, 8, 48, 16), (1, 8, 48, 16), (7, 3, 5, 4), (64, 2, 16, 32)],
)
def test_triton_truncate_false(M, top_k, num_experts, block_size):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    topk_ids = _make_topk_ids(M, top_k, num_experts, device=dev)
    s_t, e_t, n_t = moe_align_block_size(topk_ids, block_size, num_experts, backend=Backend.TRITON)
    s_f, e_f, n_f = moe_align_block_size(
        topk_ids, block_size, num_experts, backend=Backend.TRITON, truncate=False
    )
    total = M * top_k
    max_pad = total + (num_experts + 1) * (block_size - 1)
    max_blocks = (max_pad + block_size - 1) // block_size
    used = int(n_f.item()) // block_size
    assert e_f.numel() == max_blocks                       # fixed shape
    torch.testing.assert_close(s_f, s_t, rtol=0, atol=0)   # sorted_ids unchanged
    torch.testing.assert_close(n_f, n_t, rtol=0, atol=0)   # num_post unchanged
    torch.testing.assert_close(e_f[:used], e_t, rtol=0, atol=0)  # used prefix matches truncate=True
    assert torch.all(e_f[used:] == 0)                      # tail sentinel
    # full triton output equals full reference output in fixed-shape mode
    s_r, e_r, n_r = moe_align_block_size_ref(topk_ids, block_size, num_experts, truncate=False)
    torch.testing.assert_close(e_f, e_r, rtol=0, atol=0)


@pytest.mark.parametrize("seed", range(8))
def test_triton_matches_reference_randomized(seed):
    # Random shapes/routings exercise the per-expert offset + padding index math
    # (skewed counts, empty experts, partial last blocks) against the oracle.
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    # Host-side scalar draws use a CPU generator (the values become python ints);
    # the device tensor uses a matching-device generator (a CUDA generator with a
    # CPU output tensor — or vice-versa — raises a device-mismatch error).
    gc = torch.Generator().manual_seed(seed)
    M = int(torch.randint(1, 40, (1,), generator=gc).item())
    top_k = int(torch.randint(1, 9, (1,), generator=gc).item())
    num_experts = int(torch.randint(2, 33, (1,), generator=gc).item())
    block_size = int(2 ** torch.randint(2, 6, (1,), generator=gc).item())
    # Skew the routing so some experts get many slots and some get none.
    gd = torch.Generator(device=dev).manual_seed(seed)
    logits = torch.randn(M, top_k, num_experts, generator=gd, device=dev)
    topk_ids = logits.argmax(dim=-1).to(torch.int32)
    got = moe_align_block_size(topk_ids, block_size, num_experts, backend=Backend.TRITON)
    ref = moe_align_block_size_ref(topk_ids, block_size, num_experts)
    for g_out, r_out in zip(got, ref, strict=True):
        torch.testing.assert_close(g_out, r_out, rtol=0, atol=0)
