# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: mxfp4_paged_gather backends vs the paged-gather torch oracle
(issue #27 / DeepSeek-V4 DSA indexer on gfx942).

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32). mxfp4 dequant is
exact for the FP4 codes, so the only error is the bf16 round at the output.
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import mxfp4_paged_gather
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.gather.mxfp4 import dequant_mxfp4, make_mxfp4_kv
from xkernels.ops.gather.reference import mxfp4_paged_gather_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("mxfp4_paged_gather")
_GROUP = 32


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _inputs(num_blocks, block_size, head_dim, num_seqs, topk, dev, seed=0):
    packed, scale, _ = make_mxfp4_kv(
        num_blocks, block_size, head_dim, group_size=_GROUP, device=dev, seed=seed
    )
    g = torch.Generator(device=dev).manual_seed(seed + 1)
    max_blocks = max(1, (num_blocks // num_seqs))
    # Each seq owns a disjoint slice of physical blocks via its block table.
    block_table = torch.empty(num_seqs, max_blocks, device=dev, dtype=torch.int32)
    for s in range(num_seqs):
        perm = torch.randperm(num_blocks, generator=g, device=dev)[:max_blocks]
        block_table[s] = perm.to(torch.int32)
    max_pos = max_blocks * block_size
    sel_pos = torch.randint(
        0, max_pos, (num_seqs, topk), generator=g, device=dev, dtype=torch.int32
    )
    # Sprinkle padding sentinels.
    pad_mask = torch.rand(num_seqs, topk, generator=g, device=dev) < 0.2
    sel_pos = torch.where(pad_mask, torch.full_like(sel_pos, -1), sel_pos)
    return packed, scale, block_table, sel_pos, block_size


@pytest.mark.parametrize(
    "num_blocks,block_size,head_dim,num_seqs,topk",
    [
        (8, 16, 128, 2, 12),  # V4 indexer head_dim=128
        (6, 8, 64, 3, 5),
        (4, 32, 256, 1, 20),
    ],
)
def test_triton_matches_reference(num_blocks, block_size, head_dim, num_seqs, topk):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    out_dtype = torch.float32 if _INTERP else torch.bfloat16
    packed, scale, bt, sel, bs = _inputs(
        num_blocks, block_size, head_dim, num_seqs, topk, dev
    )
    out = mxfp4_paged_gather(
        packed, scale, bt, sel, block_size=bs, group_size=_GROUP,
        out_dtype=out_dtype, backend=Backend.TRITON,
    )
    ref = mxfp4_paged_gather_ref(
        packed, scale, bt, sel, block_size=bs, group_size=_GROUP, out_dtype=out_dtype
    )
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


def test_padding_sentinel_zeros():
    dev = _device()
    out_dtype = torch.float32 if _INTERP else torch.bfloat16
    packed, scale, _ = make_mxfp4_kv(4, 8, 64, group_size=_GROUP, device=dev)
    bt = torch.zeros(1, 1, device=dev, dtype=torch.int32)
    # All slots padded -> output must be all zeros.
    sel = torch.full((1, 6), -1, device=dev, dtype=torch.int32)
    backend = Backend.TRITON if _HAS_TRITON else Backend.REFERENCE
    out = mxfp4_paged_gather(
        packed, scale, bt, sel, block_size=8, group_size=_GROUP,
        out_dtype=out_dtype, backend=backend,
    )
    assert torch.count_nonzero(out) == 0


def test_reference_matches_direct_dequant():
    """A single non-padded selection equals the standalone dequant of that row."""
    dev = _device()
    packed, scale, deq = make_mxfp4_kv(3, 8, 64, group_size=_GROUP, device=dev)
    # block_table maps seq0 logical block 0 -> physical block 2.
    bt = torch.tensor([[2]], device=dev, dtype=torch.int32)
    # pos=5 -> within-block index 5 of physical block 2.
    sel = torch.tensor([[5]], device=dev, dtype=torch.int32)
    out = mxfp4_paged_gather_ref(
        packed, scale, bt, sel, block_size=8, group_size=_GROUP, out_dtype=torch.float32
    )
    expected = dequant_mxfp4(packed[2, 5], scale[2, 5], _GROUP)
    torch.testing.assert_close(out[0, 0], expected, atol=1e-5, rtol=1e-5)


def test_dequant_mxfp4_known_codes():
    """Spot-check the E2M1 + E8M0 decode against hand-computed values."""
    dev = _device()
    # Two bytes -> 4 nibbles: 0x21 -> [1.0, 1.5]; 0x83 -> [-0.0, 2.0]
    # nibble decode: 0x1->0.5? No: code 1 -> 0.5. Let's use explicit codes.
    # nib 2 -> 1.0, nib 3 -> 1.5, nib 8 -> -0.0, nib 4 -> 2.0
    packed = torch.tensor([[0x32, 0x48]], device=dev, dtype=torch.uint8)  # [.., 2]
    # head_dim=4, group_size=4 -> one scale group, byte 127 -> 2^0 = 1.
    scale = torch.tensor([[127]], device=dev, dtype=torch.uint8)
    deq = dequant_mxfp4(packed, scale, group_size=4)  # [1, 4]
    # byte 0x32: low nibble 2 -> 1.0, high nibble 3 -> 1.5
    # byte 0x48: low nibble 8 -> -0.0, high nibble 4 -> 2.0
    expected = torch.tensor([[1.0, 1.5, 0.0, 2.0]], device=dev)
    torch.testing.assert_close(deq.abs(), expected.abs(), atol=1e-6, rtol=1e-6)
