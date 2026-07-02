# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fused Triton ``topk_softmax`` kernel (issue #70).

One program per token row: load all ``E`` expert logits into registers, fp32
online softmax, then an iterative argmax top-k selection that emits the selected
weights/ids in **descending probability order** (matching the reference's
``torch.topk(sorted=True)`` so element-wise ``verify`` parity holds), with an
optional fp32 renormalization of the selected weights.

This is the ``sgl_kernel.topk_softmax`` op fused into a single launch — the
unfused baseline (the reference card) is two launches (``torch.softmax`` +
``torch.topk``) plus a renorm pointwise. The MoE router of every MoE model calls
this once per forward; on the ROCm path (the issue's motivation) it replaces the
torch fallback.

Selection algorithm (small topk, typically <=8):
  maintain a ``selected`` boolean mask over ``BLOCK_E``; each of ``TOPK`` iters
  takes ``argmax`` of ``where(selected | ~valid, -inf, probs)``, records the
  weight + expert id, then ORs that id's one-hot into ``selected``. The
  one-hot is ``tl.arange(0, BLOCK_E) == idx`` (a runtime-scalar index broadcast
  against the constexpr arange). ``TOPK`` is a constexpr so the loop unrolls;
  ``BLOCK_E = next_pow2(E)`` so the load is a single coalesced vectorized read.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["topk_softmax_triton", "topk_softmax_kernel"]


@triton.jit
def topk_softmax_kernel(
    gating_ptr,
    weights_ptr,
    ids_ptr,
    stride_gm,
    stride_wm,
    E,
    RENORMALIZE,  # runtime int (0/1); uniform-scalar control flow
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
):
    # One program per token row. Load the whole expert axis into registers (E is
    # the expert count, e.g. 256 for DeepSeek-V3; BLOCK_E = next_pow2(E)).
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    valid = offs < E
    g = tl.load(gating_ptr + row * stride_gm + offs, mask=valid, other=0.0).to(tl.float32)

    # fp32 online softmax over the expert axis (subtract row max for stability).
    gmax = tl.max(g, axis=0)
    e = tl.exp(g - gmax)
    e = tl.where(valid, e, 0.0)  # padding -> 0 prob (never selected over reals)
    ssum = tl.sum(e, axis=0)
    probs = e / ssum  # [BLOCK_E] fp32; valid positions sum to 1 over [0,E)

    # Iterative argmax top-k selection (descending — matches torch.topk sorted=True).
    # ``selected`` masks out already-picked experts; padding is excluded via valid.
    selected = tl.zeros([BLOCK_E], dtype=tl.int1)
    wsum = tl.zeros([], dtype=tl.float32)  # accumulator for renorm
    base_w = row * TOPK
    for j in tl.static_range(TOPK):
        cand = tl.where(selected | (~valid), -float("inf"), probs)
        idx = tl.argmax(cand, axis=0)              # scalar expert index (int)
        val = tl.max(cand, axis=0)                 # scalar probability (fp32)
        tl.store(weights_ptr + base_w + j, val)
        tl.store(ids_ptr + base_w + j, idx.to(tl.int32))
        wsum += val
        selected = selected | (offs == idx)        # one-hot of the picked index

    # Optional renormalize: divide each selected weight by their sum (fp32).
    if RENORMALIZE != 0:
        inv = 1.0 / wsum
        for j in tl.static_range(TOPK):
            p = weights_ptr + base_w + j
            tl.store(p, tl.load(p) * inv)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(p, 1)


def topk_softmax_triton(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host launcher for the fused Triton topk_softmax kernel.

    Args match the reference (:func:`xkernels.ops.moe.topk_softmax.topk_softmax_ref`);
    returns ``(topk_weights [M,topk] fp32, topk_ids [M,topk] int32)``.
    """
    gating_output = gating_output.contiguous()
    M, E = gating_output.shape
    if not (1 <= int(topk) <= E):
        raise ValueError(
            f"topk must satisfy 1 <= topk <= E (got topk={topk}, E={E})"
        )
    device = gating_output.device
    weights = torch.empty((M, int(topk)), dtype=torch.float32, device=device)
    ids = torch.empty((M, int(topk)), dtype=torch.int32, device=device)
    block_e = _next_pow2(E)
    grid = (M,)
    topk_softmax_kernel[grid](
        gating_output,
        weights,
        ids,
        gating_output.stride(0),
        weights.stride(0),
        E,
        1 if renormalize else 0,
        BLOCK_E=block_e,
        TOPK=int(topk),
        num_warps=4,
    )
    return weights, ids


register("topk_softmax", Backend.TRITON)(topk_softmax_triton)
