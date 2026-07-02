# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton sampling kernels (issue #69): inverse-CDF multinomial draws.

Two kernels, one program per batch row, both DETERMINISTIC given the external
``uniform_samples`` input (the RNG lives outside the kernel -- see
``ops/sampling/sampling.py`` for why that makes ``verify`` bit-exact):

  * ``sampling_kernel``       -- inverse-CDF over the full vocab distribution.
  * ``top_k_sampling_kernel`` -- mask to top-k (iterative argmax, tie-break
    ascending id -- same selection as ``topk_softmax``), renormalize, inverse-CDF.

Bit-exactness contract. Token selection is integer-exact (a mismatch is an
``abs_err >= 1`` and fails any tolerance), so the device cumulative sum MUST land
the ``cdf > u`` crossing at the same index as the reference's ``torch.cumsum`` +
``torch.searchsorted(right=True)``. The reference cumsum is a sequential
left-to-right fp32 prefix over the contiguous row; the kernel reproduces that
order with ``tl.cumsum`` (a prefix scan over the loaded row vector). On
well-separated sweep distributions the crossing is never within 1 fp32-ULP of
``u``, so the scan associativity is irrelevant -- the token matches bit-for-bit.
(See the spec numerics.notes for the measure-zero adversarial caveat.)

Vocabulary-size stance (mirrors ``temperature_softmax``). The whole row is loaded
into registers in one coalesced read (``BLOCK_V = next_pow2(V)``). This is
correct and fast for moderate vocab (V up to a few thousand, e.g. MoE expert
distributions); the production LLM vocab (V ~ 100k+) needs a staged streaming
scan -- the SAME documented limitation as the DSL ``temperature_softmax`` row
kernel, and a tuning follow-up, NOT a contract change.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = [
    "sampling_from_probs_triton",
    "top_k_sampling_from_probs_triton",
    "sampling_kernel",
    "top_k_sampling_kernel",
]


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(p, 1)


@triton.jit
def sampling_kernel(
    probs_ptr,
    uniform_ptr,
    out_ptr,
    stride_pb,
    V,
    BLOCK_V: tl.constexpr,
):
    """Inverse-CDF multinomial draw, one token per row."""
    row = tl.program_id(0)
    u = tl.load(uniform_ptr + row)  # scalar fp32 draw in [0, 1)

    offs = tl.arange(0, BLOCK_V)
    valid = offs < V
    p = tl.load(probs_ptr + row * stride_pb + offs, mask=valid, other=0.0).to(tl.float32)

    # FIXED-POINT inverse-CDF. Scale to int64 (the scale 2**30 is an exact fp32
    # power of two -> identical q in every backend) then INTEGER cumsum. Integer
    # addition is associative + exact, so this prefix is order-INDEPENDENT -- a
    # parallel scan here matches torch.cumsum(int64) bit-for-bit, and the
    # ``cdf > u`` crossing lands at the same token in every backend. (A floating
    # cumsum diverges between scan orders at V~4096 and flips tokens; the integer
    # representation removes that divergence entirely.)
    SCALE: tl.constexpr = 1073741824  # 1 << 30
    q = (p * SCALE).to(tl.int64)
    cdf = tl.cumsum(q, axis=0)  # [BLOCK_V] int64 prefix -- associative-exact
    u_int = (u * SCALE).to(tl.int64)

    # Leftmost j with cdf[j] > u_int. argmax over (cdf > u_int).to(int32) returns
    # the FIRST True (tl.argmax returns the lowest index on ties); if NO entry
    # exceeds u_int (u >= row total, possible under fp rounding), fall back to
    # V-1. Clamp padding into range.
    gt = cdf > u_int
    idx = tl.argmax(gt.to(tl.int32), axis=0)
    has = tl.max(gt.to(tl.int32), axis=0)  # 1 if any crossing, else 0
    token = tl.where(has == 1, idx, V - 1)
    token = tl.minimum(token, V - 1)
    tl.store(out_ptr + row, token.to(tl.int32))


@triton.jit
def top_k_sampling_kernel(
    probs_ptr,
    uniform_ptr,
    out_ptr,
    stride_pb,
    V,
    TOPK: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """Mask to top-k, renormalize, then inverse-CDF draw."""
    row = tl.program_id(0)
    u = tl.load(uniform_ptr + row)  # scalar fp32 draw in [0, 1)

    offs = tl.arange(0, BLOCK_V)
    valid = offs < V
    p = tl.load(probs_ptr + row * stride_pb + offs, mask=valid, other=0.0).to(tl.float32)

    # Iterative argmax top-k selection (descending prob, ties by ascending vocab id
    # -- tl.argmax returns the lowest index on ties, matching the reference's stable
    # descending argsort). Identical pattern to topk_softmax_kernel.
    selected = tl.zeros([BLOCK_V], dtype=tl.int1)
    for _ in tl.static_range(TOPK):
        cand = tl.where(selected | (~valid), -float("inf"), p)
        idx = tl.argmax(cand, axis=0)
        selected = selected | (offs == idx)  # one-hot of the picked index

    # Renormalize the kept top-k to sum to 1; mask everything else to 0.
    kept = tl.where(selected, p, 0.0)
    total = tl.sum(kept, axis=0)
    masked = kept / total  # [BLOCK_V] fp32; kept positions sum to 1

    # FIXED-POINT inverse-CDF over the renormalized (sparse) distribution -- see
    # sampling_kernel: integer cumsum is associative-exact, so the crossing is
    # bit-exact across scan orders/backends.
    SCALE: tl.constexpr = 1073741824  # 1 << 30
    q = (masked * SCALE).to(tl.int64)
    cdf = tl.cumsum(q, axis=0)
    u_int = (u * SCALE).to(tl.int64)
    gt = cdf > u_int
    sidx = tl.argmax(gt.to(tl.int32), axis=0)
    has = tl.max(gt.to(tl.int32), axis=0)
    token = tl.where(has == 1, sidx, V - 1)
    token = tl.minimum(token, V - 1)
    tl.store(out_ptr + row, token.to(tl.int32))


def sampling_from_probs_triton(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
) -> torch.Tensor:
    """Host launcher for the inverse-CDF ``sampling_kernel``.

    Args match the reference (:func:`xkernels.ops.sampling.sampling.sampling_from_probs_ref`);
    returns ``token_ids [B]`` int32.
    """
    probs = probs.contiguous()
    B, V = probs.shape
    device = probs.device
    out = torch.empty(B, dtype=torch.int32, device=device)
    grid = (B,)
    sampling_kernel[grid](
        probs,
        uniform_samples.reshape(B),
        out,
        probs.stride(0),
        V,
        BLOCK_V=_next_pow2(V),
        num_warps=4,
    )
    return out


def top_k_sampling_from_probs_triton(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Host launcher for ``top_k_sampling_kernel``.

    Args match the reference (:func:`xkernels.ops.sampling.sampling.top_k_sampling_from_probs_ref`);
    returns ``token_ids [B]`` int32.
    """
    probs = probs.contiguous()
    B, V = probs.shape
    if not (1 <= int(top_k) <= V):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= V (got top_k={top_k}, V={V})")
    device = probs.device
    out = torch.empty(B, dtype=torch.int32, device=device)
    grid = (B,)
    top_k_sampling_kernel[grid](
        probs,
        uniform_samples.reshape(B),
        out,
        probs.stride(0),
        V,
        TOPK=int(top_k),
        BLOCK_V=_next_pow2(V),
        num_warps=4,
    )
    return out


register("sampling_from_probs", Backend.TRITON)(sampling_from_probs_triton)
register("top_k_sampling_from_probs", Backend.TRITON)(top_k_sampling_from_probs_triton)
