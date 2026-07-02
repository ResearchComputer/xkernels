# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Decoding-time sampling ops (issue #69): the stochastic half of the sampler.

This is the hand-path complement to the DSL ``temperature_softmax`` op: that op
produces a probability distribution, and *these* ops draw tokens from one. The
family this module seeds (canonical_op ``sampling``):

  * ``sampling_from_probs(probs, uniform_samples)``            -> token_ids   [inverse-CDF multinomial]
  * ``top_k_sampling_from_probs(probs, uniform_samples, top_k)`` -> token_ids   [mask top-k, renorm, inverse-CDF]

The remaining two flashinfer family members -- ``top_p_sampling_from_probs`` and
``top_k_top_p_sampling_from_probs`` -- need a device-side *sort* (the nucleus
cutoff is defined on the descending-order probabilities) and are intentionally
NOT in this increment; they are a focused follow-up (see the spec preconditions).

The load-bearing design call: **sampling is stochastic, so the RNG is EXTERNAL.**
``uniform_samples`` is an *input* tensor (one draw in [0, 1) per row), not
generated inside the kernel. This makes each op a DETERMINISTIC function of its
inputs: the same ``(probs, uniform_samples)`` always yields the same token, so
``verify`` is bit-exact (reference and device kernel consume the identical
uniform and must land on the identical token). This is how production samplers
decouple the RNG stream from the CDF traversal -- the host owns the RNG; the
kernel owns the traversal.

The second load-bearing call is the inverse-CDF *bit-exactness*. Token selection
is integer-exact (a mismatch is ``abs_err >= 1`` and fails any tolerance), so the
device kernel's cumulative sum MUST land the ``cdf > u`` crossing at the same
index as the reference. The reference uses ``torch.cumsum`` (a sequential
left-to-right prefix over a contiguous row) + ``torch.searchsorted(right=True)``
(the leftmost ``j`` with ``cdf[j] > u``); the device kernel reproduces the same
serial prefix order. See the numerics notes in each spec for the (measure-zero)
adversarial caveat -- the same discontinuity class as ``topk_softmax``'s near-tie.
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register

__all__ = [
    "sampling_from_probs",
    "sampling_from_probs_ref",
    "top_k_sampling_from_probs",
    "top_k_sampling_from_probs_ref",
]

# Fixed-point scale for the inverse-CDF cumulative sum. A power of two (exact in
# fp32), so ``probs * _FX_SCALE`` lands identically in every backend; the cumsum
# is then an INTEGER prefix (associative + exact -> order-independent -> bit-
# exact across scan orders/backends). ~1e9 levels > fp32's ~7 sig digits, so this
# introduces no quantization error beyond fp32's own precision.
_FX_SCALE = float(1 << 30)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  sampling_from_probs -- inverse-CDF multinomial draw
# ═══════════════════════════════════════════════════════════════════════════════


def sampling_from_probs_ref(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
) -> torch.Tensor:
    """Reference: draw one token per row by inverse-CDF sampling.

    Args:
        probs: ``[B, V]`` probability distribution (fp32; bf16/fp16 upcast). Each
            row should sum to ~1 (a softmax output). It need not be normalized to
            machine precision: the CDF is taken as-is and the fallback handles a
            total < 1 (see below).
        uniform_samples: ``[B]`` uniform draws in ``[0, 1)``, one per row. These
            are an *input*, not generated here -- this is what makes the op
            deterministic and bit-exact ``verify``-able.

    Returns:
        ``token_ids`` ``[B]`` int32 -- for each row, the leftmost vocab index
        ``j`` whose cumulative probability first exceeds ``uniform_samples[b]``
        (i.e. the inverse-CDF sample). If no index exceeds the draw (the draw is
        >= the row total, possible under fp rounding when probs sum to < 1), the
        LAST valid index ``V-1`` is returned.

    Notes:
        * Inverse-CDF on a FIXED-POINT (int64) representation. ``q = probs * 2**30``
          (the scale is an exact fp32 power of two, so the fp32 multiply lands
          identically in every backend); the cumulative sum is then an INTEGER
          prefix. Integer addition is associative and exact, so the prefix is
          order-INDEPENDENT -- a parallel scan (the device kernel) and a
          sequential scan (torch) give the same int64 cumsum bit-for-bit, and the
          ``cdf > u`` crossing lands at the same token in every backend. This is
          why the op is HONESTLY bit-exact (not merely "measure-zero" -- a
          floating cumsum genuinely diverges between scan orders at V~4096 and
          flips tokens, which a naive float implementation hits).
        * The draw is ``uniform_samples`` (external RNG). Re-running with the same
          ``probs`` AND the same ``uniform_samples`` returns the same token; this
          is the determinism contract, not an implementation detail.
    """
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2-D [B, V], got shape {tuple(probs.shape)}")
    B, V = probs.shape
    p = probs.float()
    q = (p * _FX_SCALE).to(torch.int64)            # fixed-point probs (int64)
    cdf = torch.cumsum(q, dim=1)                    # int64 prefix -- associative-exact
    u_int = (uniform_samples.to(torch.float32).reshape(B, 1) * _FX_SCALE).to(torch.int64)
    # leftmost j with cdf[b, j] > u_int[b]; searchsorted returns V if u >= total -> clamp.
    token = torch.searchsorted(cdf, u_int, right=True).reshape(B)
    token = torch.clamp(token, max=V - 1)
    return token.to(torch.int32)


register("sampling_from_probs", Backend.REFERENCE)(sampling_from_probs_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  top_k_sampling_from_probs -- mask to top-k, renormalize, then inverse-CDF
# ═══════════════════════════════════════════════════════════════════════════════


def top_k_sampling_from_probs_ref(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Reference: keep the top-k tokens, renormalize, then inverse-CDF sample.

    Args:
        probs: ``[B, V]`` probability distribution (fp32; bf16/fp16 upcast).
        uniform_samples: ``[B]`` uniform draws in ``[0, 1)`` (external RNG).
        top_k: number of tokens to keep per row (``1 <= top_k <= V``). The top-k
            tokens are selected by DESCENDING probability, ties broken by
            ASCENDING vocab id (a stable descending argsort -- the same canonical
            order as ``topk_softmax``; load-bearing for bf16-induced ties).

    Returns:
        ``token_ids`` ``[B]`` int32.

    Notes:
        * The masked-out tokens get probability 0; the kept top-k are
          renormalized to sum to 1; then ``sampling_from_probs`` is applied.
        * Because the masked distribution is sparse at the original vocab
          indices, the CDF walks original-index order (flat over the zeroed-out
          tail) -- the draw can never land on a masked token (it has 0 mass).
        * The top-k *selection* is integer-exact (tie-break ascending id); the
          inverse-CDF draw is bit-exact given ``uniform_samples`` (see
          :func:`sampling_from_probs_ref`). The whole op is deterministic.
    """
    if probs.dim() != 2:
        raise ValueError(f"probs must be 2-D [B, V], got shape {tuple(probs.shape)}")
    B, V = probs.shape
    if not (1 <= int(top_k) <= V):
        raise ValueError(f"top_k must satisfy 1 <= top_k <= V (got top_k={top_k}, V={V})")
    p = probs.float()
    # Top-k selection: descending prob, ties by ascending vocab id (stable argsort).
    order = torch.argsort(p, dim=1, descending=True, stable=True)  # [B, V]
    keep = order[:, : int(top_k)]  # [B, top_k] int64 -- the kept vocab indices
    # Build a masked, renormalized distribution at the ORIGINAL vocab indices.
    masked = torch.zeros_like(p)
    masked.scatter_(1, keep, p.gather(1, keep))  # kept probs at original positions, 0 elsewhere
    total = masked.sum(dim=1, keepdim=True).clamp_min(1e-30)
    masked = masked / total  # renormalize the kept set to sum to 1
    return sampling_from_probs_ref(masked, uniform_samples)


register("top_k_sampling_from_probs", Backend.REFERENCE)(top_k_sampling_from_probs_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  public dispatch
# ═══════════════════════════════════════════════════════════════════════════════


def sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Inverse-CDF multinomial draw. See :func:`sampling_from_probs_ref`.

    ``backend="auto"`` picks the fastest registered backend (Triton device kernel
    when available, else the pure-torch reference).
    """
    return dispatch(
        "sampling_from_probs", probs, uniform_samples, backend=backend
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_k: int,
    *,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Top-k filtered multinomial draw. See :func:`top_k_sampling_from_probs_ref`."""
    return dispatch(
        "top_k_sampling_from_probs", probs, uniform_samples, top_k, backend=backend
    )
