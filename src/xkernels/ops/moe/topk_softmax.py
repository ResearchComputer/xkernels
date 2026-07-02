# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fused MoE gating: softmax + top-k + optional renormalization (issue #70).

The MoE router's first step: take per-token expert logits, softmax over the
expert axis (fp32), select the top-k experts, and optionally renormalize the
selected weights to sum to 1. This is the ``sgl_kernel.topk_softmax`` op that
mini-sglang's ROCm path currently lacks a native kernel for (it ships only CUDA
x86_64 wheels); the ROCm path falls back to a torch ``softmax`` + ``topk``.

This module ships the backend-neutral reference (the correctness oracle every
backend card is checked against) + the public dispatch function. The math is
written for **clarity not speed** (pure torch, fp32 softmax); a backend-shaped
reference is an anti-goal (§10). The contract is the natural compute-and-return
form — ``(topk_weights, topk_ids)`` — NOT sgl-kernel's in-place form; the
consumer adapts (allocate buffers, call, copy out). See the Op Spec at
``registry/ops/topk_softmax.spec.json`` for the authoritative contract.

The fused *device* kernel (Triton) is a separate card; this reference is what
makes ``verify`` runnable with no GPU (the gateway skill's CPU-satisfiable gate).
"""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch, register

__all__ = ["topk_softmax", "topk_softmax_ref"]


def topk_softmax_ref(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: softmax over experts -> top-k -> optional renormalize.

    Args:
        gating_output: ``[M, E]`` per-token expert logits (bf16, fp16, or fp32).
            ``E`` is the expert count (e.g. 256 for DeepSeek-V3).
        topk: number of experts to select per token (``1 <= topk <= E``).
        renormalize: if True, divide the selected top-k weights by their sum so
            they re-normalize to 1.0 (the DeepSeek-V3 / SGLang convention: the
            selected experts' probabilities are re-normalized after truncating
            the tail). If False, the weights are the raw softmax probabilities of
            the selected experts (summing to < 1).

    Returns:
        ``(topk_weights, topk_ids)`` where ``topk_weights`` is ``[M, topk]`` fp32
        (descending probability within each row) and ``topk_ids`` is
        ``[M, topk]`` int32 (the selected expert indices, paired positionally
        with the weights).

    Raises:
        ValueError: if ``topk`` is not in ``[1, E]``.

    Notes:
        * Softmax is computed in **fp32** regardless of the input dtype (the
          logits are upcast); the top-k *selection* is therefore exact-integer.
        * The top-k is selected in **descending probability, ties broken by
          ascending expert id** (a stable descending argsort). This canonical
          order is a CONTRACT requirement: bf16/fp16 logits quantize many experts
          to the same fp32 probability (exact ties are NOT measure-zero), so
          every backend must agree element-wise. The Triton device kernel gets
          this order for free (``tl.argmax`` returns the lowest index on ties).
          For distinct probabilities this is identical to ``torch.topk(sorted=True)``.
        * This is the ``sgl_kernel.topk_softmax`` semantics: softmax over ALL
          experts first, THEN top-k, THEN optional renorm — NOT softmax over the
          selected experts only.
    """
    M, E = gating_output.shape
    if not (1 <= int(topk) <= E):
        raise ValueError(
            f"topk must satisfy 1 <= topk <= E (got topk={topk}, E={E})"
        )
    # fp32 softmax over the expert axis (numerically stable: subtract row max).
    probs = torch.softmax(gating_output.float(), dim=1)  # [M, E] fp32
    # Canonical selection order: DESCENDING probability, with ties broken by
    # ASCENDING expert id (a STABLE descending argsort preserves input order for
    # equal keys, and the input is ascending id). This is a CONTRACT requirement,
    # not an implementation detail: bf16/fp16 logits quantize many experts to the
    # same fp32 probability, so exact ties are NOT measure-zero here, and every
    # backend must agree element-wise. The Triton device kernel achieves the same
    # order for free because ``tl.argmax`` returns the LOWEST index on ties -- so
    # the iterative argmax naturally picks lower-id experts first among ties.
    # (``torch.topk`` was NOT used: its tie-break is unspecified and diverges from
    # tl.argmax, which false-fails element-wise parity on bf16 inputs.)
    order = torch.argsort(probs, dim=1, descending=True, stable=True)  # [M, E]
    ids = order[:, : int(topk)]                                   # [M, topk] int64
    weights = probs.gather(1, ids)                               # [M, topk] fp32
    if renormalize:
        weights = weights / weights.sum(dim=1, keepdim=True)
    return weights.contiguous(), ids.to(torch.int32).contiguous()


register("topk_softmax", Backend.REFERENCE)(topk_softmax_ref)


def topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    *,
    renormalize: bool = True,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE gating: softmax + top-k + optional renormalization. See
    :func:`topk_softmax_ref` for the semantics. Returns ``(topk_weights, topk_ids)``.

    ``backend="auto"`` picks the fastest registered backend (Triton device kernel
    when available, else the pure-torch reference).
    """
    return dispatch(
        "topk_softmax",
        gating_output,
        topk,
        renormalize=renormalize,
        backend=backend,
    )
