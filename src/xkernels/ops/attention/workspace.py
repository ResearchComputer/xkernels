# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Reusable output-buffer workspaces for hot decode kernels (issue #52).

In decode serving, attention kernels run once per layer per token. Per-call
output allocation (``torch.empty``) both costs a small slice of latency and —
more importantly — defeats CUDA/HIP **graph capture**, which requires the same
memory addresses across captures. A workspace lets the serving stack
**allocate once per max decode shape and reuse** across decode steps.

This module ships per-op workspace dataclasses. Each bundles the output
tensor(s) a kernel would otherwise ``torch.empty`` on every call, plus an
``allocate(...)`` classmethod sized from the deterministic shape bounds and a
``matches(...)`` validator. Pass one to the public op via ``workspace=ws``;
``backend="auto"`` then routes it through ``dispatch`` to the Triton backend,
which writes into the preallocated buffer instead of allocating. The reference
backend ignores it (it always returns a fresh tensor).

STALE-DATA SAFETY (issue validation bullet 3): the kernels these workspaces
serve **fully overwrite every element** of their output tensors on every call
(the flash softmax always produces a value for every (token, head, dim); seq_ids
is fully overwritten by ``torch.searchsorted``). So reusing a *larger* workspace
for a smaller bucket is safe — the caller simply reads the valid leading slice
``out[:B]`` / ``out[:num_tokens]`` (the returned tensor is that slice). The MoE
combine outputs that need *selective* zeroing (atomic-add into skipped-expert
slots) are NOT in this family and are a separate, trickier follow-up.

Graph-capture recipe::

    ws = PagedAttentionWorkspace.allocate(B_max, H_q, D, device="cuda", dtype=bf16)
    # capture once:
    g = torch.cuda.CUDAGraph()
    out = xkernels.paged_attention(q, ..., workspace=ws)
    with torch.cuda.graph(g):
        out = xkernels.paged_attention(q, ..., workspace=ws)
    # replay every decode step (same addresses -> capturable):
    q.copy_(new_q); g.replay(); use(out)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "PagedAttentionWorkspace",
    "PagedAttentionPrefillWorkspace",
    "SparseMlaAttentionWorkspace",
]


def _same_device(a: torch.device, b) -> bool:
    """Device equality that resolves the ``cuda`` vs ``cuda:0`` ambiguity.

    ``torch.device('cuda') != torch.device('cuda:0')`` as objects even though
    both denote the current GPU; normalize to ``(type, index)`` so a caller
    passing ``device='cuda'`` matches a buffer allocated on ``cuda:0``.
    """
    bd = torch.device(b) if not isinstance(b, torch.device) else b
    if a.type != bd.type:
        return False
    if a.type != "cuda":
        return True  # cpu has no index ambiguity
    # index None ('cuda') means current device; resolve to torch's choice.
    ai = a.index if a.index is not None else torch.cuda.current_device()
    bi = bd.index if bd.index is not None else torch.cuda.current_device()
    return ai == bi


@dataclass
class PagedAttentionWorkspace:
    """Output workspace for :func:`xkernels.paged_attention` (decode).

    ``out [B, H_q, D]`` is the single output tensor. May be allocated at a
    ``B_max >= B`` (per-request max batch); the kernel returns the valid
    ``out[:B]`` slice. Allocate once per max decode batch and reuse.
    """

    out: torch.Tensor

    @classmethod
    def allocate(
        cls, B: int, H_q: int, D: int, *, device, dtype: torch.dtype
    ) -> PagedAttentionWorkspace:
        return cls(out=torch.empty(B, H_q, D, device=device, dtype=dtype))

    def matches(self, B: int, H_q: int, D: int, *, device, dtype: torch.dtype) -> bool:
        o = self.out
        return (
            o.ndim == 3
            and o.shape[1] == H_q
            and o.shape[2] == D
            and o.shape[0] >= B
            and _same_device(o.device, device)
            and o.dtype == dtype
        )


@dataclass
class PagedAttentionPrefillWorkspace:
    """Output + scratch workspace for :func:`xkernels.paged_attention_prefill`.

    ``out [num_tokens, H_q, D]`` — the output (may be allocated at
    ``num_tokens_max >= num_tokens``; the kernel returns ``out[:num_tokens]``).
    ``seq_ids [num_tokens]`` int32 — the per-token sequence index, computed each
    call by ``torch.searchsorted`` and fully overwritten (no stale-data risk).
    """

    out: torch.Tensor
    seq_ids: torch.Tensor

    @classmethod
    def allocate(
        cls, num_tokens: int, H_q: int, D: int, *, device, dtype: torch.dtype
    ) -> PagedAttentionPrefillWorkspace:
        return cls(
            out=torch.empty(num_tokens, H_q, D, device=device, dtype=dtype),
            seq_ids=torch.empty(num_tokens, device=device, dtype=torch.int32),
        )

    def matches(
        self, num_tokens: int, H_q: int, D: int, *, device, dtype: torch.dtype
    ) -> bool:
        return (
            self.out.ndim == 3
            and self.out.shape[1] == H_q
            and self.out.shape[2] == D
            and self.out.shape[0] >= num_tokens
            and _same_device(self.out.device, device)
            and self.out.dtype == dtype
            and self.seq_ids.ndim == 1
            and self.seq_ids.shape[0] >= num_tokens
            and _same_device(self.seq_ids.device, device)
            and self.seq_ids.dtype == torch.int32
        )


@dataclass
class SparseMlaAttentionWorkspace:
    """Output workspace for :func:`xkernels.sparse_mla_attention`.

    All three tensors are fully overwritten every call. May be allocated at
    ``T_max >= T`` (max query tokens); the kernel returns the ``[:T]`` slice of
    each.
    """

    out: torch.Tensor
    lse: torch.Tensor
    maxl: torch.Tensor

    @classmethod
    def allocate(
        cls, T: int, H: int, d_v: int, *, device, dtype: torch.dtype
    ) -> SparseMlaAttentionWorkspace:
        return cls(
            out=torch.empty(T, H, d_v, device=device, dtype=dtype),
            lse=torch.empty(T, H, device=device, dtype=torch.float32),
            maxl=torch.empty(T, H, device=device, dtype=torch.float32),
        )

    def matches(
        self, T: int, H: int, d_v: int, *, device, dtype: torch.dtype
    ) -> bool:
        return (
            self.out.ndim == 3
            and self.out.shape[1] == H
            and self.out.shape[2] == d_v
            and self.out.shape[0] >= T
            and _same_device(self.out.device, device)
            and self.out.dtype == dtype
            and self.lse.ndim == 2
            and self.lse.shape[1] == H
            and self.lse.shape[0] >= T
            and _same_device(self.lse.device, device)
            and self.lse.dtype == torch.float32
            and self.maxl.ndim == 2
            and self.maxl.shape[1] == H
            and self.maxl.shape[0] >= T
            and _same_device(self.maxl.device, device)
            and self.maxl.dtype == torch.float32
        )
