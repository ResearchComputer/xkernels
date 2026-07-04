# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Reusable workspace buffers for the hot MoE decode kernels (issue #52).

The MoE decode path has three allocation sites the issue names:

* ``moe_align_block_size`` (5 sort/pad scratch buffers),
* ``fused_moe_int4_w4a16`` (the fp32 combine / token-indexed scratch output),
* ``fused_moe_mxfp4`` (the SwiGLU activation scratch + fp32 combine output).

Each gets a per-op dataclass with ``allocate()`` (sized from deterministic shape
bounds) + ``matches()`` (validates the runtime shape fits, with the same
device/dtype + ``cuda`` vs ``cuda:0`` normalization as the attention
workspaces). Pass one via the public op's ``workspace=`` kwarg; it threads
through ``dispatch`` to the Triton backend.

STALE-DATA POLICY (issue validation bullet: "keep explicit zeroing only for
atomic-add combine outputs or EP partial outputs"):

* **Atomic-add outputs** (``combine_out`` for fused int4, ``out`` for mxfp4, EP
  ``scratch``) are RE-ZEROED every call into the workspace buffer. The kernel
  atomic-accumulates into them, so a stale nonzero would corrupt the result.
* **Fully-overwritten buffers** (non-EP ``scratch`` for int4, ``act`` for mxfp4)
  are written directly into the workspace slice with NO zero — every live
  element is assigned by the GEMM, so reuse is safe.
* **``moe_align`` scratch** (``tokens_cnts`` histogram, ``cumsum`` prefix-sum,
  ``sorted_ids`` pad-id fill) is RE-INITIALIZED every call — these are
  counters/accumulators, not fully-overwritten, so the init is load-bearing and
  cannot be skipped. The win for align is purely ADDRESS STABILITY (graph
  capture), not skipping the init kernel.

The eager-mode allocation savings are marginal (torch's caching allocator makes
``torch.empty``/``zeros`` nearly free on size reuse — see wiki §15); the
load-bearing value is enabling **CUDA/HIP graph capture**, which requires the
same buffer addresses across captures.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "MoeAlignWorkspace",
    "MoeInt4Workspace",
    "MoeMxfp4Workspace",
]


def _same_device(a: torch.device, b) -> bool:
    """Device equality resolving the ``cuda`` vs ``cuda:0`` ambiguity.

    Mirrors ``xkernels.ops.attention.workspace._same_device``.
    """
    bd = torch.device(b) if not isinstance(b, torch.device) else b
    if a.type != bd.type:
        return False
    if a.type != "cuda":
        return True
    ai = a.index if a.index is not None else torch.cuda.current_device()
    bi = bd.index if bd.index is not None else torch.cuda.current_device()
    return ai == bi


def _align_max_pad(M: int, top_k: int, num_experts: int, block_size: int) -> int:
    """The ``moe_align_block_size`` output length (deterministic). Mirrors the
    formula in ``align_kernel.moe_align_block_size_triton`` /
    ``w4a16.moe_align_block_size_ref``::

        max_pad = M*top_k + (num_experts+1)*(block_size-1)
    """
    return M * top_k + (num_experts + 1) * (block_size - 1)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


# ═══════════════════════════════════════════════════════════════════════════════
# moe_align_block_size
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MoeAlignWorkspace:
    """Preallocated sort/pad scratch for ``moe_align_block_size``.

    Five int32 buffers, sized from ``(M, top_k, num_experts, block_size)``:

    * ``sorted_ids``  ``[max_pad]`` — token-id slots, unused hold ``pad_id``
    * ``expert_ids``  ``[max_blocks]`` — per-block expert id
    * ``num_post``    ``[1]`` — used token count
    * ``tokens_cnts`` ``[num_experts+1, num_experts]`` — per-thread expert histogram
    * ``cumsum``      ``[num_experts+1]`` — exclusive prefix sum of expert counts

    All five are RE-INITIALIZED every call (fill ``sorted_ids`` with ``pad_id``,
    zero the rest) because they are counters/accumulators the kernel increments
    and scatter-writes into — reuse is for address stability (graph capture),
    NOT to skip the init. The ``truncate=False`` path is already sync-free
    (no ``.item()`` host round-trip), so a workspace makes the whole align call
    graph-capturable.
    """

    sorted_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_post: torch.Tensor
    tokens_cnts: torch.Tensor
    cumsum: torch.Tensor

    @classmethod
    def allocate(
        cls,
        M: int,
        top_k: int,
        num_experts: int,
        block_size: int,
        *,
        device,
    ) -> MoeAlignWorkspace:
        max_pad = _align_max_pad(M, top_k, num_experts, block_size)
        max_blocks = _cdiv(max_pad, block_size)
        return cls(
            sorted_ids=torch.empty(max_pad, dtype=torch.int32, device=device),
            expert_ids=torch.empty(max_blocks, dtype=torch.int32, device=device),
            num_post=torch.empty(1, dtype=torch.int32, device=device),
            tokens_cnts=torch.empty(
                (num_experts + 1, num_experts), dtype=torch.int32, device=device
            ),
            cumsum=torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        )

    def matches(
        self,
        M: int,
        top_k: int,
        num_experts: int,
        block_size: int,
        *,
        device,
    ) -> bool:
        max_pad = _align_max_pad(M, top_k, num_experts, block_size)
        max_blocks = _cdiv(max_pad, block_size)
        return (
            self.sorted_ids.numel() >= max_pad
            and self.expert_ids.numel() >= max_blocks
            and self.num_post.numel() >= 1
            and self.tokens_cnts.shape == (num_experts + 1, num_experts)
            and self.cumsum.numel() >= num_experts + 1
            and _same_device(self.sorted_ids.device, device)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# fused_moe_int4_w4a16
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MoeInt4Workspace:
    """Preallocated combine / scratch buffers for ``fused_moe_int4_w4a16``.

    Two output buffers, sized from ``(max_M, top_k, N)``:

    * ``combine_out`` ``[max_M, N]`` fp32 — used by the ``fused_combine=True``
      path (atomic-add accumulate); RE-ZEROED every call.
    * ``scratch``     ``[max_M*top_k, N]`` in ``dtype`` — used by the
      ``fused_combine=False`` path; written directly with NO zero under non-EP
      (every element assigned by the GEMM), RE-ZEROED under EP (atomic-add for
      the rank partial).

    The workspace also carries an optional inner ``align_workspace`` so the
    whole call (align + GEMM) is graph-capturable: the Triton backend threads it
    to its internal ``moe_align_block_size`` call. Sized lazily on first use if
    not provided, since its ``block_size`` depends on the resolved GEMM config.

    ``allocate()`` sizes BOTH output buffers — only the one for the active
    ``fused_combine`` path is read each call (the other is idle memory, small at
    decode buckets).
    """

    combine_out: torch.Tensor
    scratch: torch.Tensor
    dtype: torch.dtype
    align_workspace: MoeAlignWorkspace | None = None

    @classmethod
    def allocate(
        cls,
        max_M: int,
        top_k: int,
        N: int,
        *,
        dtype: torch.dtype,
        device,
    ) -> MoeInt4Workspace:
        return cls(
            combine_out=torch.empty((max_M, N), dtype=torch.float32, device=device),
            scratch=torch.empty((max_M * top_k, N), dtype=dtype, device=device),
            dtype=dtype,
        )

    def matches(
        self,
        M: int,
        top_k: int,
        N: int,
        *,
        dtype: torch.dtype,
        device,
    ) -> bool:
        return (
            self.combine_out.shape[1] == N
            and self.combine_out.shape[0] >= M
            and self.scratch.shape[1] == N
            and self.scratch.shape[0] >= M * top_k
            and self.scratch.dtype == dtype
            and _same_device(self.combine_out.device, device)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# fused_moe_mxfp4
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MoeMxfp4Workspace:
    """Preallocated activation / combine buffers for ``fused_moe_mxfp4``.

    Two buffers, sized from ``(max_M, top_k, num_experts, block_size, ispp, hidden)``:

    * ``act`` ``[max_pad, ispp]`` in ``dtype`` — the SwiGLU activation scratch
      (gate_up GEMM output, down GEMM input); written directly with NO zero
      (every element assigned by the gate_up GEMM).
    * ``out`` ``[max_M, hidden]`` fp32 — the down-stage combine output;
      RE-ZEROED every call (atomic-add accumulate).

    Like ``MoeInt4Workspace``, carries an optional inner ``align_workspace``
    for whole-call graph capture. ``max_pad`` follows the ``moe_align`` formula
    (``M*top_k + (E+1)*(block_size-1)``); size it for the max decode batch.
    """

    act: torch.Tensor
    out: torch.Tensor
    dtype: torch.dtype
    align_workspace: MoeAlignWorkspace | None = None

    @classmethod
    def allocate(
        cls,
        max_M: int,
        top_k: int,
        num_experts: int,
        block_size: int,
        ispp: int,
        hidden: int,
        *,
        dtype: torch.dtype,
        device,
    ) -> MoeMxfp4Workspace:
        max_pad = _align_max_pad(max_M, top_k, num_experts, block_size)
        return cls(
            act=torch.empty((max_pad, ispp), dtype=dtype, device=device),
            out=torch.empty((max_M, hidden), dtype=torch.float32, device=device),
            dtype=dtype,
        )

    def matches(
        self,
        M: int,
        top_k: int,
        num_experts: int,
        block_size: int,
        ispp: int,
        hidden: int,
        *,
        dtype: torch.dtype,
        device,
    ) -> bool:
        max_pad = _align_max_pad(M, top_k, num_experts, block_size)
        return (
            self.act.shape[1] == ispp
            and self.act.shape[0] >= max_pad
            and self.act.dtype == dtype
            and self.out.shape == (self.out.shape[0], hidden)
            and self.out.shape[0] >= M
            and _same_device(self.act.device, device)
        )
