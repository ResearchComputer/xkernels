# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton ``moe_align_block_size`` (issue #4): sort/pad routed token-slots into
per-expert blocks, the vLLM/SGLang-style histogram + padded prefix-sum + scatter.

The fused-MoE block GEMM consumes ``(sorted_token_ids, expert_ids,
num_tokens_post_padded)`` so each ``BLOCK_SIZE_M`` row tile maps to exactly one
expert. The reference (``moe_align_block_size_ref``) is a correctness-first torch
loop (argsort + per-expert python padding); this is the on-device port.

Algorithm (mirrors vLLM's ``moe_align_block_size_triton`` 4-stage kernel, with a
vectorized ``expert_ids`` pass instead of a data-dependent fill loop so it also
runs unchanged under ``TRITON_INTERPRET=1``):

The flattened ``M*top_k`` token-slots are split into ``num_experts`` contiguous
chunks, one per program. ``tokens_cnts`` is a ``[num_experts+1, num_experts]``
matrix: row ``p+1`` = histogram of program ``p``'s chunk over experts.

* **stage1 (count)**  each program tallies its chunk into row ``pid+1``.
* **stage2 (cumsum down programs)**  column ``e`` is prefix-summed over the
  program rows, so ``tokens_cnts[p, e]`` becomes the count of expert ``e`` in
  chunks ``0..p-1`` — i.e. program ``p``'s base offset inside expert ``e``'s run.
* **stage3 (padded prefix-sum)**  a single program rounds each expert's total up
  to a ``block_size`` multiple into ``cumsum[e+1]`` and writes
  ``num_tokens_post_padded``.
* **expert_ids**  one program per block ``b`` finds its owning expert as
  ``#{e in [1, num_experts] : cumsum[e] <= b*block_size}`` (a vectorized scan of
  the monotone ``cumsum``).
* **stage4 (scatter)**  each program walks its chunk again and writes token-slot
  ``t`` to ``cumsum[e] + tokens_cnts[pid, e]`` (post-incremented).

Because chunks are contiguous and walked in order, within-expert ordering is the
same stable order as the reference's ``argsort(stable=True)``, so the full output
triple matches the reference bit-for-bit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["moe_align_block_size_triton", "moe_align_block_size_ep_triton"]

# Sentinel for masked-out cumsum lanes. Must be a `tl.constexpr` instance (not a
# plain module global) to be readable from inside a @triton.jit function — the
# real compiler rejects bare globals even though the interpreter tolerates them.
_I32_MAX = tl.constexpr(2147483647)


@triton.jit
def _align_stage1_count(
    topk_ids_ptr,
    tokens_cnts_ptr,  # [num_experts+1, num_experts] int32, zero-initialised
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    """Per-program histogram of its token chunk into ``tokens_cnts[pid+1, :]``."""
    pid = tl.program_id(0)
    start = pid * tokens_per_thread
    row = (pid + 1) * num_experts
    for i in range(tokens_per_thread):
        if start + i < numel:
            e = tl.load(topk_ids_ptr + start + i)
            cur = tl.load(tokens_cnts_ptr + row + e)
            tl.store(tokens_cnts_ptr + row + e, cur + 1)


@triton.jit
def _align_stage2_cumsum(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
):
    """Prefix-sum expert column ``pid`` down the program rows (in place)."""
    pid = tl.program_id(0)  # expert column
    last = 0
    for i in range(1, num_experts + 1):
        cur = tl.load(tokens_cnts_ptr + i * num_experts + pid)
        last = last + cur
        tl.store(tokens_cnts_ptr + i * num_experts + pid, last)


@triton.jit
def _align_stage3_pad(
    num_tokens_post_padded_ptr,  # [1] int32
    tokens_cnts_ptr,
    cumsum_ptr,  # [num_experts+1] int32, zero-initialised
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    """Round each expert's total up to a ``block_size`` multiple -> ``cumsum``."""
    last = 0
    final_row = num_experts * num_experts  # row `num_experts` holds the totals
    for i in range(1, num_experts + 1):
        cnt = tl.load(tokens_cnts_ptr + final_row + i - 1)
        last = last + tl.cdiv(cnt, block_size) * block_size
        tl.store(cumsum_ptr + i, last)
    tl.store(num_tokens_post_padded_ptr, last)


@triton.jit
def _align_expert_ids(
    cumsum_ptr,
    expert_ids_ptr,  # [num_m_blocks] int32
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    EXPERTS_P2: tl.constexpr,  # next_pow2(num_experts + 1)
):
    """Block ``b`` belongs to ``#{e in [1, num_experts] : cumsum[e] <= b*block}``.

    Vectorized scan of the monotone ``cumsum``. ``EXPERTS_P2`` is
    ``next_pow2(num_experts + 1)`` because ``tl.arange`` needs a power-of-2 length
    and the lanes must reach index ``num_experts`` (we read ``cumsum[1..num_experts]``);
    masked / out-of-range lanes load the ``_I32_MAX`` sentinel so they never count.
    """
    b = tl.program_id(0)
    off = b * block_size
    e = tl.arange(0, EXPERTS_P2)
    valid = (e >= 1) & (e <= num_experts)
    cs = tl.load(cumsum_ptr + e, mask=valid, other=_I32_MAX)
    expert = tl.sum(((cs <= off) & valid).to(tl.int32), axis=0)
    # Unused trailing blocks (off >= total padded) count all experts -> num_experts,
    # one past the valid 0-based range. Map to sentinel 0 (matches the tokenspeed
    # contract); used blocks are always < num_experts so they are untouched.
    expert = tl.where(expert >= num_experts, 0, expert)
    tl.store(expert_ids_ptr + b, expert)


@triton.jit
def _align_stage4_scatter(
    topk_ids_ptr,
    sorted_token_ids_ptr,  # [max_pad] int32, pre-filled with pad_id
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel: tl.constexpr,
    tokens_per_thread: tl.constexpr,
):
    """Scatter each token-slot to ``cumsum[e] + tokens_cnts[pid, e]`` (post-inc)."""
    pid = tl.program_id(0)
    start = pid * tokens_per_thread
    row = pid * num_experts  # this program's base offsets (from stage2)
    for i in range(tokens_per_thread):
        if start + i < numel:
            e = tl.load(topk_ids_ptr + start + i)
            cnt = tl.load(tokens_cnts_ptr + row + e)
            rank = cnt + tl.load(cumsum_ptr + e)
            tl.store(sorted_token_ids_ptr + rank, start + i)
            tl.store(tokens_cnts_ptr + row + e, cnt + 1)


def moe_align_block_size_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    truncate: bool = True,
    workspace=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort/pad routed token-slots into per-expert blocks (issue #4, Triton).

    Drop-in for ``moe_align_block_size_ref`` — identical signature and outputs.

    Args:
        topk_ids: ``[M, top_k]`` int32 expert id per token-slot.
        block_size: GEMM block size each expert run is padded up to.
        num_experts: number of experts.
        workspace: optional :class:`~xkernels.ops.moe.workspace.MoeAlignWorkspace`
            (issue #52). When given and it ``matches`` the runtime shape, the
            five scratch buffers are RE-INITIALIZED IN PLACE into the workspace
            (``sorted_ids`` filled with ``pad_id``, the rest zeroed) so the
            buffer addresses are stable across calls -- the precondition for
            CUDA/HIP graph capture. The init cost is unchanged (counters, not
            fully-overwritten); the win is address stability. ``None`` = allocate.

    Returns:
        ``(sorted_token_ids [max_pad], expert_ids [n // block_size],
        num_tokens_post_padded [1])`` where ``max_pad = M*top_k +
        (num_experts+1)*(block_size-1)``, ``n = num_tokens_post_padded`` and
        unused ``sorted_token_ids`` slots hold ``pad_id = M*top_k``.

        With ``truncate=False`` (graph-capturable), ``expert_ids`` is the full
        ``max_blocks = cdiv(max_pad, block_size)`` length with unused trailing
        blocks set to 0 and no device->host sync.
    """
    flat = topk_ids.reshape(-1).contiguous()
    numel = flat.numel()
    pad_id = numel
    M, top_k = topk_ids.shape
    device = topk_ids.device

    max_pad = numel + (num_experts + 1) * (block_size - 1)
    max_blocks = triton.cdiv(max_pad, block_size)

    # Reuse the workspace buffers when provided + matching (issue #52): re-init
    # the counters/fills IN PLACE so the addresses are stable for graph capture.
    # These are histogram counters / accumulators, so the init is load-bearing
    # and cannot be skipped -- the win is address stability, not skipping zero.
    if workspace is not None and workspace.matches(
        M, top_k, num_experts, block_size, device=device
    ):
        sorted_ids = workspace.sorted_ids
        expert_ids = workspace.expert_ids
        num_post = workspace.num_post
        tokens_cnts = workspace.tokens_cnts
        cumsum = workspace.cumsum
        sorted_ids.fill_(pad_id)
        expert_ids.zero_()
        num_post.zero_()
        tokens_cnts.zero_()
        cumsum.zero_()
    else:
        sorted_ids = torch.full((max_pad,), pad_id, dtype=torch.int32, device=device)
        expert_ids = torch.zeros((max_blocks,), dtype=torch.int32, device=device)
        num_post = torch.zeros((1,), dtype=torch.int32, device=device)
        tokens_cnts = torch.zeros((num_experts + 1, num_experts), dtype=torch.int32, device=device)
        cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=device)

    tokens_per_thread = triton.cdiv(numel, num_experts)
    grid = (num_experts,)

    _align_stage1_count[grid](flat, tokens_cnts, num_experts, numel, tokens_per_thread)
    _align_stage2_cumsum[grid](tokens_cnts, num_experts)
    _align_stage3_pad[(1,)](num_post, tokens_cnts, cumsum, num_experts, block_size)
    _align_expert_ids[(max_blocks,)](
        cumsum,
        expert_ids,
        num_experts,
        block_size,
        EXPERTS_P2=triton.next_power_of_2(num_experts + 1),
    )
    _align_stage4_scatter[grid](
        flat, sorted_ids, tokens_cnts, cumsum, num_experts, numel, tokens_per_thread
    )

    if truncate:
        # Eager mode: device->host sync to trim expert_ids to the used blocks.
        n = int(num_post.item())
        return sorted_ids[:max_pad], expert_ids[: n // block_size], num_post
    # Sync-free / fixed-shape mode (graph-capturable): no .item(); expert_ids is
    # the full max_blocks length with unused trailing blocks = 0. Slice
    # sorted_ids / expert_ids to the runtime-M bounds so a workspace allocated
    # for a larger max_M returns the same M-shaped output as the alloc path
    # (issue #52 smaller-M reuse).
    return sorted_ids[:max_pad], expert_ids[:max_blocks], num_post


def moe_align_block_size_ep_triton(
    topk_ids: torch.Tensor,
    block_size: int,
    num_local_experts: int,
    expert_map: torch.Tensor,
    *,
    truncate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Device-side expert-parallel (EP) sort/pad routing — no host sync (issue #50).

    Drop-in device-side replacement for ``moe_align_block_size_ep`` that the fused
    MoE launchers call on GPU so every end-to-end call skips the torch argsort +
    Python-padding tax of the reference. ``num_local_experts`` (= the rank-local
    weight-row count, host-known from ``packed.shape[0]``) is passed in, so there
    is **no** ``e_local = .sum().item()`` sync; the global->local remap is a single
    device-side gather on ``expert_map`` and the dispatch is built by the
    sync-free Triton align (``truncate=False``).

    Routing (the "ghost expert" trick): each routed slot's global expert id is
    remapped to its local row via ``expert_map``; non-local ids collapse to ONE
    sentinel = ``num_local_experts`` (one past the valid local rows) instead of one
    bin per non-local expert. The 4-stage Triton align then runs over
    ``num_local_experts + 1`` experts, so the launch grid scales with the LOCAL
    expert count (not the global one) — matching the reference's small grid
    instead of inflating it ~``ep_size``x. The sentinel's blocks land in
    ``expert_ids`` as ``-1``; the fused-MoE GEMM's ``FILTER_EXPERT`` path skips
    them, so this rank computes only its local experts' partial output (the caller
    all-reduces the partials across the EP group).

    Args:
        topk_ids: ``[M, top_k]`` int32 **global** expert ids.
        block_size: GEMM block size each expert run is padded up to.
        num_local_experts: ``E_local`` — the rank-local expert (weight-row) count.
            Host-known (e.g. ``packed.shape[0]``); passing it in avoids the
            ``.sum().item()`` sync the reference pays on every call.
        expert_map: ``[num_global_experts]`` int tensor mapping global id -> local
            row in ``[0, num_local_experts)`` (``-1`` if not on this rank).
        truncate: forwarded to :func:`moe_align_block_size_triton`; defaults to
            ``False`` (sync-free / graph-capturable).

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)`` where
        ``expert_ids`` holds **local** rows in ``[0, num_local_experts)`` with
        ``-1`` for the collapsed non-local sentinel blocks (skipped by the GEMM's
        ``FILTER_EXPERT``).
    """
    e_local = num_local_experts  # host-known (rank-local weight-row count); no sync
    emap = expert_map.to(device=topk_ids.device)
    # Remap global -> local row; non-local (-1) collapses to one sentinel = e_local
    # so every routed id lands in [0, e_local] (valid for an e_local+1 expert align).
    local_or_neg = emap[topk_ids.long()].to(torch.int32)
    collapsed = torch.where(local_or_neg >= 0, local_or_neg, e_local)
    sorted_ids, expert_ids_g, num_post = moe_align_block_size_triton(
        collapsed, block_size, e_local + 1, truncate=truncate
    )
    # Sentinel blocks (owning expert == e_local) -> -1 for the GEMM's FILTER_EXPERT.
    # e_local is one past the valid local rows [0, e_local), so this never clobbers
    # a real local expert (including the e_local == 0 / owns-no-experts case).
    expert_ids = torch.where(
        expert_ids_g == e_local, torch.full_like(expert_ids_g, -1), expert_ids_g
    ).to(torch.int32)
    return sorted_ids, expert_ids, num_post


register("moe_align_block_size", Backend.TRITON)(moe_align_block_size_triton)
