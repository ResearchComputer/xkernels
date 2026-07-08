# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public attention ops (``mha_merge_state``, ``dsa_indexer_logits``,
    ``apply_rope``): each dispatches to a registered backend."""

from __future__ import annotations

import torch

from ..._backends import Backend
from ..._dispatch import dispatch
from . import (
    dsa_reference,  # noqa: F401  (registers dsa_indexer_logits REFERENCE)
    reference,  # noqa: F401  (registers REFERENCE backend)
    sparse_mla_reference,  # noqa: F401  (registers sparse_mla_attention REFERENCE)
)
from . import (
    paged_attention as _paged_attn_mod,  # noqa: F401  (registers paged_attention REFERENCE)
)
from . import (
    paged_attention_prefill as _paged_prefill_mod,  # noqa: F401  (registers paged_attention_prefill REFERENCE)
)
from .dsa_reference import (
    dsa_indexer_topk_from_logits,  # noqa: F401  (re-export diagnostics helper)
)
from .paged_attention import paged_attention  # noqa: F401  (public re-export)
from .sparse_mla_decode import flash_mla_with_kvcache  # noqa: F401  (re-export decode)


def dsa_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """DeepSeek-V4 DSA indexer logits (issue #27): weighted ReLU MQA scores that
    select the top-k KV for sparse attention. Portable gfx942 replacement for the
    NVIDIA-only ``deep_gemm.fp8_fp4_mqa_logits``.

    ``logits[t, j] = sum_h weights[t, h] * relu(q[t, h, :] . k[j, :])``.

    Args:
        q: ``[T, H, D]`` indexer queries (``H = index_n_heads``, ``D = index_head_dim``).
        k: ``[K, D]`` single shared indexer key per KV position (MQA).
        weights: ``[T, H]`` (or ``[T, H, 1]``) per-head combine weights.
        lengths: optional ``[T]`` int — valid KV columns per query.
        row_starts: optional ``[T]`` int — first valid KV column per query
            (defaults to 0). Columns outside the window are masked to ``-inf``.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``logits [T, K]`` fp32. Pair with :func:`dsa_indexer_topk_from_logits` to obtain the
        top-k KV indices, or call :func:`dsa_indexer_topk` for the fused path.
    """
    return dispatch(
        "dsa_indexer_logits",
        q,
        k,
        weights,
        lengths=lengths,
        row_starts=row_starts,
        backend=backend,
    )


def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    topk: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
    backend: Backend | str = "auto",
) -> torch.Tensor:
    """Fused DSA indexer top-k (issue #54): weighted ReLU MQA logits + top-k
    selection in one pass, without materializing the full ``[T, K]`` fp32 logits.

    ``indices[t, :] = topk_w(mask(sum_h weights[t,h] * relu(q[t,h,:] . k[j,:])))``
    in **descending logit order** (ties by ascending KV id).

    This is the fused replacement for the two-step path
    (``dsa_indexer_logits`` + ``dsa_indexer_topk_from_logits``): a streaming
    Triton kernel computes logits tile-by-tile and keeps a running top-k
    candidate buffer, writing only the selected ``[T, topk]`` int32 indices.

    Args:
        q: ``[T, H, D]`` indexer queries (``H = index_n_heads``, ``D = index_head_dim``).
        k: ``[K, D]`` single shared indexer key per KV position (MQA).
        weights: ``[T, H]`` (or ``[T, H, 1]``) per-head combine weights.
        topk: number of KV positions to select per query (``1 <= topk <= K``).
        lengths: optional ``[T]`` int — valid KV columns per query.
        row_starts: optional ``[T]`` int — first valid KV column per query
            (defaults to 0). Columns outside the window are masked to ``-inf``
            and never selected.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``indices [T, topk]`` int32 in descending-logit order (ties by ascending
        KV id). See registry/ops/dsa_indexer_topk.spec.json numerics.notes for the
        boundary-discontinuity caveat on bf16/fp16 near-ties.
    """
    return dispatch(
        "dsa_indexer_topk",
        q,
        k,
        weights,
        topk=topk,
        lengths=lengths,
        row_starts=row_starts,
        backend=backend,
    )


def mha_merge_state(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention partials by their log-sum-exp (online softmax).

    For chunked-prefill / split-KV MLA: combine per-KV-chunk partial outputs and
    LSEs into a single output + merged LSE.

    Args:
        out_a, out_b: ``[T, H, D]`` partial outputs (bf16 or fp32).
        lse_a, lse_b: ``[T, H]`` fp32 log-sum-exp (natural-log basis).
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out [T, H, D] in out_a.dtype, lse [T, H] fp32)``.
    """
    return dispatch("mha_merge_state", out_a, lse_a, out_b, lse_b, backend=backend)


def sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    d_v: int | None = None,
    backend: Backend | str = "auto",
    workspace=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek-V4 sparse-MLA attention compute (issue #32): flash softmax over
    the DSA-indexer-selected latent KV. Portable gfx942 replacement for the
    NVIDIA-only ``flash_mla`` sparse/decode kernels.

    ``workspace`` (optional :class:`SparseMlaAttentionWorkspace`): reuse
    preallocated ``out``/``lse``/``maxl`` buffers to avoid per-call allocation
    and enable graph capture (issue #52). Ignored by the reference backend.

    Args:
        q: ``[T, H, D]`` latent queries (D = kv_lora_rank + rope; V4: 512).
        kv: ``[Kv, D]`` shared latent MQA cache.
        indices: ``[T, topk]`` int32 columns into ``kv`` (``<0`` = padding).
        sm_scale: softmax scale on the q.k score.
        topk_length: optional ``[T]`` int — valid column count per query.
        attn_sink: optional ``[H]`` (or scalar) per-head sink logit.
        d_v: value/output dim (first ``d_v`` latent dims). Defaults to ``D``.
        backend: ``"auto"`` or a ``Backend`` / its string value.

    Returns:
        ``(out [T, H, d_v], lse [T, H] fp32, max_logits [T, H] fp32)``.
    """
    return dispatch(
        "sparse_mla_attention",
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        d_v=d_v,
        workspace=workspace,
        backend=backend,
    )


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    *,
    d_v: int | None = None,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prefill sparse-MLA (upstream-faithful name). ``kv`` is the bf16 latent
    workspace ``[Kv, 1, D]`` and ``indices`` is ``[T, 1, topk]`` (the ``1`` is the
    MQA KV head). Returns ``(out, max_logits, lse)`` in upstream order.
    """
    kv2 = kv.squeeze(1) if kv.dim() == 3 else kv
    idx2 = indices.squeeze(1) if indices.dim() == 3 else indices
    out, lse, maxl = sparse_mla_attention(
        q,
        kv2,
        idx2,
        sm_scale=sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        d_v=d_v,
        backend=backend,
    )
    return out, maxl, lse


def get_mla_metadata(*args, **kwargs) -> tuple[torch.Tensor, int]:
    """Scheduling metadata (upstream-faithful name). V4 calls this no-arg and
    threads ``[0]`` into the decode kernel as an opaque ``tile_scheduler_metadata``
    that this compute path ignores. Returns ``(placeholder int32 tensor,
    num_splits=1)`` — no split-KV scheduling (a future split path would reuse
    ``mha_merge_state`` #3).
    """
    return torch.empty(0, dtype=torch.int32), 1


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    backend: Backend | str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotary position embedding (RoPE) from a precomputed cos/sin cache (issue #68).

    The flashinfer ``apply_rope_with_cos_sin_cache`` convention (no ROCm wheel --
    this op is the AMD/ROCm path for mini-sglang). Applies the GPT-NeoX rotate-half
    rotation in fp32, in place of ``query`` / ``key`` conceptually (returns new
    tensors):

        cs = cos_sin_cache[positions]                 # [T, D] (packed)
        cos, sin = cs[:, :D/2], cs[:, D/2:]           # [T, D/2]
        q_rot = concat(q[..., :D/2]*cos - q[..., D/2:]*sin,
                       q[..., D/2:]*cos + q[..., :D/2]*sin)
        # same for key

    Args:
        query: ``[T, H, D]`` (bf16 / fp16); rotated in fp32, cast back on store.
        key: ``[T, H, D]`` (same dtype as ``query``).
        positions: ``[T]`` int32 absolute token positions (the gather index into
            ``cos_sin_cache``; must be ``< P``).
        cos_sin_cache: ``[P, D]`` fp32 packed cache -- columns ``[0, D/2)`` are cos,
            ``[D/2, D)`` are sin over the ``D/2`` rotation frequencies.
        backend: ``"auto"`` (triton when available, else reference) or a
            ``Backend`` / its string value.

    Returns:
        ``(query_out [T, H, D], key_out [T, H, D])`` in the input dtype.
    """
    return dispatch(
        "apply_rope",
        query=query,
        key=key,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        backend=backend,
    )


def paged_attention_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    scale: float,
    backend: Backend | str = "auto",
    workspace=None,
) -> torch.Tensor:
    """Variable-length paged grouped-query attention (PREFILL/EXTEND).

    Thin re-export of
    :func:`xkernels.ops.attention.paged_attention_prefill.paged_attention_prefill`.
    Passes ``workspace`` through to the Triton backend (issue #52).
    """
    return dispatch(
        "paged_attention_prefill",
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        scale=scale,
        workspace=workspace,
        backend=backend,
    )
