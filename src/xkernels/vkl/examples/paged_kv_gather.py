# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored paged-KV-cache gather — the addressing family's N-D-index showcase.

This is the **second** op built on the data-addressing family
(``docs/brainstorm/06`` A4 case (a)), and the one that exercises the **N-D
index** gather path: where RoPE gathers a 2-D cache by a 1-D index
(``cache[positions]``), paged attention gathers a 4-D KV pool by a **2-D** index
(``pool[page_table]``). It is the make-or-break unpage primitive behind paged
attention (issue #71's building block) and RadixCache serving (vLLM /
mini-sglang): before attention can run, the per-sequence pages must be gathered
out of the shared pool by the request's ``page_table``.

The op is a single data-ADDRESSING gather:

  * ``kv_cache``    ``[num_pages, page_size, num_kv_heads, head_dim]``  — the
    paged pool (a single K or V cache; flashinfer/vLLM store K and V as separate
    tensors, so a real serving call gathers each once).
  * ``page_table``  ``[num_seqs, max_num_pages] int`` — for each sequence, the
    list of page indices into ``kv_cache``'s axis 0 (the GATHER INDEX).
  * ``out``         ``[num_seqs, max_num_pages, page_size, num_kv_heads, head_dim]``
    = ``kv_cache[page_table]`` — the index's FULL shape replaces axis 0
    (in-place ``index_select`` placement).

It is a **pure copy/gather**: zero arithmetic, so the correctness gate is the
sharpest possible — ``max_abs == 0`` and ``max_rel == 0`` are the ONLY passing
states; any drift is a codegen bug, never rounding. The torch oracle
(``kv_cache[page_table]`` advanced indexing) is bit-exact with the device
lowering (``tl.load(pool + page_table * stride)``), because the index is an
INPUT tensor (no data-dependent control flow — case (a), oracle-safe).

Why this is DSL-expressible but ``paged_attention`` (#71) is NOT: the gather
itself is pure addressing, so it lives here. What makes #71 hand-path is its
**ragged/segmented reduction** (``cu_seqlens``: each request attends over a
*different, data-determined* number of keys) plus online-softmax flash decoding
(the A4-case-(b) monoid). The unpage step this op factors out is the clean,
DSL-expressible slice of #71 — the consumer's job is to take this gathered
output and run the (hand-path) attention reduction over it.
"""
from __future__ import annotations

from .. import (
    Launch,
    Numerics,
    Target,
    TensorDecl,
    kernel,
    launch,
    targets,
)
from ..tiles import bf16, fp16, fp32, int32

__all__ = ["paged_kv_gather"]


@kernel(
    id="paged_kv_gather@1.0.0",
    kernel="paged_kv_gather",
    canonical_op="gather",
    name="gather a paged KV cache by a page table (paged-attention unpage step)",
    signature=(
        "out[num_seqs, max_num_pages, page_size, num_kv_heads, head_dim] = "
        "kv_cache[page_table]; a pure data-ADDRESSING gather (N-D index)"
    ),
    inputs={
        # float tensor first so the output-dtype representative is the cache dtype.
        "kv_cache": TensorDecl(
            rank=4, dtype=(bf16, fp16, fp32),
            symbols=("num_pages", "page_size", "num_kv_heads", "head_dim"),
        ),
        "page_table": TensorDecl(rank=2, dtype=(int32,), symbols=("num_seqs", "max_num_pages")),
    },
    outputs={
        "out": TensorDecl(
            rank=5, dtype=(bf16, fp16, fp32),
            symbols=("num_seqs", "max_num_pages", "page_size", "num_kv_heads", "head_dim"),
        ),
    },
    constraints=(
        "page_size % 1 == 0",
    ),
    preconditions=(
        "page_table[s, p] < num_pages (a valid page index into kv_cache axis 0);",
        "kv_cache.dtype == out.dtype (the gather copies, it does not cast);",
        "num_kv_heads <= num_qo_heads at the consuming attention op (the GQA ratio;",
        "  Qwen3-4B = 8 KV heads for 32 QO heads) — enforced downstream, not here.",
    ),
    numerics=Numerics(
        rtol=0.0,
        atol=0.0,
        cross_backend_rtol=0.0,
        by_dtype={
            "fp32": {"rtol": 0.0, "atol": 0.0},
            "bf16": {"rtol": 0.0, "atol": 0.0},
            "fp16": {"rtol": 0.0, "atol": 0.0},
        },
        notes=(
            "A pure gather/copy — no arithmetic, so the gate is bit-exact "
            "(max_abs == 0, max_rel == 0). Any drift is a codegen bug in the N-D "
            "gather lowering, never rounding. The torch oracle is "
            "kv_cache[page_table] (advanced indexing); the device lowering is "
            "tl.load(pool + page_table_value * stride)."
        ),
    ),
    shape_sweep="paged_kv_gather",
    fusions=(),
)
@launch(Launch.elementwise())  # routes to the multi-dim addressing lowering
                        # (_TritonGenMultiDim) because the body has a Gather node
@targets(triton=Target(
    backend="triton",
    arch="amd_cdna3",
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "Pure gather: one page read per (seq, page_slot), no compute. "
        "Memory-bound — the bandwidth ceiling is set by the 5-D copy volume. "
        "Exercises the N-D-index gather codegen (2-D page_table -> 5-D out)."
    ),
))
def paged_kv_gather(ctx):
    """Build the paged-gather math IR: a single N-D gather along the pool's axis 0."""
    pool = ctx.load("kv_cache")        # [num_pages, page_size, num_kv_heads, head_dim]
    pt = ctx.load("page_table")        # [num_seqs, max_num_pages]  (the 2-D index)
    out = ctx.gather(pool, pt, axis=0)  # -> [num_seqs, max_num_pages, page_size, ...]
    # Cast to the launch's runtime out_dtype before store. The gather output's IR
    # TensorRef carries the declared tuple's representative dtype (the first,
    # e.g. bf16); without this cast `_output_dtype` would allocate the output as
    # that literal even when the cache runs in fp32/fp16 -> bf16-quantization
    # drift. Every DSL op casts to out_dtype() before store for exactly this
    # reason (the cast is a no-op once dtypes align: out.dtype == cache.dtype).
    ctx.store("out", out.cast(ctx.out_dtype()))
