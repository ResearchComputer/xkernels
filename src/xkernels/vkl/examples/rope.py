# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored rotary position embedding (RoPE) — issue #68.

RoPE is the showcase for the **data-addressing family** added to the math IR
(``Gather``/``Slice``/``Unsqueeze``/``Concat``, docs/brainstorm/06 A4 case (a)).
The op applies rotary embeddings from a precomputed cos/sin cache, the
flashinfer ``apply_rope_with_cos_sin_cache`` convention:

  * ``positions`` ``[T] int`` — absolute token position (the GATHER INDEX).
  * ``query``/``key`` ``[T, H, D]`` — the tensors to rotate.
  * ``cos_sin_cache`` ``[P, D]`` — ``concat(cos[:, :D/2], sin[:, :D/2])`` packed,
    so columns ``[0, D/2)`` are cos and ``[D/2, D)`` are sin over the ``D/2``
    rotation frequencies.

The math is a fixed DAG the (now addressing-complete) oracle expresses end-to-end:

  1. ``cs = gather(cos_sin_cache, positions)``           — the data-dependent
     **addressing** (the cache row is selected by the ``positions`` INPUT, not by
     a value computed in the kernel — case (a), oracle-safe).
  2. split ``cs`` and each head into rotation halves via ``Slice(axis=-1, 0,
     "shape//2")`` / ``("shape//2", "shape")`` — ``head_size`` is symbolic (``D``)
     at trace time, so the bounds are ``str`` exprs over ``shape``.
  3. ``unsqueeze`` cos/sin to ``[T,1,D/2]`` so they broadcast over heads.
  4. the rotate-half products (``o1 = q1*cos - q2*sin``; ``o2 = q2*cos + q1*sin``)
     — pointwise, fp32.
  5. ``concat`` the halves back along the head axis → ``[T,H,D]``.

This is bit-exact with the hand ``minisgl`` ``_rope_ref`` (the rotate-half form;
flashinfer's packed-cache kernel computes the identical products directly). The
CPU oracle IS the reference; the reference card passes ``verify`` on CPU.

**Device codegen (KNOWN BUG — not yet verified on device):** RoPE is a
multi-dim (``[T,H,D]``) gather + per-axis broadcast, so it exercises the
multi-dim addressing lowering (``_TritonGenMultiDim`` in ``lower/mathbody.py``):
a flat-tiled grid whose kernel decomposes each lane's offset into per-axis
coords and lets every ``Gather``/``Slice``/``Concat``/``Unsqueeze`` compute its
own address. **The CPU oracle + reference card ARE bit-exact (``max_rel=0.0``,
``verify("apply_rope.reference@1.0.0")`` passes), but the generated triton
DEVICE kernel currently CRASHES with an illegal-memory-access on GB10 (sm_121) —
a true OOB in the multi-dim address computation, confirmed by
``compute-sanitizer``.** Earlier docstrings claimed "verified bit-exact on
GB10"; that was stale (almost certainly ``TRITON_INTERPRET=1``, which gives
false confidence — it materializes every ``tl.load`` in bounds; see the
``diagnose-wrong-results`` skill). The triton backend is therefore NOT wired
into ``ops/attention``; the public ``xkernels.apply_rope`` dispatches to
REFERENCE until the lowering is fixed. Diagnosis + repro:
``meta/docs/wiki/04-gotchas.md`` §14. The data-addressing family does NOT yet
close the A4 loop on real hardware — only on the CPU oracle.
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

__all__ = ["apply_rope"]


@kernel(
    id="apply_rope@1.0.0",
    kernel="apply_rope",
    canonical_op="attention",
    name="rotary position embedding (RoPE) from a cos/sin cache",
    signature=(
        "(query_out[T,H,D], key_out[T,H,D]) = rope(query, key, positions, "
        "cos_sin_cache); cs=gather(cache, positions); halves=split(D/2); "
        "out = concat(q1*cos-q2*sin, q2*cos+q1*sin)"
    ),
    inputs={
        # float tensors first so the output-dtype representative is a float, not
        # the int positions index (out_dtype resolution picks the first input).
        "query": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "H", "D")),
        "key": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "H", "D")),
        "cos_sin_cache": TensorDecl(rank=2, dtype=(fp32,), symbols=("P", "D")),
        "positions": TensorDecl(rank=1, dtype=(int32,), symbols=("T",)),
    },
    outputs={
        "query_out": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "H", "D")),
        "key_out": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "H", "D")),
    },
    constraints=("D % 2 == 0",),
    preconditions=(
        "cos_sin_cache columns [0, D/2) are cos, [D/2, D) are sin (packed);",
        "positions[t] < P (a valid row index into cos_sin_cache);",
        "query.dtype == key.dtype == {query_out,key_out}.dtype;",
        "D (head_size) in {64, 128, 256, 512} (the supported RoPE head sizes).",
    ),
    numerics=Numerics(
        rtol=1e-2,
        atol=1e-2,
        cross_backend_rtol=1e-2,
        by_dtype={
            "fp32": {"rtol": 1e-5, "atol": 1e-6},
            "bf16": {"rtol": 1e-2, "atol": 1e-2},
            "fp16": {"rtol": 1e-2, "atol": 1e-2},
        },
        notes=(
            "RoPE rotate-half from a packed cos/sin cache. Pure pointwise over the "
            "rotation products + a data-ADDRESSING gather of the cache by the "
            "positions input (no data-dependent control flow). fp32 products, cast "
            "to the head dtype on store. Bit-exact with minisgl _rope_ref."
        ),
    ),
    shape_sweep="apply_rope",
    fusions=(),
)
@launch(Launch.elementwise())  # routes to the multi-dim addressing lowering
                        # (_TritonGenMultiDim) because the body has addressing nodes
@targets(triton=Target(
    backend="triton",
    arch="amd_cdna3",
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "per-token cos/sin-cache gather + rotate-half pointwise. Memory-bound "
        "(one cache read per token, broadcast over heads). Device lowering needs "
        "the multi-dim gather launch pattern (docs/brainstorm/06 A4 follow-up)."
    ),
))
def apply_rope(ctx):
    """Build the RoPE math IR: gather -> split halves -> rotate-half -> concat."""
    # 1. data-ADDRESSING gather: pick each token's cos/sin row by its position.
    cs = ctx.gather(ctx.load("cos_sin_cache"), ctx.load("positions"), axis=0)  # [T, D]
    # 2. split the packed cache into cos (first half) / sin (second half).
    cos = ctx.unsqueeze(ctx.slice(cs, axis=1, start=0, stop="shape//2"), axis=1)   # [T,1,D/2]
    sin = ctx.unsqueeze(ctx.slice(cs, axis=1, start="shape//2", stop="shape"), axis=1)
    # 3. rotate-half products in fp32 (cos/sin broadcast over the head axis).
    q = ctx.load("query").cast("fp32")
    k = ctx.load("key").cast("fp32")
    q1 = ctx.slice(q, axis=2, start=0, stop="shape//2")
    q2 = ctx.slice(q, axis=2, start="shape//2", stop="shape")
    k1 = ctx.slice(k, axis=2, start=0, stop="shape//2")
    k2 = ctx.slice(k, axis=2, start="shape//2", stop="shape")
    oq = ctx.concat(q1 * cos - q2 * sin, q2 * cos + q1 * sin, axis=2)  # [T,H,D]
    ok = ctx.concat(k1 * cos - k2 * sin, k2 * cos + k1 * sin, axis=2)
    # 4. cast back to the head dtype; store both outputs.
    ctx.store("query_out", oq.cast(ctx.out_dtype()))
    ctx.store("key_out", ok.cast(ctx.out_dtype()))
