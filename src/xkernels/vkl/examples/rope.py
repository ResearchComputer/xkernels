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

**Device codegen — VERIFIED on GB10 (sm_121).** RoPE is a multi-dim (``[T,H,D]``)
gather + per-axis broadcast, lowered by ``_TritonGenMultiDim`` in
``lower/mathbody.py``: a flat-tiled grid whose kernel decomposes each lane's
offset into per-axis coords and lets every ``Gather``/``Slice``/``Concat``/
``Unsqueeze`` compute its own address. ``verify("apply_rope.triton@1.0.0")`` is
compiled=True, passed=True (5/5, max_rel 7.5e-3 < bf16 rtol);
``verify("apply_rope.reference@1.0.0")`` bit-exact; ``verify_parity`` agrees.
The triton backend is wired (``ops/attention/triton/rope_kernel.py``).

**The one gotcha (now fixed).** The per-axis broadcast index is emitted as
``coord % shape``, which is correct only for non-negative coords. A ``Concat``
b-branch shifts its output coord by ``-len_a`` (negative for the discarded
lanes), and CUDA/Triton ``%`` follows C sign (``-1 % 64 == -1``, NOT Python's
``63``), so an unfloored modulo yielded a negative offset → an OOB read *before*
the buffer. Fixed by ``_floored_mod`` (``((x%n)+n)%n``) in ``lower/mathbody.py``;
result-preserving for all non-negative-coord loads. The
``illegal-memory-access`` it caused was once misdiagnosed as a "multi-dim
decomposition" bug — see ``meta/docs/wiki/04-gotchas.md`` §14 for the full
lesson (always dump + read the *generated source* before theorizing about
codegen). The data-addressing family now closes the A4 loop on real hardware.
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


# ═══════════════════════════════════════════════════════════════════════════════
# apply_rope_gqa — grouped-query RoPE in ONE launch (issue #104)
# ═══════════════════════════════════════════════════════════════════════════════
#
# ``apply_rope@1.0.0`` (#68) declares both ``query`` and ``key`` as ``[T, H, D]``
# with a single shared ``H``. Every modern dense model is GQA (Qwen3 32/8, Llama3
# 32/8, Apertus 32/8): ``query`` has ``Hq = g * Hk`` heads while ``key`` keeps
# ``Hk``. mini-sglang's adapter (``ops/rotary.py``) therefore calls ``apply_rope``
# **twice** — rotating ``q`` and ``k`` separately, discarding one output each
# time — because the shared-``H`` spec forbids ``Hq != Hk`` in one call. That
# double-launch dominates decode TPOT at batch=1 on GB10 (issue #104: ~365 µs
# eager for the GQA pair vs ~92 µs for the torch reference).
#
# This op removes the double-launch: ``query`` carries ``Hq`` heads, ``key``
# carries ``Hk`` heads, and ``Hq % Hk == 0`` is the GQA divisibility constraint.
# The body is IDENTICAL to ``apply_rope`` — RoPE's cos/sin depend only on
# position (not head index), so the existing ``unsqueeze`` -> ``[T, 1, D/2]``
# broadcast over the head axis rotates ``Hq`` query heads and ``Hk`` key heads
# independently by the SAME per-position cos/sin, in one launch. The math is
# still a fixed DAG the addressing oracle expresses (gather -> split halves ->
# rotate-half -> concat), bit-exact with minisgl ``_rope_ref`` applied to each
# tensor.
#
# **Device lowering — VERIFIED on GB10 (sm_121).** The multi-dim lowering
# (``_TritonGenMultiDim`` / ``_launch_multidim`` in ``lower/mathbody.py``) was
# extended (#104) so each Store gets its OWN coord decomposition (from its own
# output shape) and its OWN ``offs < numel_<out>`` mask, over a grid sized by
# ``max(numel)``. That is what makes different-sized outputs
# (``query_out=[T,Hq,D]`` vs ``key_out=[T,Hk,D]``, ``Hq != Hk``) safe — the
# smaller output is masked to its own numel, so the larger output's grid does
# not write out-of-bounds. The triton card is committed + registered
# (``rope_kernel.py``), and ``verify`` + ``verify_parity`` pass on GB10:
# ``verify("apply_rope_gqa.triton@1.0.0", arch="nvidia_sm121")`` 5/5
# (max_abs=1.5e-05), ``verify_parity`` agree (max_rel=0.0074 < 0.01). At
# batch=1 (T=1, Hq=32, Hk=8, D=128) the single GQA launch is 167 us vs 343 us
# for the old two-MHA-launch path (51% lower) — the issue-#104 win.
@kernel(
    id="apply_rope_gqa@1.0.0",
    kernel="apply_rope_gqa",
    canonical_op="attention",
    name="rotary position embedding (RoPE) for grouped-query attention (one launch)",
    signature=(
        "(query_out[T,Hq,D], key_out[T,Hk,D]) = rope(query[T,Hq,D], "
        "key[T,Hk,D], positions, cos_sin_cache); Hq = g*Hk; "
        "cs=gather(cache,positions); halves=split(D/2); "
        "out = concat(q1*cos-q2*sin, q2*cos+q1*sin)"
    ),
    inputs={
        # float tensors first so the output-dtype representative is a float.
        "query": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "Hq", "D")),
        "key": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "Hk", "D")),
        "cos_sin_cache": TensorDecl(rank=2, dtype=(fp32,), symbols=("P", "D")),
        "positions": TensorDecl(rank=1, dtype=(int32,), symbols=("T",)),
    },
    outputs={
        "query_out": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "Hq", "D")),
        "key_out": TensorDecl(rank=3, dtype=(bf16, fp16), symbols=("T", "Hk", "D")),
    },
    constraints=(
        "D % 2 == 0",
        "Hq % Hk == 0",
    ),
    preconditions=(
        "cos_sin_cache columns [0, D/2) are cos, [D/2, D) are sin (packed);",
        "positions[t] < P (a valid row index into cos_sin_cache);",
        "query.dtype == key.dtype == {query_out,key_out}.dtype;",
        "Hq % Hk == 0 (grouped-query: query has g*Hk heads, key has Hk heads; "
        "Hq == Hk is the MHA special case, also accepted);",
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
            "GQA RoPE rotate-half from a packed cos/sin cache: query (Hq heads) "
            "and key (Hk heads) rotate by the SAME per-position cos/sin (which "
            "broadcast over the head axis) in one launch, removing the "
            "double-launch the shared-H apply_rope@1.0.0 forced. Pure pointwise "
            "over the rotation products + a data-ADDRESSING gather of the cache "
            "by the positions input (no data-dependent control flow). fp32 "
            "products, cast to the head dtype on store. Bit-exact with minisgl "
            "_rope_ref applied to q and k separately."
        ),
    ),
    shape_sweep="apply_rope_gqa",
    fusions=(),
)
@launch(Launch.elementwise())  # multi-dim addressing lowering (per-output grids: #104)
@targets(triton=Target(
    backend="triton",
    arch="nvidia_sm121",  # GB10 — the serving regression target (issue #104)
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "per-token cos/sin-cache gather + rotate-half pointwise over BOTH the "
        "Hq query heads and the Hk key heads in one launch. Memory-bound (one "
        "cache read per token, broadcast over heads). Verified on GB10 (sm_121): "
        "verify 5/5, verify_parity agree. The multi-dim codegen "
        "(_TritonGenMultiDim/_launch_multidim) gives each Store its own coord "
        "decomposition + ``offs < numel_<out>`` mask so different-sized outputs "
        "(Hq != Hk) do not OOB (#104)."
    ),
))
def apply_rope_gqa(ctx):
    """GQA-native RoPE: rotate query (Hq) and key (Hk) in one launch (#104)."""
    # 1. data-ADDRESSING gather: each token's cos/sin row, shared by q and k.
    cs = ctx.gather(ctx.load("cos_sin_cache"), ctx.load("positions"), axis=0)  # [T, D]
    # 2. split the packed cache into cos (first half) / sin (second half) and
    #    unsqueeze over the head axis -> [T, 1, D/2], broadcasting to Hq AND Hk.
    cos = ctx.unsqueeze(ctx.slice(cs, axis=1, start=0, stop="shape//2"), axis=1)
    sin = ctx.unsqueeze(ctx.slice(cs, axis=1, start="shape//2", stop="shape"), axis=1)
    # 3. rotate-half products in fp32 (cos/sin broadcast over each head count).
    q = ctx.load("query").cast("fp32")   # [T, Hq, D]
    k = ctx.load("key").cast("fp32")      # [T, Hk, D]
    q1 = ctx.slice(q, axis=2, start=0, stop="shape//2")
    q2 = ctx.slice(q, axis=2, start="shape//2", stop="shape")
    k1 = ctx.slice(k, axis=2, start=0, stop="shape//2")
    k2 = ctx.slice(k, axis=2, start="shape//2", stop="shape")
    oq = ctx.concat(q1 * cos - q2 * sin, q2 * cos + q1 * sin, axis=2)  # [T, Hq, D]
    ok = ctx.concat(k1 * cos - k2 * sin, k2 * cos + k1 * sin, axis=2)  # [T, Hk, D]
    # 4. cast back to the head dtype; store both outputs (different sizes).
    ctx.store("query_out", oq.cast(ctx.out_dtype()))
    ctx.store("key_out", ok.cast(ctx.out_dtype()))
