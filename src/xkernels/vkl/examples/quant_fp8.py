# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored fp8 quantization helpers (issue #57).

The fp8 blockscale GEMM (``mm_fp8_blockscale``) consumes per-token-group fp8
activations. Today the helper (``ops/gemm/reference.py:per_token_group_quant_fp8``)
is a Python loop of small torch ops — fine as the CPU/reference path, but on the
inference path (right before the fp8 GEMM) it launches many tiny kernels. This
op is the Triton-able contract for it: one program per group, amax + scale +
cast in a single launch.

The math is a fixed DAG the math IR now expresses end-to-end:
``amax = reduce_max(|x|)`` → ``scale = max(amax, eps) / FP8_MAX`` →
``q = clamp(x / scale, ±FP8_MAX).cast(fp8)``. That needs exactly the
``reduce_max`` + ``abs`` + ``maximum/minimum`` + literal + fp8-cast primitives
added alongside this op — every node is in the doc-10 pointwise+reduce set.

It is rowwise over the GROUPED 2D view ``x [G, B]`` (one row = one
quantization group of ``B`` elements): reduce over the last axis (the group),
emit TWO outputs of DIFFERENT dtype — ``q [G, B] fp8`` and ``scale [G] fp32``.
The mixed-dtype output is the load-bearing reason the lowering now resolves
per-output dtype from the IR instead of one global ``out_dtype``. The caller
reshapes the natural activation ``[M, K]`` to groups ``[M*(K//block), block]``
(i.e. ``x.view(M, K//block, block).reshape(-1, block)``) and reshapes the
outputs back (``q`` to ``[M, K]``, ``scale`` to ``[M, K//block]``).

Numerics are bit-exact with the hand ``per_token_group_quant_fp8`` (same op
order, same OCP ``scale = amax/FP8_MAX``), so the body IS a faithful
``per_block_quant_fp8`` applies the SAME rowwise quantization DAG to a grouped
view of weight blocks: ``x[G, B]`` where one row is one flattened
``block x block`` tile (``B = block*block``). The caller reshapes
``w[N,K] -> [ceil(N/block)*ceil(K/block), block*block]`` for full tiles and
reshapes ``q``/``scale`` back to ``[N,K]`` / ``[ceil(N/block), ceil(K/block)]``.
Non-full tail tiles need a padding wrapper or the hand helper path; the VKL op
is the full-tile fast path the rowwise lowering can express.

Numerics are bit-exact with the hand helpers over identical grouped/full-tile
operands. FP8_MAX = 448.0 (float8_e4m3fn, the OCP / NVIDIA encoding and the
``fp8`` short name); the fnuz (max 240) AMD-native encoding is an arch override
(mixed-precision-convert / port-across-arch), not the base op.
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
from ..tiles import bf16, fp32

__all__ = ["per_token_group_quant_fp8", "per_block_quant_fp8"]

# OCP fp8 e4m3fn max (matches ops/gemm/reference.py:_FP8_MAX). The fnuz variant
# (240, AMD CDNA3-native) is a per-arch override, not this base op.
FP8_MAX = 448.0
_EPS = 1e-12  # amax floor (matches the hand reference's .clamp_min(1e-12))


@kernel(
    id="per_token_group_quant_fp8@1.0.0",
    kernel="per_token_group_quant_fp8",
    canonical_op="reduce",
    name="per-token-group fp8 e4m3 quantization",
    signature=(
        "(q[G,B] fp8, scale[G] fp32) = quantize(x[G,B]) per group; "
        "amax=reduce_max|x|; scale=amax/FP8_MAX; q=clamp(x/scale).fp8"
    ),
    inputs={
        "x": TensorDecl(rank=2, dtype=(fp32, bf16), symbols=("G", "B")),
    },
    outputs={
        "q": TensorDecl(rank=2, dtype=("fp8",), symbols=("G", "B")),
        "scale": TensorDecl(rank=1, dtype=(fp32,), symbols=("G",)),
    },
    constraints=("B % 16 == 0",),
    preconditions=(
        "x is the GROUPED view [G, B] of an activation (G = num tokens * K//block);",
        "caller reshapes [M,K] -> [M*(K//block), block] and the outputs back",
        "fp8 dtype is float8_e4m3fn (FP8_MAX=448); fnuz (240) is an arch override",
    ),
    numerics=Numerics(
        rtol=1e-2,
        atol=1e-2,
        reduce_dtype=fp32,
        cross_backend_rtol=1e-2,
        by_dtype={
            "fp32": {"rtol": 1e-6, "atol": 1e-6},
            "bf16": {"rtol": 1e-2, "atol": 1e-2},
        },
        notes=(
            "Per-token-group fp8 quant. amax reduced in fp32; scale = amax/FP8_MAX "
            "(OCP e4m3fn, max 448). Bit-exact with the hand "
            "ops/gemm/reference.py:per_token_group_quant_fp8 (same op order). "
            "bf16 rtol covers the amax reduction order on GPU."
        ),
    ),
    shape_sweep="per_token_group_quant_fp8",
    fusions=(),
)
@launch(Launch.rowwise())  # one program per group; reduce_max over the group (last axis)
@targets(triton=Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "memory-bound row-wise reduce; one kernel for amax + scale + fp8 cast "
        "(replaces a Python loop of small torch ops on the inference path)."
    ),
))
def per_token_group_quant_fp8(ctx):
    """Build the math IR: amax -> scale -> clamp -> fp8 cast (two outputs)."""
    v = ctx.load("x").cast("fp32")                              # [G, B] fp32
    amax = ctx.reduce_max(ctx.abs(v), axis=1, accum_dtype="fp32")  # [G] fp32 (per group)
    amax_safe = ctx.maximum(amax, ctx.lit(_EPS))                # floor (matches ref clamp_min)
    scale = amax_safe / ctx.lit(FP8_MAX)                        # [G] fp32  -> output
    qf = v / scale                                              # [G, B] (broadcast)
    qf = ctx.minimum(ctx.maximum(qf, ctx.lit(-FP8_MAX)), ctx.lit(FP8_MAX))  # clamp ±FP8_MAX
    ctx.store("q", qf.cast("fp8"))                              # [G, B] fp8  -> output
    ctx.store("scale", scale)                                   # [G] fp32     -> output


@kernel(
    id="per_block_quant_fp8@1.0.0",
    kernel="per_block_quant_fp8",
    canonical_op="quantize",
    name="per-block fp8 e4m3 quantization",
    signature=(
        "(q[G,B] fp8, scale[G] fp32) = quantize(flattened block tiles x[G,B]); "
        "B = block*block; amax=reduce_max|x|; scale=amax/FP8_MAX"
    ),
    inputs={
        "x": TensorDecl(rank=2, dtype=(fp32, bf16), symbols=("G", "B")),
    },
    outputs={
        "q": TensorDecl(rank=2, dtype=("fp8",), symbols=("G", "B")),
        "scale": TensorDecl(rank=1, dtype=(fp32,), symbols=("G",)),
    },
    constraints=("B % 16 == 0",),
    preconditions=(
        "x is a GROUPED full-tile view [G, B] of weight blocks;",
        "B = block*block, e.g. 16384 for DeepSeek block=128 full tiles;",
        "caller reshapes [N,K] full tiles -> [ceil(N/block)*ceil(K/block), B] "
        "and outputs back to [N,K] / [ceil(N/block), ceil(K/block)];",
        "tail tiles require padding or the hand helper path;",
        "fp8 dtype is float8_e4m3fn (FP8_MAX=448); fnuz (240) is an arch override",
    ),
    numerics=Numerics(
        rtol=1e-2,
        atol=1e-2,
        reduce_dtype=fp32,
        cross_backend_rtol=1e-2,
        by_dtype={
            "fp32": {"rtol": 1e-6, "atol": 1e-6},
            "bf16": {"rtol": 1e-2, "atol": 1e-2},
        },
        notes=(
            "Per-block fp8 quant over flattened full block tiles. amax reduced "
            "in fp32; scale = amax/FP8_MAX (OCP e4m3fn, max 448). Bit-exact "
            "with ops/gemm/reference.py:per_block_quant_fp8 for full tiles after "
            "the same block flattening. Tail tiles are outside this base VKL op."
        ),
    ),
    shape_sweep="per_block_quant_fp8",
    fusions=(),
)
@launch(Launch.rowwise())  # one program per flattened block tile
@targets(triton=Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "memory-bound row-wise reduce over flattened block tiles; one kernel for "
        "amax + scale + fp8 cast (full-tile weight preparation path)."
    ),
))
def per_block_quant_fp8(ctx):
    """Build the math IR: per flattened block tile amax -> scale -> fp8 cast."""
    v = ctx.load("x").cast("fp32")
    amax = ctx.reduce_max(ctx.abs(v), axis=1, accum_dtype="fp32")
    amax_safe = ctx.maximum(amax, ctx.lit(_EPS))
    scale = amax_safe / ctx.lit(FP8_MAX)
    qf = v / scale
    qf = ctx.minimum(ctx.maximum(qf, ctx.lit(-FP8_MAX)), ctx.lit(FP8_MAX))
    ctx.store("q", qf.cast("fp8"))
    ctx.store("scale", scale)
