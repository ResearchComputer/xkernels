# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored ``gemm_fp16`` — fp16 dense GEMM with fp32 accumulation.

This is the fp16 twin of ``gemm_bf16`` (the Phase 2.0 tiled_2d worked example):
the SAME math IR (``load`` -> ``MMA`` in fp32 -> ``cast`` to the output dtype ->
``store``) and the SAME 2D-tiled launch with a K-loop. Only the dtype tuple and
the numerics bar move. fp32 accumulation is the load-bearing numerical invariant
(``out = a.float() @ b.float()``, cast to fp16), so the auto-reference is
bit-exact with the device kernel's arithmetic.

Bare GEMM, no fusion, no dequant — the cleanest possible stress of the 2D-tile +
K-loop + MMA lowering in fp16. The tiling comes from the launch pattern
(``Launch.tiled_2d()``) + the math IR's subscript (Einstein labels ``a[m,k]``,
``b[k,n]``, ``out[m,n]``): the output dims (m, n) are the 2D grid; the contracted
dim (k) is the K-loop. No wave size, no instruction, no L5 shape is named — the
body stays portable (above L3).

Lives in-package at ``xkernels.vkl.examples.gemm_fp16`` (re-exported as
``examples.gemm_fp16``), so the ``@kernel`` body self-registers as the
auto-reference (``xkernels.vkl.auto:gemm_fp16``) on import and the registry
drift gate / ``reference_callable`` resolve from a fresh process.
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
from ..tiles import fp16, fp32

__all__ = ["gemm_fp16"]


@kernel(
    id="gemm_fp16@1.0.0",
    kernel="gemm_fp16",
    canonical_op="gemm",
    name="dense fp16 GEMM (fp32 accumulate)",
    signature="out[M,N] = (a[M,K] @ b[K,N]) accumulated in fp32, cast to out dtype",
    inputs={
        "a": TensorDecl(rank=2, dtype=(fp32, fp16), symbols=("M", "K")),
        "b": TensorDecl(rank=2, dtype=(fp32, fp16), symbols=("K", "N")),
    },
    outputs={
        "out": TensorDecl(rank=2, dtype=(fp32, fp16), symbols=("M", "N")),
    },
    constraints=(
        "M % 16 == 0",
        "N % 16 == 0",
        "K % 16 == 0",
        "dtype(a) == dtype(b)",
    ),
    preconditions=(
        "a, b contiguous row-major",
        "out dtype follows the point dtype",
    ),
    numerics=Numerics(
        rtol=1e-2,
        atol=1e-1,
        reduce_dtype=fp32,
        cross_backend_rtol=1e-2,
        by_dtype={
            "fp32": {"rtol": 1e-4, "atol": 1e-3},
            "fp16": {"rtol": 1e-2, "atol": 1e-1},
        },
        notes=(
            "Dense GEMM accumulated in fp32 (out = a.float() @ b.float(), cast). "
            "fp16 carries 10 mantissa bits (~3x bf16's precision), so the default "
            "rtol is tightened to 1e-2 vs the bf16 twin's 2e-2; atol stays 1e-1 to "
            "absorb the output-magnitude scaling with K. Covers the final fp32->fp16 "
            "cast plus fp32 accumulation-order differences; recalibrate empirically "
            "on the target arch (cf. gemm_bf16, docs/brainstorm/11 §11)."
        ),
    ),
    shape_sweep="gemm_fp16",
    fusions=(),
)
@launch(Launch.tiled_2d())  # 2D grid over (M, N); the MMA's K dim becomes the K-loop
@targets(
    triton=Target(
        backend="triton",
        arch="any",
        roofline="compute_bound",
        scratch_kind="smem",  # K-loop tiles stream through smem/lds (by target)
        regime="compute-bound dense GEMM; tl.dot on the matrix engine.",
        # The autotune search space (docs/brainstorm/10 §5). Tile knobs recompile
        # the tl.constexpr tiles; num_warps/num_stages are Triton launch metas
        # (software-pipeline depth hides global-memory latency).
        knobs={
            "BLOCK_M": (64, 128, 256),
            "BLOCK_N": (64, 128, 256),
            "BLOCK_K": (32, 64),
            "num_warps": (4, 8),
            "num_stages": (2, 3, 4),
        },
    )
)
def gemm_fp16(ctx):
    """Build the math IR for a bare fp16 GEMM (the body IS the computation).

    ``ctx`` is the build-mode math-IR builder; inputs/outputs are referenced BY
    NAME. The IR built here lowers to torch ``matmul`` (reference) and Triton's
    tiled ``tl.dot`` (device). fp32 accumulation is explicit (``accum_dtype``).
    """
    a = ctx.load("a")  # [M, K]
    b = ctx.load("b")  # [K, N]
    acc = ctx.mma(a, b, accum_dtype=fp32)  # [M, N] fp32  (the heavy op)
    out = acc.cast(ctx.out_dtype())  # [M, N] out dtype
    ctx.store("out", out)
