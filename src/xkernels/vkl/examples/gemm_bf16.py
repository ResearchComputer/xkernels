# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored ``gemm_bf16`` — the Phase 2.0 go/no-no worked example.

This is the second op lowered through the DSL and the FIRST that the Phase 1.5
row-reduce IR cannot express: a 2D-tiled GEMM needs an MMA node, a K-loop, and a
2D output grid — none of which the row-reduce expr set has. So this op drives the
**math-IR convergence** (docs/brainstorm/11 §11): the body builds the doc-10 math
IR (``MMA``/``Pointwise``), and two interpreters lower it — torch ``matmul`` (the
auto-reference) and Triton's tiled ``tl.dot`` K-loop (the device kernel, the
``08`` §3 / hand ``mm_fp8_blockscale_kernel.py`` idiom).

Bare GEMM, no fusion, no dequant — the cleanest possible stress of the 2D-tile +
K-loop + MMA lowering. The arithmetic: ``out = (a @ b)`` accumulated in fp32, cast
to the output dtype. The fp32 accumulation is the load-bearing numerical
invariant (matches the hand reference ``a.float() @ b.float()``).

The tiling comes from the launch pattern (``Launch.tiled_2d()``) + the math IR's
``subscript`` (Einstein labels ``a[m,k]``, ``b[k,n]``, ``out[m,n]``): the output
dims (m, n) are the 2D grid; the contracted dim (k) is the K-loop. No wave size,
no instruction, no L5 shape is named — the body stays portable (above L3).
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
from ..tiles import fp32

__all__ = ["gemm_bf16"]


@kernel(
    id="gemm_bf16@1.0.0",
    kernel="gemm_bf16",
    canonical_op="gemm",
    name="dense bf16 GEMM (fp32 accumulate)",
    signature="out[M,N] = (a[M,K] @ b[K,N]) accumulated in fp32, cast to out dtype",
    inputs={
        "a": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("M", "K")),
        "b": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("K", "N")),
    },
    outputs={
        "out": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("M", "N")),
    },
    constraints=(
        "M % 16 == 0",
        "N % 16 == 0",
        "K % 16 == 0",
        "dtype(a) == dtype(b)",
    ),
    preconditions=("a, b contiguous row-major", "out dtype follows the point dtype"),
    numerics=Numerics(
        rtol=2e-2,
        atol=1e-1,
        reduce_dtype=fp32,
        cross_backend_rtol=2e-2,
        by_dtype={
            "fp32": {"rtol": 1e-4, "atol": 1e-3},
            "bf16": {"rtol": 2e-2, "atol": 1e-1},
        },
        notes=(
            "Dense GEMM accumulated in fp32 (out = a.float() @ b.float(), cast). "
            "bf16 rtol 2e-2 covers the final fp32->bf16 cast (~1-2 ULP at the "
            "output magnitude, which scales with K) plus accumulation-order "
            "differences; calibrated empirically, see docs/brainstorm/11 §11."
        ),
    ),
    shape_sweep="gemm_bf16",
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
        # The autotune search space (docs/brainstorm/10 §5; the
        # autotune-knob-sweep skill enumerates this). Tile knobs recompile the
        # tl.constexpr tiles; num_warps/num_stages are Triton launch metas
        # (software-pipeline depth hides global-memory latency — the
        # correct-but-slow -> ceiling-reaching lever, Phase 2.2).
        knobs={
            "BLOCK_M": (64, 128, 256),
            "BLOCK_N": (64, 128, 256),
            "BLOCK_K": (32, 64),
            "num_warps": (4, 8),
            "num_stages": (2, 3, 4),
        },
    )
)
def gemm_bf16(ctx):
    """Build the math IR for a bare GEMM (the body IS the computation).

    ``ctx`` is the build-mode math-IR builder; inputs/outputs are referenced BY
    NAME. The IR built here lowers to torch ``matmul`` (reference) and Triton's
    tiled ``tl.dot`` (device). fp32 accumulation is explicit (``accum_dtype``).
    """
    a = ctx.load("a")  # [M, K]
    b = ctx.load("b")  # [K, N]
    acc = ctx.mma(a, b, accum_dtype=fp32)  # [M, N] fp32  (the heavy op)
    out = acc.cast(ctx.out_dtype())  # [M, N] out dtype
    ctx.store("out", out)


# Phase 2.1: the native CUDA override for sm_121 (GB10 Grace-Blackwell). The
# body builds the SAME math IR (the oracle property — checked by
# ``check_override_math_ir``), but lowers to a native nvcc-compiled kernel
# (``vkl.lower.cuda``). Its load-bearing value TODAY is mechanism validation: it
# proves a per-target override compiles to a real native kernel, registers as the
# ``cuda`` backend, and passes ``verify`` against the exact oracle on real
# hardware. (The earlier "triton degrades fp32 to tf32" framing was a
# misdiagnosis of an oracle-side tf32 bug, fixed in ``run_reference``; the
# triton backend does true fp32 on this arch. So this override is not a
# correctness fix — it is the live pipeline that the CUTLASS/wgmma ceiling work
# lands on top of.) Performance is correct-but-slow (CUDA-core FMA); the bf16
# tensor-core ceiling is the CUTLASS follow-up.
@gemm_bf16.target("cuda", arch="nvidia_sm121")
def gemm_bf16_cuda(ctx):
    """Same math IR as the portable body; lowered to native CUDA by lower/cuda.py."""
    a = ctx.load("a")
    b = ctx.load("b")
    acc = ctx.mma(a, b, accum_dtype=fp32)
    ctx.store("out", acc.cast(ctx.out_dtype()))
