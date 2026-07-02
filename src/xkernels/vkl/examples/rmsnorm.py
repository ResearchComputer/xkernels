# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored ``rmsnorm`` — plain single-tensor RMSNorm (issue #66).

mini-sglang's ``RMSNorm`` (porting to AMD ROCm) wants a plain single-tensor
RMSNorm, flashinfer-compatible: ``rmsnorm(x, weight, eps)``. xkernels had
``dual_rmsnorm`` (the MLA paired-latent variant) but no plain single op, so the
ROCm path fell back to a torch reference. This op fills that gap.

It is the cleanest possible DSL candidate: the math is a fixed DAG of
load / cast-to-fp32 / mul / reduce_sum / div / lit / rsqrt / cast / mul /
store — every node is in the doc-10 math IR's pointwise+reduce set. It is
literally ``dual_rmsnorm`` with one branch instead of two (one input tensor, one
weight, one output), so the Phase 2.0b ``rowwise`` lowering covers it unchanged.

The body's cast order mirrors the hand RMSNorm reference (and dual_rmsnorm):
variance reduced in fp32, then ``(x * inv_rms).to(out_dtype) * w`` — the fp32
reduction is the load-bearing numerical invariant. eps is baked at 1e-6 (the
Llama / DeepSeek default; matches dual_rmsnorm) because the math IR carries it
as a literal, not a runtime scalar.

Decorator order is bottom-up (applied by Python innermost-first):
``@targets`` / ``@launch`` attach metadata the topmost ``@kernel`` reads.
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
from ..tiles import bf16, fp16, fp32

__all__ = ["rmsnorm"]

_EPS = 1e-6  # Llama / DeepSeek RMSNorm epsilon (matches dual_rmsnorm)


@kernel(
    id="rmsnorm@1.0.0",
    kernel="rmsnorm",
    canonical_op="norm",
    name="plain single-tensor RMSNorm",
    signature="out[T,d] = (x[T,d] * rsqrt(mean(x^2)[T] + eps)) * w[d], fp32 reduce",
    inputs={
        "x": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("T", "d")),
        "w": TensorDecl(rank=1, dtype=(fp32, bf16, fp16), symbols=("d",)),
    },
    outputs={
        "out": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("T", "d"), reduces_over="x"),
    },
    constraints=(
        "dtype(x) == dtype(w)",
    ),
    preconditions=("x is contiguous row-major", "eps = 1e-6 baked into the kernel"),
    numerics=Numerics(
        rtol=1.6e-2,
        atol=1e-2,
        reduce_dtype=fp32,
        cross_backend_rtol=1.6e-2,
        by_dtype={
            "fp32": {"rtol": 1e-5, "atol": 1e-6},
            "bf16": {"rtol": 1.6e-2, "atol": 1e-2},
            "fp16": {"rtol": 1.6e-2, "atol": 1e-2},
        },
        notes=(
            "Variance reduced in fp32 (rsqrt of fp32 mean-of-squares); identical "
            "numerics to dual_rmsnorm's per-latent branch. eps=1e-6 baked "
            "(Llama/DeepSeek default)."
        ),
    ),
    shape_sweep="rmsnorm",
    fusions=(),
)
@launch(Launch.rowwise())  # one Triton program per token row T; reduce over the last axis
@targets(triton=Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",  # row-wise reduce; no smem/lds, just registers
    regime=(
        "memory-bound row-wise reduce; one kernel/one pass. Portable Triton "
        "runs on amd_cdna3 (gfx942); a native LDS/MFMA-tuned override for the "
        "MI300X ceiling is the GPU-gated tune-for-cdna follow-up (Phase 2.1)."
    ),
))
def rmsnorm(ctx):
    """Build the math IR for plain RMSNorm (the body IS the computation).

    ``ctx`` is the build-mode math-IR builder (``MathBodyCtx``); inputs/outputs
    are referenced BY NAME. The IR built here lowers to torch (reference) and
    Triton (device) — the SAME nodes, two interpreters. Identical to one branch
    of ``dual_rmsnorm``.
    """
    v = ctx.load("x").cast("fp32")                         # fp32 reduction (load-bearing)
    ss = ctx.reduce_sum(v * v, axis=1, accum_dtype="fp32")  # sum of squares over d
    mean = ss / ctx.dim("x", axis=1)                       # / d  (the reduction width)
    inv = ctx.rsqrt(mean + ctx.lit(_EPS))                  # inv-RMS in fp32
    # Cast to out dtype before multiplying by w to match the reference.
    xn = (v * inv).cast(ctx.out_dtype())
    ctx.store("out", xn * ctx.load("w"))                   # weight in its native (= output) dtype
