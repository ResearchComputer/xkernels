# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored ``dual_rmsnorm`` — the worked example (docs/brainstorm/04 Ex.1).

One source spells the WHOLE contract that today lives across eight hand-written
artifacts in four paradigms (spec + reference + sweep + card + Triton kernel +
input-gen + registration). The header projects 1:1 to the Op Spec; the body
builds the doc-10 **math IR** (``MMA``/``Reduce``/``Pointwise``) that lowers to
BOTH torch (the auto-reference) AND Triton (the device kernel) — one
computation, two lowerings, structurally guaranteed to agree
(docs/brainstorm/02 §1).

Phase 2.0b moved this example onto the math IR (it was the Phase 1.5 seed for
the bespoke row-reduce IR, now retired). The body spells the arithmetic in the
math builder (``ctx.load``, ``ctx.reduce_sum``, arithmetic via overloaded ``T``).
Its cast order mirrors the hand REFERENCE (``(x*inv).to(out_dtype) * w``), so the
torch evaluator is bit-exact with ``dual_rmsnorm_ref``; the generated Triton
kernel matches within tolerance. The fp32 reduction (``load -> fp32 -> sum``) is
the load-bearing numerical invariant, identical to both the hand kernel and the
hand reference.

Decorator order matters (applied bottom-up): ``@targets`` and ``@launch`` attach
metadata to the body; ``@kernel`` reads it and builds the ``KernelSpec``. The
body takes only ``ctx`` (inputs are referenced BY NAME — the IR is symbolic in
shape/dtype).
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

__all__ = ["dual_rmsnorm"]

_EPS = 1e-6


@kernel(
    id="dual_rmsnorm@1.0.0",
    kernel="dual_rmsnorm",
    canonical_op="norm",
    name="fused parallel dual RMSNorm",
    signature="(rmsnorm(x1,w1), rmsnorm(x2,w2)) in a single launch",
    inputs={
        "x1": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("T", "d1")),
        "w1": TensorDecl(rank=1, dtype=(fp32, "bf16"), symbols=("d1",)),
        "x2": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("T", "d2")),
        "w2": TensorDecl(rank=1, dtype=(fp32, "bf16"), symbols=("d2",)),
    },
    outputs={
        "out1": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("T", "d1"), reduces_over="x1"),
        "out2": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("T", "d2"), reduces_over="x2"),
    },
    constraints=(
        "dtype(x1) == dtype(w1)",
        "dtype(x2) == dtype(w2)",
    ),
    preconditions=("x1 and x2 share the same leading dim T", "inputs are contiguous"),
    numerics=Numerics(
        rtol=1.6e-2,
        atol=1e-2,
        reduce_dtype=fp32,
        cross_backend_rtol=1.6e-2,
        by_dtype={
            "fp32": {"rtol": 1e-5, "atol": 1e-6},
            "bf16": {"rtol": 1.6e-2, "atol": 1e-2},
        },
        notes="Variance reduced in fp32 (rsqrt of fp32 mean-of-squares).",
    ),
    shape_sweep="dual_rmsnorm",
    fusions=("parallel_pair",),
)
@launch(Launch.rowwise())  # one Triton program per token row T; reduces over the last axis
@targets(triton=Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",  # row-wise reduce; no smem/lds, just registers
    regime="memory-bound row-wise reduce; one kernel/one pass for both latents.",
))
def dual_rmsnorm(ctx):
    """Build the math IR for both latents (the body IS the computation).

    ``ctx`` is the build-mode math-IR builder (``MathBodyCtx``); inputs/outputs
    are referenced BY NAME. The IR built here is lowered to torch (reference)
    and Triton (device) — the SAME nodes, two interpreters.
    """

    def rmsnorm(x: str, w: str, out: str):
        v = ctx.load(x).cast("fp32")            # fp32 reduction (load-bearing)
        ss = ctx.reduce_sum(v * v, axis=1, accum_dtype="fp32")   # sum of squares
        mean = ss / ctx.dim(x, axis=1)          # / d  (the reduction width)
        inv = ctx.rsqrt(mean + ctx.lit(_EPS))
        xn = (v * inv).cast(ctx.out_dtype())    # cast to output dtype BEFORE * w (matches ref)
        ctx.store(out, xn * ctx.load(w))        # weight in its native (= output) dtype

    rmsnorm("x1", "w1", "out1")
    rmsnorm("x2", "w2", "out2")
