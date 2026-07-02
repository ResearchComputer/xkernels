# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored temperature softmax, the deterministic slice of sampling (#69/#70).

The open mini-sglang sampling/gating issues need more than softmax: top-k/top-p
selection and multinomial sampling are data-selection/RNG operations and remain
hand-path by design. The stable row-wise softmax prefix is still DSL-expressible:

``scaled = logits / temperatures[:, None]``
``m = reduce_max(scaled)``
``ex = exp(scaled - m)``
``probs = ex / reduce_sum(ex)``

This example exists to pin two VKL capabilities needed by those issue families:
multiple row-wise reductions in one body, and a rank-1 per-row input broadcast
(``temperatures[B]``) across the reduced vocab axis.
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

__all__ = ["temperature_softmax"]


@kernel(
    id="temperature_softmax@1.0.0",
    kernel="temperature_softmax",
    canonical_op="reduce",
    name="temperature-scaled row-wise softmax",
    signature="probs[B,V] fp32 = softmax(logits[B,V] / temperatures[B])",
    inputs={
        "logits": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("B", "V")),
        "temperatures": TensorDecl(rank=1, dtype=(fp32,), symbols=("B",)),
    },
    outputs={
        "probs": TensorDecl(rank=2, dtype=(fp32,), symbols=("B", "V")),
    },
    constraints=(),
    preconditions=(
        "temperatures are positive and finite",
        "top-k/top-p selection and RNG sampling are intentionally outside this DSL op",
    ),
    numerics=Numerics(
        rtol=6e-3,
        atol=1e-3,
        reduce_dtype=fp32,
        cross_backend_rtol=6e-3,
        by_dtype={
            "fp32": {"rtol": 6e-3, "atol": 1e-3},
            "bf16": {"rtol": 6e-3, "atol": 1e-3},
            "fp16": {"rtol": 6e-3, "atol": 1e-3},
        },
        notes=(
            "Stable row-wise softmax prefix for sampling/top-k gating. Logits and "
            "temperature divide promote to fp32; max and sum reductions are fp32. "
            "Tolerance covers Triton tl.exp approximation vs torch.exp on GB10 "
            "(observed max_abs < 8e-4, max_rel < 5.2e-3). Value-dependent "
            "top-k/top-p and RNG sampling stay hand-path."
        ),
    ),
    shape_sweep="temperature_softmax",
)
@launch(Launch.rowwise())
@targets(triton=Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",
    regime=(
        "row-wise stable softmax over vocab/expert dimension; one program per row, "
        "with per-row temperature scalar broadcast across the reduction tile."
    ),
    knobs={"num_warps": (4, 8)},
))
def temperature_softmax(ctx):
    logits = ctx.load("logits").cast("fp32")
    temp = ctx.unsqueeze(ctx.load("temperatures").cast("fp32"), axis=1)
    scaled = logits / temp
    row_max = ctx.reduce_max(scaled, axis=1, accum_dtype="fp32")
    ex = ctx.exp(scaled - row_max)
    denom = ctx.reduce_sum(ex, axis=1, accum_dtype="fp32")
    ctx.store("probs", ex / denom)
