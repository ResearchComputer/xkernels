# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored gated activations — ``silu_and_mul`` / ``gelu_and_mul`` (issue #67).

mini-sglang's FFN block calls ``flashinfer.silu_and_mul`` / ``gelu_and_mul``
(no ROCm wheel). xkernels fuses SwiGLU *inside* ``fused_ffn`` but never exposed
the bare gated-activation op. These two ops fill that gap.

Mathematically a gated activation is ``act(gate) * up`` — pure pointwise, no
reduction — so it does not fit ``rowwise`` (which tiles a Reduce axis) or
``tiled_2d`` (which needs an MMA). It is the canonical user of the
``Launch.elementwise()`` pattern added alongside this op: one program per flat
tile of the output, every node a Load/Pointwise/Store over the same grid.

The flashinfer/vLLM signature packs gate+up into ONE ``[..., 2K]`` tensor; the
*contract* here is the mathematically-honest TWO-input form ``act(gate[M,K]) *
up[M,K]`` (gate and up are independent tensors — the packing is a caller
convention). A consumer with the packed buffer does ``gate, up = x.chunk(2,
dim=-1)`` then calls this op. Keeping two inputs avoids a static-slice math node
and keeps the IR at its designed ~6 kinds (the slice would be a 7th).

gelu uses the tanh approximation (``0.5x(1+tanh(...))``) — the form flashinfer /
vLLM's ``gelu_and_mul`` use (and what the GELU(tanh) issue specifies). silu is
``x * sigmoid(x)``. Both compute the nonlinearity in the operand's promoted
precision (torch.sigmoid upcasts bf16→fp32 internally) then cast back to the
output dtype — more accurate than a pure-bf16 silu, and the body IS the
auto-reference so this precision choice is the contract.
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

__all__ = ["silu_and_mul", "gelu_and_mul"]


def _gated_numerics() -> Numerics:
    # Pure pointwise: the only divergence vs an fp32 oracle is the final
    # fp32->bf16/fp16 cast of the activation. bf16/fp16 rtol 1.6e-2 covers it
    # (same bar as the norms); fp32 is near-exact.
    return Numerics(
        rtol=1.6e-2,
        atol=1e-2,
        reduce_dtype=fp32,  # activations evaluated in promoted (fp32) precision
        cross_backend_rtol=1.6e-2,
        by_dtype={
            "fp32": {"rtol": 1e-5, "atol": 1e-6},
            "bf16": {"rtol": 1.6e-2, "atol": 1e-2},
            "fp16": {"rtol": 1.6e-2, "atol": 1e-2},
        },
        notes=(
            "Pure pointwise gated activation. Nonlinearity in promoted (fp32) "
            "precision, cast to out dtype; only divergence is that final cast."
        ),
    )


_TARGET = Target(
    backend="triton",
    arch="any",
    roofline="memory_bound",
    scratch_kind="registers",  # flat-1D pointwise; no smem/lds
    regime=(
        "memory-bound elementwise; one flat-1D kernel. Portable Triton runs on "
        "amd_cdna3 (gfx942); vectorized loads + a wider BLOCK are the tune lever."
    ),
    knobs={"BLOCK": (1024, 2048, 4096)},
)


@kernel(
    id="silu_and_mul@1.0.0",
    kernel="silu_and_mul",
    canonical_op="activation",
    name="SiLU-gated multiply (SwiGLU activation)",
    signature="out[M,K] = silu(gate[M,K]) * up[M,K]",
    inputs={
        "gate": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
        "up":   TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
    },
    outputs={
        "out": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
    },
    constraints=("dtype(gate) == dtype(up)",),
    preconditions=(
        "gate, up share shape [M,K] and are contiguous row-major",
        "for the flashinfer one-tensor [M,2K] form: gate, up = x.chunk(2, dim=-1)",
    ),
    numerics=_gated_numerics(),
    shape_sweep="silu_and_mul",
    fusions=(),
)
@launch(Launch.elementwise())
@targets(triton=_TARGET)
def silu_and_mul(ctx):
    """``out = silu(gate) * up``  where  ``silu(x) = x * sigmoid(x)``.

    The nonlinearity is evaluated in fp32 (gate upcast) then cast to the output
    dtype — the flashinfer/vLLM convention and the honest spelling given the math
    IR's dtype-representative (``dtype[0]``) decl.
    """
    g = ctx.load("gate").cast("fp32")
    u = ctx.load("up")
    ctx.store("out", (ctx.silu(g) * u).cast(ctx.out_dtype()))


@kernel(
    id="gelu_and_mul@1.0.0",
    kernel="gelu_and_mul",
    canonical_op="activation",
    name="GELU(tanh)-gated multiply",
    signature="out[M,K] = gelu_tanh(gate[M,K]) * up[M,K]",
    inputs={
        "gate": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
        "up":   TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
    },
    outputs={
        "out": TensorDecl(rank=2, dtype=(fp32, bf16, fp16), symbols=("M", "K")),
    },
    constraints=("dtype(gate) == dtype(up)",),
    preconditions=(
        "gate, up share shape [M,K] and are contiguous row-major",
        "gelu uses the tanh approximation (flashinfer/vLLM gelu_and_mul form)",
    ),
    numerics=_gated_numerics(),
    shape_sweep="gelu_and_mul",
    fusions=(),
)
@launch(Launch.elementwise())
@targets(triton=_TARGET)
def gelu_and_mul(ctx):
    """``out = gelu_tanh(gate) * up``  (tanh-approx GELU, evaluated in fp32)."""
    g = ctx.load("gate").cast("fp32")
    u = ctx.load("up")
    ctx.store("out", (ctx.gelu(g) * u).cast(ctx.out_dtype()))
