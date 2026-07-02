# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The frozen math IR — the correctness oracle (docs/brainstorm/10 §1).

These dataclasses express the *what* (the computation), never the *how* (the
schedule). They are FROZEN: an edit primitive may never produce or mutate a
math node (the ``gate`` enforces this — docs/brainstorm/10 §5, last row). The
CPU lowering of this algebra *is* the Op Spec's ``numerics.reference``
(``reference.py``), so a schedule edit literally cannot make the reference drift.

The algebra is deliberately tiny (~9 node kinds): pointwise / reduce / mma, plus
the data-ADDRESSING family (``Gather`` / ``Slice`` / ``Concat`` — added
2026-07-02, docs/brainstorm/06 A4 resolved). The addressing nodes are
oracle-safe: their torch lowering is bit-exact with their device lowering, and
the index is an *input* tensor (no data-dependent control flow). If an op needs
data-SELECTION (a gather whose index is a value computed in the kernel, a
sort, a top-k, an RNG), the math IR cannot express it → that op falls back to a
hand-written reference (the ``06`` A4 case-(c) line). The algebra's *limits are
the auto-reference's limits, made explicit.*
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Dtype short names reused across the substrate (registry/dtypes.py).
Dtype = str  # "fp32" | "bf16" | "fp16" | "fp8e4m3" | ...

# A dim may be a concrete int or a symbolic name bound at sweep time.
Dim = int | str


@dataclass(frozen=True)
class TensorRef:
    """A named tensor in the computation (an op input/output or an intermediate).

    Carries dtype + shape so every downstream check is decidable from the IR
    alone, with no running code (docs/brainstorm/10 §1).
    """

    name: str
    dtype: Dtype
    shape: tuple[Dim, ...]
    subscript: tuple[str, ...] = ()  # Einstein-ish dim labels, e.g. ("m", "k")


@dataclass(frozen=True)
class Load:
    """Read a tensor into the computation (the entry point of a dataflow graph)."""

    ref: TensorRef


@dataclass(frozen=True)
class Reduce:
    """A reduction: ``sum`` | ``max`` | ``rsqrt`` over one axis.

    ``accum_dtype`` MUST equal the Op Spec's ``numerics.reduce_dtype`` — the
    ``gate`` checks this (docs/brainstorm/10 §5, row 2). This is the rule that
    stops a bf16-accumulate schedule from silently drifting the reference.
    """

    x: TensorRef
    op: Literal["sum", "max", "rsqrt"]
    axis: int
    accum_dtype: Dtype
    out: TensorRef  # the reduced result, named for downstream nodes to reference


@dataclass(frozen=True)
class MMA:
    """A matrix multiply-accumulate: ``out += a @ b``.

    The only "heavy" op. ``accum_dtype`` MUST equal ``numerics.reduce_dtype``
    (checked). On a device this is what a ``MapTo(instruction=wgmma/mfma)``
    schedules; on CPU it lowers to ``torch.matmul`` in fp32.
    """

    a: TensorRef
    b: TensorRef
    accum_dtype: Dtype
    out: TensorRef


@dataclass(frozen=True)
class Pointwise:
    """A unary/binary pointwise fn (cast, scale, bias, activation, residual).

    Pure pointwise is numerically exact — tolerance unchanged — which is why
    fusion of pointwise chains is always safe (fuse-elementwise-chain skill).
    """

    fn: str  # "cast" | "scale" | "bias" | "mul" | "add" | "gelu" | "silu" | ...
    args: tuple[TensorRef, ...]
    out_dtype: Dtype
    out: TensorRef


@dataclass(frozen=True)
class Store:
    """Write a computed value to an output tensor (a dataflow graph sink)."""

    ref: TensorRef
    val: TensorRef


# ─── Addressing nodes (added 2026-07-02 — docs/brainstorm/06 A4 resolved) ───────
# These resolve the A4 "sparse / indexed ops" open question for the
# data-ADDRESSING case (case (a)). Each is oracle-safe: it is pure, parallel, and
# deterministic, and its torch lowering IS bit-exact with its device lowering:
#
#   Gather :  torch ``base[index]``  ≡  triton ``tl.load(base + index*stride)``
#   Slice  :  torch ``base[..., a:b]`` ≡  triton ``tl.load(base + (a+arange)*s)``
#   Concat :  torch ``torch.cat([a,b])`` ≡  triton two loads into one register tile
#
# The INDEX is an input tensor (e.g. RoPE ``positions``), NOT a value computed
# inside the kernel — so there is no data-dependent control flow, only
# data-dependent addressing. Data-SELECTION (top-k/sort/RNG) stays hand-path
# (case (c)): it cannot be an obviously-correct pure DAG, so admitting it would
# reopen the drift gap the math IR exists to close (06 A4).


@dataclass(frozen=True)
class Gather:
    """``out = base[index]`` along ``axis`` (data-ADDRESSING, oracle-safe).

    ``index`` is a tensor (an input or intermediate); ``base`` is gathered along
    ``axis``. RoPE's cos/sin-cache lookup (``cs = cos_sin_cache[positions]``) is
    the canonical use. The index is data, but the gather is fully parallel and
    deterministic — no control flow branches on a device value — so the torch
    lowering (advanced indexing) is bit-exact with the triton lowering
    (``tl.load(base + index*stride)``). Case (a) of the A4 scope line.
    """

    base: TensorRef
    index: TensorRef
    axis: int
    out: TensorRef


@dataclass(frozen=True)
class Slice:
    """``out = base[..., start:stop]`` along ``axis`` (a static-range sub-view).

    Oracle-safe addressing: ``start``/``stop`` are known at TRACE time, so this
    is a pure sub-range load — never a data-dependent slice. Bounds may be a
    python ``int`` OR a ``str`` expression over ``shape`` (the operand's
    concrete axis size, resolved at eval time) — e.g. RoPE halves the head axis
    with ``Slice(axis=-1, 0, "shape//2")`` / ``("shape//2", "shape")``, since
    ``head_size`` is symbolic (``D``) at trace time.
    """

    base: TensorRef
    axis: int
    start: int | str
    stop: int | str
    out: TensorRef


@dataclass(frozen=True)
class Concat:
    """``out = cat([a, b], axis)`` (reassemble a tensor from two parts).

    Oracle-safe addressing: a pure register-tile reassembly. RoPE's
    ``rotate_half``-equivalent writes the two rotation halves to the output via
    a ``Concat`` along the head axis.
    """

    a: TensorRef
    b: TensorRef
    axis: int
    out: TensorRef


@dataclass(frozen=True)
class Unsqueeze:
    """Insert a size-1 axis (a pure view; lets a per-row quantity broadcast).

    Oracle-safe addressing: ``torch.unsqueeze``. RoPE's cos/sin are per-token
    ``[T, D//2]`` but must multiply a per-(token,head) ``[T, H, D//2]`` tile — an
    ``Unsqueeze(axis=1)`` makes them ``[T, 1, D//2]`` so the pointwise mul
    broadcasts. This is data-*shaping*, not data-dependent control.
    """

    base: TensorRef
    axis: int
    out: TensorRef


# The union of math nodes. A math IR is a sequence of these (a dataflow DAG
# expressed as an ordered list; downstream nodes reference upstream ``out`` names).
MathNode = Load | Reduce | MMA | Pointwise | Store | Gather | Slice | Concat | Unsqueeze


@dataclass(frozen=True)
class MathIR:
    """A frozen computation: the ordered math nodes + their tensor declarations."""

    nodes: tuple[MathNode, ...]
    tensors: dict[str, TensorRef] = field(default_factory=dict)

    def output_names(self) -> tuple[str, ...]:
        """Names of tensors written by a Store (the op's outputs)."""
        return tuple(n.ref.name for n in self.nodes if isinstance(n, Store))

    def reduce_dtype(self) -> Dtype | None:
        """The accumulation dtype declared by any Reduce/MMA node, if present."""
        for n in self.nodes:
            if isinstance(n, (Reduce, MMA)):
                return n.accum_dtype
        return None
