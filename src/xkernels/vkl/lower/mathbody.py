# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Math-IR body lowering: tile-DSL body -> math IR -> {torch reference, Triton}.

This is the Phase 2.0 convergence module (docs/brainstorm/11 §11). The Phase 1.5
``rowreduce.py`` built a row-reduce-specific expr IR; that IR cannot express a
2D-tiled GEMM (no 2D tiles, no K-loop, no MMA). This module builds the doc-10
**math IR** (``ir/math.py``: ``MMA``/``Reduce``/``Pointwise``/``Load``/``Store``)
— the frozen correctness oracle — and lowers it two ways:

  * ``_TorchEval``: walks the math nodes; ``MMA`` -> ``torch.matmul`` in fp32 (the
    auto-reference, identical to a hand ``a.float() @ b.float()``).
  * ``_TritonGen``: walks the math nodes; ``MMA`` -> the tiled ``tl.dot`` K-loop
    (docs/brainstorm/08 §3; the hand ``mm_fp8_blockscale_kernel.py`` idiom).

**The tiling comes from the launch pattern + the math node's ``subscript``.**
A ``Launch.tiled_2d()`` body's ``MMA(a[m,k], b[k,n]) -> out[m,n]`` lowers as a 2D
grid over (m, n) with a K-loop over the contracted ``k`` dim — the Einstein-ish
labeling the math IR was designed for (docs/brainstorm/10 §1). A ``Launch.rowwise()``
body's ``Reduce`` over the last axis lowers as one program per leading-dim row,
with the reduced dim as the program-local 1D tile (the dual_rmsnorm shape).

Phase 2.0a shipped ``tiled_2d`` (bare GEMM). Phase 2.0b adds ``rowwise`` and
**retires ``rowreduce.py``** — one body-IR, two interpreters (torch + Triton),
both launch patterns lowered from the SAME math nodes. The torch evaluator is
launch-agnostic (vectorized torch needs no grid).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...registry.dtypes import to_torch_dtype
from ..ir.math import (
    MMA,
    Concat,
    Gather,
    Load,
    MathIR,
    MathNode,
    Pointwise,
    Reduce,
    Slice,
    Store,
    TensorRef,
    Unsqueeze,
)

# ═══════════════════════════════════════════════════════════════════════════════
# §1  T — a handle over a math-IR tensor name, with operator overloads
# ═══════════════════════════════════════════════════════════════════════════════


class T:
    """A thin handle over a math-IR tensor NAME so the body spells arithmetic.

    ``a * b`` builds a ``Pointwise(mul)`` node; ``a.cast('bf16')`` builds a
    ``Pointwise(cast)``. Each returns a fresh ``T`` wrapping the new node's
    output name. The body reads like the math while constructing the IR.
    """

    __slots__ = ("ctx", "name")

    def __init__(self, ctx: MathBodyCtx, name: str):
        self.ctx = ctx
        self.name = name

    def __mul__(self, other: T) -> T:
        return self.ctx._pointwise("mul", (self, other))

    def __add__(self, other: T) -> T:
        return self.ctx._pointwise("add", (self, other))

    def __sub__(self, other: T) -> T:
        return self.ctx._pointwise("sub", (self, other))

    def __truediv__(self, other: T) -> T:
        return self.ctx._pointwise("div", (self, other))

    def __neg__(self) -> T:
        return self.ctx._unary("neg", self)

    def cast(self, dtype: str) -> T:
        return self.ctx._cast(self, dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  MathBodyCtx — the build-mode ctx the body receives
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MathBody:
    """The built artifact: the math IR + the input/output declarations.

    ``in_decls``/``out_decls`` map name -> ``TensorRef`` (carrying the declared
    dtype tuple + symbolic shape + subscript). The lowering binds concrete
    shapes/dtype at eval/launch time.
    """

    ir: MathIR
    in_decls: dict[str, TensorRef]
    out_decls: dict[str, TensorRef]


class MathBodyCtx:
    """Build-mode ctx. Body calls append ordered math nodes (a dataflow DAG).

    Inputs/outputs are referenced BY NAME. Each builder method creates a fresh
    intermediate ``TensorRef`` (``_t0``, ``_t1``, ...) so the IR carries dtype +
    shape + subscript for every value — the torch evaluator and the Triton
    codegen are decidable from the IR alone, with no running code.
    """

    def __init__(self, in_decls: dict[str, TensorRef], out_decls: dict[str, TensorRef]):
        self.in_decls = in_decls
        self.out_decls = out_decls
        self._nodes: list[MathNode] = []
        self._tensors: dict[str, TensorRef] = {**in_decls, **out_decls}
        self._loaded: set[str] = set()
        self._n = 0

    # --- handle constructors -------------------------------------------------
    def load(self, name: str) -> T:
        """Reference an input/output tensor by name (returns a handle).

        Emits a ``Load`` node marking the dataflow entry point (the torch
        evaluator populates the value from the input; the codegen emits the
        tiled load). Cached per name so a body that reads ``a`` twice emits once.
        """
        if name not in self._tensors:
            raise KeyError(f"load({name!r}): not a declared input/output")
        if name not in self._loaded:
            self._nodes.append(Load(self._tensors[name]))
            self._loaded.add(name)
        return T(self, name)

    def lit(self, value: int | float) -> T:
        """A literal scalar, broadcast against any tensor (e.g. ``+ eps``).

        Lowered as a Python-side scalar baked into the trace; the torch evaluator
        returns the raw value, the codegen emits ``repr(value)``.
        """
        return T(self, self._lit_name(value))

    def out_dtype(self) -> str:
        """The kernel's output dtype (resolved from the first input at launch)."""
        return "__OUT_DTYPE__"  # sentinel; bound by the evaluator/codegen

    def dim(self, tensor: str, axis: int) -> T:
        """A symbolic scalar = ``inputs[tensor].shape[axis]`` (e.g. the reduce width).

        Used in pointwise arithmetic over the reduction (``ss / dim(x, axis=1)``
        for a mean). Resolves to the concrete shape value in the torch evaluator;
        the rowwise codegen lowers it to the runtime dim arg for its symbol.
        """
        return T(self, self._dim_name(tensor, axis))

    # --- math-node builders --------------------------------------------------
    def mma(self, a: T, b: T, accum_dtype: str) -> T:
        """``out += a @ b`` (matrix multiply-accumulate). The only heavy op.

        ``accum_dtype`` MUST equal the op's ``numerics.reduce_dtype`` (the gate
        checks this). On device this is what a ``MapTo(instruction=wgmma/mfma)``
        schedules; on CPU it lowers to ``torch.matmul`` in ``accum_dtype``.
        """
        return T(
            self,
            self._emit(
                MMA(self._ref(a), self._ref(b), accum_dtype, self._fresh_ref(a, b, accum_dtype))
            ),
        )

    def reduce_sum(self, x: T, axis: int, accum_dtype: str) -> T:
        """``sum(x)`` over ``axis``, accumulating in ``accum_dtype``."""
        out = self._fresh_ref_like(x, drop_axis=axis, dtype=accum_dtype)
        return T(self, self._emit(Reduce(self._ref(x), "sum", axis, accum_dtype, out)))

    def reduce_max(self, x: T, axis: int, accum_dtype: str) -> T:
        """``max(x)`` over ``axis`` (the amax root of per-group fp8 quant, issue #57)."""
        out = self._fresh_ref_like(x, drop_axis=axis, dtype=accum_dtype)
        return T(self, self._emit(Reduce(self._ref(x), "max", axis, accum_dtype, out)))

    # --- pointwise activations / unary math ---------------------------------
    def _unary(self, fn: str, x: T) -> T:
        out = self._fresh_ref_like(x, dtype=self._ref(x).dtype)
        return T(self, self._emit(Pointwise(fn, (self._ref(x),), self._ref(x).dtype, out)))

    def silu(self, x: T) -> T:     return self._unary("silu", x)      # x * sigmoid(x)
    def gelu(self, x: T) -> T:     return self._unary("gelu", x)      # tanh approximation
    def exp(self, x: T) -> T:      return self._unary("exp", x)
    def tanh(self, x: T) -> T:     return self._unary("tanh", x)
    def sigmoid(self, x: T) -> T:  return self._unary("sigmoid", x)
    def sqrt(self, x: T) -> T:     return self._unary("sqrt", x)
    def abs(self, x: T) -> T:      return self._unary("abs", x)
    def neg(self, x: T) -> T:      return self._unary("neg", x)

    def minimum(self, a: T, b: T) -> T:  return self._pointwise("min", (a, b))
    def maximum(self, a: T, b: T) -> T:  return self._pointwise("max", (a, b))

    def where(self, mask: T, a: T, b: T) -> T:
        """``mask ? a : b`` elementwise (data-SHAPING, oracle-safe).

        Pure pointwise — ``torch.where`` lowers bit-exact with ``tl.where``. The
        canonical use is attention masking: a causal/KV-padding mask is DERIVED
        from indices (positions, ``cu_seqlens``), never from values, so there is
        no data-dependent control flow (A4 case (a), not (c)). The output dtype
        follows ``a``/``b`` (the value operands), NOT the bool mask.
        """
        ra, rb = self._ref(a), self._ref(b)
        out = self._fresh_ref_like(a, dtype=ra.dtype)
        return T(self, self._emit(Pointwise("where", (self._ref(mask), ra, rb), ra.dtype, out)))

    def rsqrt(self, x: T) -> T:
        """``rsqrt(x)`` (pointwise; the reciprocal-sqrt root of RMSNorm)."""
        out = self._fresh_ref_like(x, dtype=self._ref(x).dtype)
        return T(self, self._emit(Pointwise("rsqrt", (self._ref(x),), self._ref(x).dtype, out)))

    # --- data-addressing nodes (docs/brainstorm/06 A4 case (a); oracle-safe) ---
    def gather(self, base: T, index: T, axis: int = 0) -> T:
        """``out = base[index]`` along ``axis`` — data-ADDRESSING (oracle-safe).

        ``index`` is a tensor (an input, e.g. RoPE ``positions``); the gather is
        pure, parallel, deterministic. Torch lowers to ``index_select`` (the
        oracle); Triton lowers to ``tl.load(base + index*stride)``. NOT for
        data-SELECTION (a gather whose index is a value computed in the kernel).
        """
        rb, ri = self._ref(base), self._ref(index)
        # N-D index gather: the index's FULL shape replaces the gathered axis
        # (in-place, index_select placement). base[P, ps, h, d] gathered on axis 0
        # by index[M, N] -> [M, N, ps, h, d]. The 1-D case (RoPE positions[T]) is
        # the special case ri.shape == (T,).
        out_shape = list(rb.shape[:axis]) + list(ri.shape) + list(rb.shape[axis + 1 :])
        out_sub = list(rb.subscript) if rb.subscript else list(range(len(rb.shape)))
        # the gathered axis takes the index's leading symbol (e.g. P -> T)
        if ri.subscript:
            out_sub[axis] = ri.subscript[0]
        out = TensorRef(self._fresh(), rb.dtype, tuple(out_shape), tuple(out_sub))
        return T(self, self._emit(Gather(rb, ri, axis, out)))

    def slice(self, base: T, axis: int, start: int, stop: int) -> T:
        """``out = base[..., start:stop]`` along ``axis`` (static range; oracle-safe).

        ``start``/``stop`` are PYTHON ints (known at trace time) — a pure
        sub-range load, never a data-dependent slice. RoPE splits the head into
        its rotation halves with two slices along the last axis.
        """
        rb = self._ref(base)
        out_shape = list(rb.shape)
        # the sliced axis size is the resolved bound width (symbolic if a str bound)
        if isinstance(start, int) and isinstance(stop, int):
            out_shape[axis] = stop - start
        out = TensorRef(self._fresh(), rb.dtype, tuple(out_shape), rb.subscript)
        return T(self, self._emit(Slice(rb, axis, start, stop, out)))

    def concat(self, a: T, b: T, axis: int) -> T:
        """``out = cat([a, b], axis)`` (reassemble two parts; oracle-safe)."""
        ra, rb = self._ref(a), self._ref(b)
        out_shape = list(ra.shape)
        # the concatenated axis size is symbolic (resolved at eval from tensors);
        # use a fresh symbol so it does not collide with an input symbol.
        out_shape[axis] = f"_cat{self._n}"
        out = TensorRef(self._fresh(), ra.dtype, tuple(out_shape), ra.subscript)
        return T(self, self._emit(Concat(ra, rb, axis, out)))

    def unsqueeze(self, base: T, axis: int) -> T:
        """Insert a size-1 axis so a per-row quantity broadcasts (oracle-safe view)."""
        rb = self._ref(base)
        out_shape = list(rb.shape[:axis]) + [1] + list(rb.shape[axis:])
        out_sub = list(rb.subscript[:axis]) + [f"_u{self._n}"] + list(rb.subscript[axis:])
        out = TensorRef(self._fresh(), rb.dtype, tuple(out_shape), tuple(out_sub))
        return T(self, self._emit(Unsqueeze(rb, axis, out)))

    def store(self, name: str, val: T) -> None:
        """Write ``val`` to output ``name`` (a dataflow graph sink)."""
        if name not in self.out_decls:
            raise KeyError(f"store({name!r}): not a declared output")
        self._nodes.append(Store(self.out_decls[name], self._ref(val)))

    # --- internals -----------------------------------------------------------
    def _ref(self, h: T) -> TensorRef:
        return self._tensors[h.name]

    def _pointwise(self, fn: str, args: tuple[T, ...]) -> T:
        out_dtype = self._ref(args[0]).dtype
        out = self._fresh_ref_like(args[0], dtype=out_dtype)
        return T(self, self._emit(Pointwise(fn, tuple(self._ref(a) for a in args), out_dtype, out)))

    def _cast(self, x: T, dtype: str) -> T:
        out = self._fresh_ref_like(x, dtype=dtype)
        return T(self, self._emit(Pointwise("cast", (self._ref(x),), dtype, out)))

    def _emit(self, node: MathNode) -> str:
        self._nodes.append(node)
        # register the node's output TensorRef (it was created by the caller)
        self._tensors[node.out.name] = node.out  # type: ignore[attr-defined]
        return node.out.name  # type: ignore[attr-defined]

    def _fresh(self) -> str:
        self._n += 1
        return f"_t{self._n}"

    def _lit_name(self, value: int | float) -> str:
        # Literals are stored as synthetic 0-d tensors; the evaluator special-cases them.
        name = f"_lit_{value!r}"
        if name not in self._tensors:
            self._tensors[name] = TensorRef(name, "fp32", (), ())
        self._nodes.append(_LitMarker(name, value))  # type: ignore[arg-type]
        return name

    def _dim_name(self, tensor: str, axis: int) -> str:
        name = f"_dim_{tensor}_{axis}"
        if name not in self._tensors:
            self._tensors[name] = TensorRef(name, "fp32", (), ())
        self._nodes.append(_DimRefMarker(name, tensor, axis))  # type: ignore[arg-type]
        return name

    def _fresh_ref(self, a: T, b: T, dtype: str) -> TensorRef:
        """MMA output shape: outer dims of a × outer dims of b (a[m,k]@b[k,n]->[m,n])."""
        ra, rb = self._ref(a), self._ref(b)
        # contracted dims = subscripts shared by a and b; out keeps the rest.
        a_subs = list(ra.subscript) if ra.subscript else list(range(len(ra.shape)))
        b_subs = list(rb.subscript) if rb.subscript else list(range(len(rb.shape)))
        contracted = [s for s in a_subs if s in b_subs]
        m_subs = [s for s in a_subs if s not in contracted]
        n_subs = [s for s in b_subs if s not in contracted]
        shape = tuple(ra.shape[i] for i, s in enumerate(a_subs) if s in m_subs) + tuple(
            rb.shape[i] for i, s in enumerate(b_subs) if s in n_subs
        )
        return TensorRef(self._fresh(), dtype, shape, tuple(m_subs + n_subs))

    def _fresh_ref_like(
        self, src: T, *, dtype: str | None = None, drop_axis: int | None = None
    ) -> TensorRef:
        r = self._ref(src)
        shape = r.shape
        sub = r.subscript
        if drop_axis is not None:
            shape = r.shape[:drop_axis] + r.shape[drop_axis + 1 :]
            sub = r.subscript[:drop_axis] + r.subscript[drop_axis + 1 :]
        return TensorRef(self._fresh(), dtype or r.dtype, shape, sub)

    def finish(self) -> MathBody:
        return MathBody(
            MathIR(tuple(self._nodes), dict(self._tensors)),
            dict(self.in_decls),
            dict(self.out_decls),
        )


@dataclass(frozen=True)
class _LitMarker:
    """A literal scalar marker (not a real math node; consumed by the evaluators)."""

    name: str
    value: int | float


@dataclass(frozen=True)
class _DimRefMarker:
    """A symbolic dim reference ``inputs[tensor].shape[axis]`` (e.g. reduction width).

    Not a real math node (the algebra stays at ~5 kinds); consumed by the
    evaluators like ``_LitMarker``. The torch evaluator resolves it to the
    concrete shape value; the rowwise codegen lowers it to the runtime dim arg
    ``d_<symbol>`` (one per reduction axis, shared by every load/weight on it).
    """

    name: str
    tensor: str
    axis: int


def build_body(
    body_fn,
    in_decls: dict[str, TensorRef],
    out_decls: dict[str, TensorRef],
) -> MathBody:
    """Run the body once in build mode, returning the math IR + decls."""
    ctx = MathBodyCtx(in_decls, out_decls)
    body_fn(ctx)
    return ctx.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Torch evaluator — the auto-reference (launch-agnostic, fully vectorized)
# ═══════════════════════════════════════════════════════════════════════════════


class _TorchEval:
    """Walk the math nodes in order, evaluating each into a dict of named tensors.

    Fully vectorized over all dims (no grid — torch needs none). ``MMA`` ->
    ``a.to(accum) @ b.to(accum)``; ``Reduce`` -> ``torch.sum``; ``Pointwise`` ->
    the matching torch op. Bit-exact with a hand ``a.float() @ b.float()``.
    """

    def __init__(self, inputs: dict[str, torch.Tensor], out_dtype: str, body: MathBody):
        self.inputs = inputs
        self.out_dtype = out_dtype
        self.body = body
        self.values: dict[str, Any] = {}

    def eval(self) -> dict[str, torch.Tensor]:
        for node in self.body.ir.nodes:
            self._eval_node(node)
        # Return only the stored outputs, in declaration order.
        return {name: self.values[name] for name in self.body.out_decls if name in self.values}

    def _eval_node(self, node: MathNode | _LitMarker) -> None:
        if isinstance(node, _LitMarker):
            self.values[node.name] = node.value
            return
        if isinstance(node, _DimRefMarker):
            self.values[node.name] = self.inputs[node.tensor].shape[node.axis]
            return
        if isinstance(node, Load):
            self.values[node.ref.name] = self.inputs[node.ref.name]
            return
        if isinstance(node, MMA):
            a = self.values[node.a.name].to(to_torch_dtype(node.accum_dtype))
            b = self.values[node.b.name].to(to_torch_dtype(node.accum_dtype))
            self.values[node.out.name] = a @ b
            return
        if isinstance(node, Reduce):
            x = self.values[node.x.name].to(to_torch_dtype(node.accum_dtype))
            if node.op == "sum":
                self.values[node.out.name] = x.sum(node.axis, keepdim=True)
            elif node.op == "max":
                self.values[node.out.name] = x.amax(node.axis, keepdim=True)
            else:
                raise NotImplementedError(f"reduce op {node.op!r}")
            return
        if isinstance(node, Pointwise):
            args = [self.values[a.name] for a in node.args]
            if node.fn == "cast":
                res = args[0]
            elif node.fn == "mul":
                res = args[0] * args[1]
            elif node.fn == "add":
                res = args[0] + args[1]
            elif node.fn == "div":
                res = args[0] / args[1]
            elif node.fn == "rsqrt":
                res = torch.rsqrt(_scalar_to_tensor(args[0]))
            elif node.fn == "sub":
                res = args[0] - args[1]
            elif node.fn == "neg":
                res = -args[0]
            elif node.fn == "abs":
                res = args[0].abs()
            elif node.fn == "exp":
                res = torch.exp(_scalar_to_tensor(args[0]))
            elif node.fn == "tanh":
                res = torch.tanh(_scalar_to_tensor(args[0]))
            elif node.fn == "sigmoid":
                res = torch.sigmoid(_scalar_to_tensor(args[0]))
            elif node.fn == "sqrt":
                res = torch.sqrt(_scalar_to_tensor(args[0]))
            elif node.fn == "silu":
                res = args[0] * torch.sigmoid(args[0])
            elif node.fn == "gelu":  # tanh approx (matches flashinfer/vLLM gelu_and_mul)
                _x = args[0]
                _inner = 0.7978845608028654 * (_x + 0.044715 * _x * _x * _x)
                res = 0.5 * _x * (1.0 + torch.tanh(_inner))
            elif node.fn == "min":
                res = _scalar_aware_binary(torch.minimum, args[0], args[1])
            elif node.fn == "max":
                res = _scalar_aware_binary(torch.maximum, args[0], args[1])
            elif node.fn == "where":
                res = torch.where(args[0].bool(), args[1], args[2])
            else:
                raise NotImplementedError(f"pointwise fn {node.fn!r}")
            # Honor the node's declared out dtype: torch promotes bf16*bf16 via
            # sigmoid to fp32; cast back so a bf16 silu STAYS bf16 (matches the
            # codegen, which stores into a bf16 tile). This is what keeps mixed-
            # dtype ops (e.g. fp8 quant's fp8 q + fp32 scale) honest.
            tgt = to_torch_dtype(self._resolve_dtype(node.out_dtype))
            self.values[node.out.name] = res.to(tgt)
            return
        if isinstance(node, Store):
            # Drop keepdim size-1 axes so a reduce stored straight to a lower-rank
            # output (e.g. amax[G,1] -> amax[G]) matches the output decl. rmsnorm's
            # full-row store is already at decl rank, so the loop is a no-op there.
            val = self.values[node.val.name]
            target_rank = len(node.ref.shape)
            while val.dim() > target_rank and val.shape[-1] == 1:
                val = val.squeeze(-1)
            self.values[node.ref.name] = val
            return
        if isinstance(node, Gather):
            base = self.values[node.base.name]
            index = self.values[node.index.name]
            # torch.index_select: index must be 1-D int64. We flatten the index,
            # select, then RESHAPE the index's shape back in-place (so an N-D
            # index like page_table[M,N] yields [M,N,...], not [M*N,...]). The
            # oracle is exact; matches `base[index]` advanced indexing for axis=0.
            idx_shape = list(index.shape)
            idx = index.reshape(-1).to(torch.int64)
            sel = torch.index_select(base, node.axis, idx)
            target = list(base.shape[: node.axis]) + idx_shape + list(base.shape[node.axis + 1 :])
            self.values[node.out.name] = sel.reshape(target)
            return
        if isinstance(node, Slice):
            base = self.values[node.base.name]
            ax = _norm_axis(node.axis, base.dim())
            start = _resolve_bound(node.start, base.shape[ax])
            stop = _resolve_bound(node.stop, base.shape[ax])
            self.values[node.out.name] = torch.narrow(base, ax, start, stop - start)
            return
        if isinstance(node, Concat):
            a = self.values[node.a.name]
            b = self.values[node.b.name]
            self.values[node.out.name] = torch.cat([a, b], dim=_norm_axis(node.axis, a.dim()))
            return
        if isinstance(node, Unsqueeze):
            base = self.values[node.base.name]
            ax = node.axis if node.axis >= 0 else node.axis + base.dim() + 1
            self.values[node.out.name] = base.unsqueeze(ax)
            return
        raise TypeError(f"unevaluable node {node!r}")

    def _resolve_dtype(self, dtype: str) -> str:
        return self.out_dtype if dtype == "__OUT_DTYPE__" else dtype


def eval_torch(
    body: MathBody, inputs: dict[str, torch.Tensor], out_dtype: str
) -> dict[str, torch.Tensor]:
    """Interpret the math IR into output tensors (the auto-reference)."""
    return _TorchEval(inputs, out_dtype, body).eval()


def _scalar_to_tensor(v: Any) -> Any:
    """Wrap a python scalar (int/float) to a 0-d fp32 tensor for a unary pointwise.

    A ``ctx.dim(tensor, axis)`` resolves to a python int in the evaluator (the
    shape value); a unary math op like ``rsqrt(dim(...))`` (a fused scale =
    ``1/sqrt(head_dim)``) needs a tensor operand. torch's binary ops already
    promote ``tensor / int``, but the unary math fns (rsqrt/sqrt/exp/...) require
    a tensor. This keeps ``dim``/``lit`` first-class scalars in any pointwise.
    """
    if isinstance(v, torch.Tensor):
        return v
    return torch.tensor(float(v), dtype=torch.float32)


def _scalar_aware_binary(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: Any,
    b: Any,
) -> torch.Tensor:
    """Apply a binary torch op, promoting a literal-scalar operand to a tensor.

    ``mul/add/div/sub`` already accept ``(tensor, python_scalar)`` in torch, but
    ``maximum/minimum`` require two tensors — and a body's ``ctx.lit(c)`` resolves
    to a python scalar in the evaluator (e.g. the fp8 quant's ``maximum(amax,
    1e-12)`` / ``minimum(t, FP8_MAX)`` clamp). Wrap the scalar into a 0-d tensor of
    the tensor operand's dtype so the clamp lowers without a special-case node.
    """
    def _as_tensor(v: Any, ref: torch.Tensor) -> Any:
        if isinstance(v, torch.Tensor):
            return v
        return torch.tensor(v, dtype=ref.dtype, device=ref.device)

    ref = a if isinstance(a, torch.Tensor) else b
    return fn(_as_tensor(a, ref), _as_tensor(b, ref))


def _norm_axis(axis: int, rank: int) -> int:
    """Normalize a possibly-negative axis (``-1`` -> ``rank-1``) for torch ops."""
    return axis % rank if rank else axis


def _resolve_bound(bound: int | str, axis_size: int) -> int:
    """Resolve a Slice bound: an ``int`` verbatim, or a ``str`` over ``shape``.

    ``"shape//2"`` -> ``axis_size // 2``; ``"shape"`` -> ``axis_size``. Only
    arithmetic on the single name ``shape`` is permitted (no builtins), so RoPE
    can halve the symbolic head axis (``D``) without a trace-time constant.
    """
    if isinstance(bound, int):
        return bound
    code = compile(str(bound), "<slice-bound>", "eval")
    allowed = {"shape": axis_size}
    for name in code.co_names:
        if name not in allowed:
            raise ValueError(f"slice bound {bound!r}: unknown name {name!r}")
    return int(eval(code, {"__builtins__": {}}, allowed))  # noqa: S307


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Triton codegen — math IR -> tiled @triton.jit (the tiled_2d launch)
# ═══════════════════════════════════════════════════════════════════════════════

_TL_DTYPE = {
    "fp32": "tl.float32",
    "bf16": "tl.bfloat16",
    "fp16": "tl.float16",
    # fp8 (GPU-codegen only; the torch evaluator maps these via registry/dtypes).
    # Triton spells e4m3fn as ``float8e4nv`` (the OCP encoding the ``fp8`` short
    # name resolves to). fnuz is the AMD-native CDNA3 encoding (arch override).
    "fp8": "tl.float8e4nv",
    "fp8_e4m3fn": "tl.float8e4nv",
    "fp8_e4m3fnuz": "tl.float8e4fnuz",
    "bool": "tl.int1",  # attention masks (the where() mask operand)
}


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _find_mma(nodes: tuple[MathNode, ...]) -> MMA:
    mmas = [n for n in nodes if isinstance(n, MMA)]
    if len(mmas) != 1:
        raise NotImplementedError(
            f"tiled_2d lowering currently handles exactly one MMA node; found {len(mmas)}"
        )
    return mmas[0]


def _dim_roles(mma: MMA, body: MathBody) -> dict:
    """Map each subscript label to a role: 'm'/'n' (grid) or 'k' (contracted loop).

    The output decl's subscripts are the grid dims (m, n); the dim shared by a
    and b but absent from the output is the contracted K-loop dim. This is the
    Einstein-ish tiling the math IR's ``subscript`` field exists for.
    """
    a_sub = list(mma.a.subscript)
    b_sub = list(mma.b.subscript)
    # The output decl that this MMA feeds (first 2D output).
    out_sub = next(iter(body.out_decls.values())).subscript
    out_dims = set(out_sub)
    a_dims, b_dims = set(a_sub), set(b_sub)
    contracted = (a_dims & b_dims) - out_dims
    if len(contracted) != 1:
        raise NotImplementedError(
            f"could not uniquely identify the contracted (K) dim: "
            f"a={a_sub} b={b_sub} out={list(out_sub)} -> contracted={contracted}"
        )
    k_dim = next(iter(contracted))
    m_dim = out_sub[0]
    n_dim = out_sub[1] if len(out_sub) > 1 else out_sub[0]
    return {
        "m": m_dim,
        "n": n_dim,
        "k": k_dim,
        "a_sub": a_sub,
        "b_sub": b_sub,
        "out_sub": list(out_sub),
    }


class _TritonGen:
    """Emit a tiled @triton.jit kernel for a bare GEMM math IR (tiled_2d launch).

    Structure (docs/brainstorm/08 §3; the hand ``mm_fp8_blockscale_kernel.py`` idiom):
      1. header: ``program_id(0)``/``(1)`` = output tile (m, n); offset aranges.
      2. ``acc = tl.zeros([BLOCK_M, BLOCK_N], fp32)``  (from the MMA node).
      3. K-loop: tiled masked loads of the MMA's operands + ``acc += tl.dot(...)``.
      4. epilogue: the post-MMA pointwise chain (cast, etc.) over the register tile.
      5. masked tiled store.
    """

    # Default tile sizes (Phase 2.0a: correctness only; autotune is Phase 2.2).
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

    def __init__(self, body: MathBody, out_dtype: str):
        self.body = body
        self.out_dtype = out_dtype
        self.mma = _find_mma(body.ir.nodes)
        self.roles = _dim_roles(self.mma, body)
        self.lines: list[str] = []

    def _tl_dtype(self, dtype: str) -> str:
        return _TL_DTYPE[self.out_dtype if dtype == "__OUT_DTYPE__" else dtype]

    def kernel_source(self) -> str:
        r = self.roles
        a, b, out_name = self.mma.a.name, self.mma.b.name, next(iter(self.body.out_decls))
        # Signature: ptrs + per-axis strides + dim values (M, N, K) + BLOCK constexprs.
        sig = [
            f"{a}_ptr",
            f"{a}_stride_{r['a_sub'][0]}",
            f"{a}_stride_{r['a_sub'][1]}",
            f"{b}_ptr",
            f"{b}_stride_{r['b_sub'][0]}",
            f"{b}_stride_{r['b_sub'][1]}",
            f"{out_name}_ptr",
            f"{out_name}_stride_{r['out_sub'][0]}",
            f"{out_name}_stride_{r['out_sub'][1]}",
            r["m"],
            r["n"],
            r["k"],
            "BLOCK_M: tl.constexpr",
            "BLOCK_N: tl.constexpr",
            "BLOCK_K: tl.constexpr",
        ]
        L = self.lines
        L.append("    pid_m = tl.program_id(0)")
        L.append("    pid_n = tl.program_id(1)")
        L.append("    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)")
        L.append("    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)")
        L.append("    offs_k = tl.arange(0, BLOCK_K)")
        L.append("    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)")
        L.append(f"    for k_start in range(0, {r['k']}, BLOCK_K):")
        L.append("        k_offs = k_start + offs_k")
        # a tile [BLOCK_M, BLOCK_K]: axis0=m (grid), axis1=k (loop)
        am = f"{a}_stride_{r['a_sub'][0]}"
        ak = f"{a}_stride_{r['a_sub'][1]}"
        L.append(f"        a_ptrs = {a}_ptr + offs_m[:, None] * {am} + k_offs[None, :] * {ak}")
        L.append(f"        a_mask = (offs_m[:, None] < {r['m']}) & (k_offs[None, :] < {r['k']})")
        L.append("        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)")
        # b tile [BLOCK_K, BLOCK_N]: axis0=k (loop), axis1=n (grid)
        bk = f"{b}_stride_{r['b_sub'][0]}"
        bn = f"{b}_stride_{r['b_sub'][1]}"
        L.append(f"        b_ptrs = {b}_ptr + k_offs[:, None] * {bk} + offs_n[None, :] * {bn}")
        L.append(f"        b_mask = (k_offs[:, None] < {r['k']}) & (offs_n[None, :] < {r['n']})")
        L.append("        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)")
        # dot: fp32 accum; ieee precision for fp32 operands (no TF32), tensor cores for bf16
        if self.out_dtype == "fp32":
            L.append('        acc += tl.dot(a_tile, b_tile, input_precision="ieee")')
        else:
            L.append("        acc += tl.dot(a_tile, b_tile)")
        # epilogue: walk post-MMA nodes (pointwise chain over acc) + the store.
        tile_vars = {self.mma.out.name: "acc"}
        # Resolve scalar markers (dim/lit) the epilogue's pointwise chain may
        # reference (e.g. a fused scale ``acc * rsqrt(dim(k, axis=0))``). The dim
        # marker maps its tensor's axis-subscript to the matching signature symbol
        # (m/n/k), exactly as the rowwise codegen resolves it to ``d_<sym>``.
        sym_to_sig = {r["m"]: r["m"], r["n"]: r["n"], r["k"]: r["k"]}
        for node in self.body.ir.nodes:
            if isinstance(node, _DimRefMarker):
                ref = self.body.ir.tensors.get(node.tensor) or self.body.in_decls.get(node.tensor)
                if ref is not None and node.axis < len(ref.subscript):
                    sym = ref.subscript[node.axis]
                    tile_vars[node.name] = sym_to_sig.get(sym, sym)
            elif isinstance(node, _LitMarker):
                tile_vars[node.name] = repr(node.value)
        emitted_mma = False
        for node in self.body.ir.nodes:
            if isinstance(node, MMA):
                emitted_mma = True
                continue
            if not emitted_mma:
                continue  # pre-MMA (Loads) consumed inside the loop already
            if isinstance(node, Pointwise):
                tile_vars[node.out.name] = self._emit_pointwise(node, tile_vars)
            elif isinstance(node, Store):
                val = tile_vars[node.val.name]
                om = f"{out_name}_stride_{r['out_sub'][0]}"
                on = f"{out_name}_stride_{r['out_sub'][1]}"
                L.append(
                    f"    out_ptrs = {out_name}_ptr "
                    f"+ offs_m[:, None] * {om} + offs_n[None, :] * {on}"
                )
                L.append(
                    f"    out_mask = (offs_m[:, None] < {r['m']}) & (offs_n[None, :] < {r['n']})"
                )
                L.append(f"    tl.store(out_ptrs, {val}, mask=out_mask)")
        src = "import triton\nimport triton.language as tl\n\n\n"
        src += f"@triton.jit\ndef _kernel({', '.join(sig)}):\n"
        src += "\n".join(self.lines) + "\n"
        return src

    def _emit_pointwise(self, node: Pointwise, tile_vars: dict[str, str]) -> str:
        args = [tile_vars[a.name] for a in node.args]
        v = f"ep_{node.out.name}"
        a0 = args[0]
        fn = node.fn
        if fn == "cast":
            self.lines.append(f"    {v} = {a0}.to({self._tl_dtype(node.out_dtype)})")
        elif fn == "mul":
            self.lines.append(f"    {v} = {a0} * {args[1]}")
        elif fn == "add":
            self.lines.append(f"    {v} = {a0} + {args[1]}")
        elif fn == "sub":
            self.lines.append(f"    {v} = {a0} - {args[1]}")
        elif fn == "div":
            self.lines.append(f"    {v} = {a0} / {args[1]}")
        elif fn == "neg":
            self.lines.append(f"    {v} = -{a0}")
        elif fn == "abs":
            self.lines.append(f"    {v} = tl.abs({a0})")
        elif fn == "exp":
            self.lines.append(f"    {v} = tl.exp({a0})")
        elif fn == "tanh":
            self.lines.append(f"    {v} = tl.tanh({a0})")
        elif fn == "sigmoid":
            self.lines.append(f"    {v} = tl.sigmoid({a0})")
        elif fn == "sqrt":
            self.lines.append(f"    {v} = tl.sqrt({a0})")
        elif fn == "rsqrt":
            self.lines.append(f"    {v} = tl.rsqrt({a0})")
        elif fn == "silu":
            self.lines.append(f"    {v} = {a0} * tl.sigmoid({a0})")
        elif fn == "gelu":  # tanh approx
            self.lines.append(
                f"    {v} = 0.5 * {a0} * (1.0 + tl.tanh("
                f"0.7978845608028654 * ({a0} + 0.044715 * {a0} * {a0} * {a0})))"
            )
        elif fn == "min":
            self.lines.append(f"    {v} = tl.minimum({a0}, {args[1]})")
        elif fn == "max":
            self.lines.append(f"    {v} = tl.maximum({a0}, {args[1]})")
        elif fn == "where":
            self.lines.append(f"    {v} = tl.where({a0}, {args[1]}, {args[2]})")
        else:
            raise NotImplementedError(f"tiled_2d epilogue pointwise {fn!r}")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# §4b  Rowwise Triton codegen — math IR -> one-program-per-row (the rowwise grid)
# ═══════════════════════════════════════════════════════════════════════════════


class _TritonGenRowwise:
    """Emit a row-wise ``@triton.jit`` kernel for a math IR under a ``rowwise`` launch.

    One program per leading-dim row (``program_id(0)`` = row). The reduction
    axis (a ``Reduce(sum, axis=last)`` node) becomes the program-local 1D tile,
    padded to ``next_pow2(dim)`` and masked (``cols < d``). Mirrors the hand
    ``dual_rmsnorm_kernel.py`` idiom: load a ``[BLOCK_D]`` tile, reduce over it
    (``tl.sum(axis=0)``), then broadcast the per-row scalar back across the tile
    for the pointwise epilogue (scale, cast, weight-mul).

    Scope (Phase 2.0b): rank-2 inputs reduced over the LAST axis, fp32 reduction,
    rank-1 weights sharing the input's reduction dim (linked by the decl
    ``subscript`` symbol). Multiple independent reductions (e.g. dual_rmsnorm's
    two latents) compose — each ``Reduce`` + its loads + store emit in node order.
    """

    def __init__(self, body: MathBody, out_dtype: str):
        self.body = body
        self.out_dtype = out_dtype
        self.lines: list[str] = []
        self.vars: dict[str, str] = {}  # node-output name -> triton var
        self.scalars: dict[str, str] = {}  # marker name -> codegen scalar str
        self.symbols: dict[str, dict[str, str]] = {}  # reduction symbol -> {block,cols,dim}
        self.row_symbol: str | None = None  # leading-dim symbol (per-row scalar outputs)
        self._n = 0

    def fresh(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _tl_dtype(self, dtype: str) -> str:
        return _TL_DTYPE[self.out_dtype if dtype == "__OUT_DTYPE__" else dtype]

    def _ref(self, name: str) -> TensorRef:
        return self.body.ir.tensors[name]

    # --- symbol (tile-width) collection -------------------------------------
    def _collect_symbols(self) -> None:
        """Every ``Reduce`` over the last axis declares a reduction tile width.

        The symbol is the operand's ``subscript[axis]`` (e.g. ``d1``); rank-1
        weights sharing that symbol reuse the same BLOCK + mask bound. Deterministic
        dict insertion order (Reduce nodes in IR order) is load-bearing — the
        launcher re-runs the same scan so dim args land in signature order.
        """
        for node in self.body.ir.nodes:
            if isinstance(node, Reduce):
                op_ref = self._ref(node.x.name)
                if node.axis != len(op_ref.shape) - 1:
                    raise NotImplementedError(
                        f"rowwise lowering reduces only over the LAST axis; "
                        f"got axis={node.axis} of rank-{len(op_ref.shape)}"
                    )
                sym = op_ref.subscript[node.axis]
                self.symbols.setdefault(
                    sym, {"block": f"BLOCK_D_{sym}", "cols": f"cols_{sym}", "dim": f"d_{sym}"}
                )
                # The leading-dim (row) symbol is axis 0 of a rank-2 reduce
                # operand; a per-row-scalar output (the reduction result, e.g.
                # amax[G]) carries THIS symbol as its last axis.
                if len(op_ref.shape) >= 2 and self.row_symbol is None:
                    self.row_symbol = op_ref.subscript[0]

    def _build_scalars(self) -> None:
        """Markers (literals + dim refs) -> codegen scalar strings."""
        for node in self.body.ir.nodes:
            if isinstance(node, _LitMarker):
                self.scalars[node.name] = repr(node.value)
            elif isinstance(node, _DimRefMarker):
                sym = self._ref(node.tensor).subscript[node.axis]
                self.scalars[node.name] = self.symbols[sym]["dim"]

    def _arg(self, name: str) -> str:
        if name in self.vars:
            return self.vars[name]
        if name in self.scalars:
            return self.scalars[name]
        raise KeyError(f"rowwise codegen: no var/scalar for {name!r}")

    # --- signature + body ---------------------------------------------------
    def _signature(self) -> list[str]:
        sig: list[str] = []
        for name in (*self.body.in_decls, *self.body.out_decls):
            ref = self.body.in_decls.get(name) or self.body.out_decls[name]
            sig.append(f"{name}_ptr")
            if len(ref.shape) == 2:
                sig.append(f"{name}_row_stride")
        for sym in self.symbols:
            sig.append(self.symbols[sym]["dim"])
        for sym in self.symbols:
            sig.append(f"{self.symbols[sym]['block']}: tl.constexpr")
        return sig

    def kernel_source(self) -> str:
        self._collect_symbols()
        self._build_scalars()
        L = self.lines
        L.append("    row = tl.program_id(0)")
        for sym in self.symbols:
            L.append(
                f"    {self.symbols[sym]['cols']} = tl.arange(0, {self.symbols[sym]['block']})"
            )
        for node in self.body.ir.nodes:
            self._emit_node(node)
        src = "import triton\nimport triton.language as tl\n\n\n"
        src += f"@triton.jit\ndef _kernel({', '.join(self._signature())}):\n"
        src += "\n".join(L) + "\n"
        return src

    def _emit_node(self, node: Any) -> None:
        if isinstance(node, Load):
            self._emit_load(node)
        elif isinstance(node, Unsqueeze):
            self._emit_unsqueeze(node)
        elif isinstance(node, Pointwise):
            self._emit_pointwise(node)
        elif isinstance(node, Reduce):
            self._emit_reduce(node)
        elif isinstance(node, Store):
            self._emit_store(node)
        # markers resolve via self.scalars; MMA is the tiled_2d path, not rowwise.

    def _emit_load(self, node: Load) -> None:
        ref = node.ref
        rank = len(ref.shape)
        sym = ref.subscript[rank - 1]
        v = self.fresh(ref.name)
        if rank == 1 and sym == self.row_symbol and sym not in self.symbols:
            # Per-row scalar input, e.g. temperatures[B] for softmax over
            # logits[B,V]. One program owns one row, so this is a scalar load
            # broadcast across that program's reduction tile.
            self.lines.append(f"    {v} = tl.load({ref.name}_ptr + row)")
            self.vars[ref.name] = v
            return
        if sym not in self.symbols:
            raise NotImplementedError(
                f"rowwise load of {ref.name!r}: axis symbol {sym!r} is not a "
                f"reduction axis (rowwise tiles only the reduced dim)"
            )
        info = self.symbols[sym]
        if rank == 2:
            self.lines.append(
                f"    {v} = tl.load({ref.name}_ptr + row * {ref.name}_row_stride "
                f"+ {info['cols']}, mask={info['cols']} < {info['dim']}, other=0.0)"
            )
        else:
            self.lines.append(
                f"    {v} = tl.load({ref.name}_ptr + {info['cols']}, "
                f"mask={info['cols']} < {info['dim']}, other=0.0)"
            )
        self.vars[ref.name] = v

    def _emit_unsqueeze(self, node: Unsqueeze) -> None:
        # Rowwise codegen already materializes per-row quantities as program
        # scalars and per-column quantities as vectors. Unsqueeze is a torch-side
        # broadcast-shape hint, so the device lowering aliases the same value.
        self.vars[node.out.name] = self._arg(node.base.name)

    def _emit_pointwise(self, node: Pointwise) -> None:
        args = [self._arg(a.name) for a in node.args]
        v = self.fresh("p")
        if node.fn == "cast":
            self.lines.append(f"    {v} = {args[0]}.to({self._tl_dtype(node.out_dtype)})")
        elif node.fn == "mul":
            self.lines.append(f"    {v} = {args[0]} * {args[1]}")
        elif node.fn == "add":
            self.lines.append(f"    {v} = {args[0]} + {args[1]}")
        elif node.fn == "div":
            self.lines.append(f"    {v} = {args[0]} / {args[1]}")
        elif node.fn == "rsqrt":
            self.lines.append(f"    {v} = tl.rsqrt({args[0]})")
        elif node.fn == "sub":
            self.lines.append(f"    {v} = {args[0]} - {args[1]}")
        elif node.fn == "neg":
            self.lines.append(f"    {v} = -{args[0]}")
        elif node.fn == "abs":
            self.lines.append(f"    {v} = tl.abs({args[0]})")
        elif node.fn == "exp":
            self.lines.append(f"    {v} = tl.exp({args[0]})")
        elif node.fn == "tanh":
            self.lines.append(f"    {v} = tl.tanh({args[0]})")
        elif node.fn == "sigmoid":
            self.lines.append(f"    {v} = tl.sigmoid({args[0]})")
        elif node.fn == "sqrt":
            self.lines.append(f"    {v} = tl.sqrt({args[0]})")
        elif node.fn == "silu":
            self.lines.append(f"    {v} = {args[0]} * tl.sigmoid({args[0]})")
        elif node.fn == "gelu":  # tanh approx
            self.lines.append(
                f"    {v} = 0.5 * {args[0]} * (1.0 + tl.tanh("
                f"0.7978845608028654 * ({args[0]} + 0.044715 * {args[0]} * {args[0]} * {args[0]})))"
            )
        elif node.fn == "min":
            self.lines.append(f"    {v} = tl.minimum({args[0]}, {args[1]})")
        elif node.fn == "max":
            self.lines.append(f"    {v} = tl.maximum({args[0]}, {args[1]})")
        elif node.fn == "where":
            self.lines.append(f"    {v} = tl.where({args[0]}, {args[1]}, {args[2]})")
        else:
            raise NotImplementedError(f"rowwise pointwise {node.fn!r}")
        self.vars[node.out.name] = v

    def _emit_reduce(self, node: Reduce) -> None:
        if node.op not in ("sum", "max"):
            raise NotImplementedError(f"rowwise reduce op {node.op!r}")
        x = self._arg(node.x.name)
        ref = self._ref(node.x.name)
        axis = _norm_axis(node.axis, len(ref.subscript))
        sym = ref.subscript[axis]
        info = self.symbols[sym]
        identity = "0.0" if node.op == "sum" else "-float('inf')"
        xm = self.fresh("m")
        self.lines.append(
            f"    {xm} = tl.where({info['cols']} < {info['dim']}, {x}, {identity})"
        )
        v = self.fresh("s")
        op = "tl.sum" if node.op == "sum" else "tl.max"
        self.lines.append(f"    {v} = {op}({xm}, axis=0)")
        self.vars[node.out.name] = v

    def _emit_store(self, node: Store) -> None:
        ref = node.ref
        rank = len(ref.shape)
        sym = ref.subscript[rank - 1]
        val = self._arg(node.val.name)
        # Per-row SCALAR output (the reduction result, e.g. amax[G]): one value
        # per program (row), stored at ptr + row. Its last-axis symbol is the
        # leading/row symbol, NOT a reduction symbol.
        if sym == self.row_symbol and sym not in self.symbols:
            self.lines.append(f"    tl.store({ref.name}_ptr + row, {val})")
            return
        info = self.symbols[sym]
        if rank == 2:
            self.lines.append(
                f"    tl.store({ref.name}_ptr + row * {ref.name}_row_stride + {info['cols']}, "
                f"{val}, mask={info['cols']} < {info['dim']})"
            )
        else:
            self.lines.append(
                f"    tl.store({ref.name}_ptr + {info['cols']}, {val}, "
                f"mask={info['cols']} < {info['dim']})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# §4c  Elementwise Triton codegen — math IR -> flat-1D pointwise (no reduction)
# ═══════════════════════════════════════════════════════════════════════════════


class _TritonGenElementwise:
    """Emit a flat-1D ``@triton.jit`` kernel for a pure-pointwise math IR.

    One program per 1D tile of the flattened output (``program_id(0)`` = tile).
    There is NO reduction axis: every node is a Load / Pointwise / Store over the
    SAME element grid, so the whole tensor is processed element-by-element. All
    inputs/outputs share the output's shape (addressed flat); this pattern does
    NOT support rank-1 broadcast (use ``rowwise`` for a weight broadcast).

    This is the launch for standalone elementwise ops — gated activations like
    ``silu_and_mul`` / ``gelu_and_mul`` (issue #67) — which have no reduction and
    therefore do not fit ``rowwise`` (which tiles a Reduce's contracted axis) or
    ``tiled_2d`` (which needs an MMA).
    """

    BLOCK = 1024

    def __init__(self, body: MathBody, out_dtype: str):
        self.body = body
        self.out_dtype = out_dtype
        self.lines: list[str] = []
        self.vars: dict[str, str] = {}
        self._n = 0
        self._needs_libdevice = False

    def fresh(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _tl_dtype(self, dtype: str) -> str:
        return _TL_DTYPE[self.out_dtype if dtype == "__OUT_DTYPE__" else dtype]

    def _signature(self) -> list[str]:
        sig: list[str] = []
        for name in (*self.body.in_decls, *self.body.out_decls):
            sig.append(f"{name}_ptr")
        sig.append("numel")
        sig.append("BLOCK: tl.constexpr")
        return sig

    def kernel_source(self) -> str:
        L = self.lines
        L.append("    pid = tl.program_id(0)")
        L.append("    offs = pid * BLOCK + tl.arange(0, BLOCK)")
        L.append("    mask = offs < numel")
        for node in self.body.ir.nodes:
            self._emit_node(node)
        src = "import triton\nimport triton.language as tl\n"
        if self._needs_libdevice:
            # Triton 3.7+ dropped ``tl.tanh``; ``libdevice.tanh`` is the faithful
            # (CUDA-quality) equivalent — the same routine torch.tanh lowers to.
            src += "from triton.language.extra import libdevice\n"
        src += "\n\n"
        src += f"@triton.jit\ndef _kernel({', '.join(self._signature())}):\n"
        src += "\n".join(L) + "\n"
        return src

    def _emit_node(self, node: Any) -> None:
        if isinstance(node, Load):
            v = self.fresh(node.ref.name)
            self.lines.append(
                f"    {v} = tl.load({node.ref.name}_ptr + offs, mask=mask, other=0.0)"
            )
            self.vars[node.ref.name] = v
        elif isinstance(node, Pointwise):
            args = [self.vars[a.name] for a in node.args]
            v = self.fresh("p")
            a0 = args[0]
            fn = node.fn
            if fn == "cast":
                self.lines.append(f"    {v} = {a0}.to({self._tl_dtype(node.out_dtype)})")
            elif fn == "mul":
                self.lines.append(f"    {v} = {a0} * {args[1]}")
            elif fn == "add":
                self.lines.append(f"    {v} = {a0} + {args[1]}")
            elif fn == "div":
                self.lines.append(f"    {v} = {a0} / {args[1]}")
            elif fn == "sub":
                self.lines.append(f"    {v} = {a0} - {args[1]}")
            elif fn == "neg":
                self.lines.append(f"    {v} = -{a0}")
            elif fn == "abs":
                self.lines.append(f"    {v} = tl.abs({a0})")
            elif fn == "exp":
                self.lines.append(f"    {v} = tl.exp({a0})")
            elif fn == "tanh":
                self._needs_libdevice = True
                self.lines.append(f"    {v} = libdevice.tanh({a0})")
            elif fn == "sigmoid":
                self.lines.append(f"    {v} = tl.sigmoid({a0})")
            elif fn == "sqrt":
                self.lines.append(f"    {v} = tl.sqrt({a0})")
            elif fn == "rsqrt":
                self.lines.append(f"    {v} = tl.rsqrt({a0})")
            elif fn == "silu":
                self.lines.append(f"    {v} = {a0} * tl.sigmoid({a0})")
            elif fn == "gelu":  # tanh approx — Triton 3.7+ has no tl.tanh
                self._needs_libdevice = True
                self.lines.append(
                    f"    {v} = 0.5 * {a0} * (1.0 + libdevice.tanh("
                    f"0.7978845608028654 * ({a0} + 0.044715 * {a0} * {a0} * {a0})))"
                )
            elif fn == "min":
                self.lines.append(f"    {v} = tl.minimum({a0}, {args[1]})")
            elif fn == "max":
                self.lines.append(f"    {v} = tl.maximum({a0}, {args[1]})")
            elif fn == "where":
                self.lines.append(f"    {v} = tl.where({a0}, {args[1]}, {args[2]})")
            else:
                raise NotImplementedError(f"elementwise pointwise {fn!r}")
            self.vars[node.out.name] = v
        elif isinstance(node, Store):
            val = self.vars[node.val.name]
            self.lines.append(f"    tl.store({node.ref.name}_ptr + offs, {val}, mask=mask)")
        # markers / Reduce / MMA are not part of the elementwise pattern.


def _has_addressing(body: MathBody) -> bool:
    """True if the IR uses any data-addressing node (Gather/Slice/Concat/Unsqueeze).

    Such bodies need the multi-dim lowering (per-axis coordinate mapping), not
    the flat-1D elementwise path (which assumes every operand shares the
    output's flat grid). RoPE (issue #68) is the canonical case.
    """
    return any(isinstance(n, (Gather, Slice, Concat, Unsqueeze)) for n in body.ir.nodes)


# ═══════════════════════════════════════════════════════════════════════════════
# §4c  Multi-dim addressing Triton codegen — math IR -> flat-tiled kernel with
#      per-axis coordinate mapping (the data-addressing launch; docs/brainstorm/06 A4)
# ═══════════════════════════════════════════════════════════════════════════════


class _TritonGenMultiDim:
    """Emit a flat-tiled ``@triton.jit`` kernel for a math IR with data-ADDRESSING.

    One program per flat tile of the output (``program_id(0)`` = tile), like the
    elementwise gen — but the output is multi-dimensional, so each lane's flat
    ``offs`` is decomposed into per-axis coordinates ``c0..c{R-1}`` (the output
    is row-major, ``D{R-1}`` contiguous). Each math node then reads its element
    via a *per-axis coordinate map* (an expr in ``c0..c{R-1}``) propagated down
    from the Store, so a ``Slice``/``Unsqueeze``/``Concat``/``Gather`` translates
    output coords into its own operand's coords:

      * ``Slice(base, axis, start)``      -> base coord[axis] = coord[axis] + start
      * ``Unsqueeze(base, axis)``         -> base drops the (size-1) axis
      * ``Concat(a, b, axis)``            -> ``tl.where(coord[axis] < len_a, a, b)``,
                                            with b's coord[axis] shifted by -len_a
      * ``Gather(base, index, axis)``     -> base coord[axis] = the loaded index

    The recursion is driven from each Store (a coord_map is passed down per edge),
    so a node consumed by two siblings with different coord_maps (RoPE's ``cs`` is
    sliced into BOTH cos and sin) is simply loaded twice — correct, just
    redundant. Emits only pure, parallel, in-bounds loads (the index is an INPUT
    tensor — A4 case (a), oracle-safe). Bit-exact with the torch evaluator.
    """

    BLOCK = 1024

    def __init__(self, body: MathBody, out_dtype: str, sym_vals: dict[str, int]):
        self.body = body
        self.out_dtype = out_dtype
        self.sym_vals = sym_vals
        self.lines: list[str] = []
        # memo: (name, coord-map signature) -> emitted value var (dedupe identical
        # loads so a fan-out doesn't blow up the kernel; different coord-maps
        # (cos vs sin path) still emit separately, which is correct).
        self._memo: dict[tuple[str, tuple[str, ...]], str] = {}
        self._shape_cache: dict[str, tuple[int, ...]] = {}
        self._n = 0
        self._needs_libdevice = False

    def fresh(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _resolve_raw(self, raw: tuple[int | str, ...]) -> tuple[int, ...]:
        """Resolve a raw (symbolic) shape tuple against sym_vals."""
        return tuple(d if isinstance(d, int) else int(self.sym_vals[d]) for d in raw)

    def _shape(self, name: str) -> tuple[int, ...]:
        """Resolve a tensor's shape by DERIVING it from its producer node.

        The builder's ``TensorRef.shape`` is lossy for symbolic slices — a
        ``Slice(axis=-1, 0, "shape//2")`` keeps the full ``"D"`` symbol on the
        sliced axis (it cannot spell ``D/2`` as one symbol), so reading the raw
        shape would report the full head size and the ``Concat`` predicate
        (``coord < len_a``) would be wrong. Instead we derive each axis from the
        IR: Slice -> ``stop - start``, Concat -> ``a + b``, Unsqueeze -> insert
        1, Gather -> the index's length, Pointwise -> the per-axis broadcast.
        """
        if name in self._shape_cache:
            return self._shape_cache[name]
        # Declared in/outputs are leaves — read their (symbolic) shape directly.
        if name in self.body.in_decls:
            sh = self._resolve_raw(self.body.in_decls[name].shape)
            self._shape_cache[name] = sh
            return sh
        if name in self.body.out_decls:
            sh = self._resolve_raw(self.body.out_decls[name].shape)
            self._shape_cache[name] = sh
            return sh
        node = self._producer(name)
        if isinstance(node, Load):
            sh = self._resolve_raw(node.ref.shape)
        elif isinstance(node, Slice):
            base = self._shape(node.base.name)
            ax = _norm_axis(node.axis, len(base))
            bsz = base[ax]
            start = _resolve_bound(node.start, bsz)
            stop = _resolve_bound(node.stop, bsz)
            sh = base[:ax] + (stop - start,) + base[ax + 1 :]
        elif isinstance(node, Concat):
            a = self._shape(node.a.name)
            b = self._shape(node.b.name)
            ax = _norm_axis(node.axis, len(a))
            sh = a[:ax] + (a[ax] + b[ax],) + a[ax + 1 :]
        elif isinstance(node, Unsqueeze):
            base = self._shape(node.base.name)
            ax = _norm_axis(node.axis, len(base) + 1)  # axis is in the UNsqueezed rank
            sh = base[:ax] + (1,) + base[ax:]
        elif isinstance(node, Gather):
            base = self._shape(node.base.name)
            idx = self._shape(node.index.name)
            ax = _norm_axis(node.axis, len(base))
            sh = base[:ax] + idx + base[ax + 1 :]  # N-D index: full idx shape replaces axis
        elif isinstance(node, Pointwise):
            arg_shapes = [self._shape(a.name) for a in node.args]
            sh = tuple(max(s[i] for s in arg_shapes) for i in range(len(arg_shapes[0])))
        else:
            sh = self._resolve_raw(self.body.ir.tensors[name].shape)
        self._shape_cache[name] = sh
        return sh

    @staticmethod
    def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * shape[i + 1]
        return tuple(st)

    def _tl_dtype(self, dtype: str) -> str:
        return _TL_DTYPE[self.out_dtype if dtype == "__OUT_DTYPE__" else dtype]

    def _signature(self) -> list[str]:
        sig = [f"{n}_ptr" for n in (*self.body.in_decls, *self.body.out_decls)]
        sig.append("numel")
        sig.append("BLOCK: tl.constexpr")
        return sig

    def kernel_source(self) -> str:
        out0 = next(iter(self.body.out_decls))
        out_shape = self._shape(out0)
        R = len(out_shape)
        L = self.lines
        L.append("    pid = tl.program_id(0)")
        L.append("    offs = pid * BLOCK + tl.arange(0, BLOCK)")
        L.append("    mask = offs < numel")
        # Decompose the flat output offset into per-axis coords (row-major:
        # the last axis is contiguous). Dims are baked as constexprs so the
        # integer divides/modulos lower to shifts where possible.
        cvar = [f"c{a}" for a in range(R)]
        L.append("    _rem = offs")
        for a in range(R - 1, -1, -1):
            L.append(f"    {cvar[a]} = _rem % {out_shape[a]}")
            if a > 0:
                L.append(f"    _rem = _rem // {out_shape[a]}")
        # Emit each Store's value tree (the coord map for the output itself is
        # the per-axis coord vars).
        for node in self.body.ir.nodes:
            if isinstance(node, Store):
                val = self._emit_value(node.val.name, cvar)
                L.append(f"    tl.store({node.ref.name}_ptr + offs, {val}, mask=mask)")
        src = "import triton\nimport triton.language as tl\n"
        if self._needs_libdevice:
            src += "from triton.language.extra import libdevice\n"
        src += "\n\n"
        src += f"@triton.jit\ndef _kernel({', '.join(self._signature())}):\n"
        src += "\n".join(L) + "\n"
        return src

    def _emit_value(self, name: str, coord: list[str]) -> str:
        """Emit (and memoize) the per-lane value of tensor ``name`` at ``coord``.

        ``coord[a]`` is a Triton expr for this tensor's axis-``a`` coordinate,
        written in terms of the output coord vars ``c0..c{R-1}``. Returns the name
        of the register var holding the value.
        """
        key = (name, tuple(coord))
        if key in self._memo:
            return self._memo[key]
        node = self._producer(name)
        v = self._emit_node(node, coord)
        self._memo[key] = v
        return v

    def _producer(self, name: str) -> MathNode:
        for n in self.body.ir.nodes:
            out = getattr(n, "out", None)
            if out is not None and out.name == name:
                return n
            if isinstance(n, Load) and n.ref.name == name:
                return n
        raise KeyError(f"no producer for tensor {name!r}")

    def _offset(self, name: str, coord: list[str]) -> str:
        """Per-lane flat offset to read ``name`` at ``coord`` (broadcast via % dim)."""
        shape = self._shape(name)
        stride = self._strides(shape)
        terms = [
            f"(({coord[a]}) % {shape[a]}) * {stride[a]}" for a in range(len(shape))
        ]
        return " + ".join(terms) if terms else "0"

    def _emit_node(self, node: MathNode, coord: list[str]) -> str:
        if isinstance(node, Load):
            off = self._offset(node.ref.name, coord)
            v = self.fresh(node.ref.name)
            self.lines.append(
                f"    {v} = tl.load({node.ref.name}_ptr + {off}, mask=mask, other=0)"
            )
            return v
        if isinstance(node, Gather):
            # N-D index gather (in-place index_select placement): the gathered
            # axis is replaced by the index's FULL shape. For output coord c[..]:
            #   * leading base axes (before `axis`) map 1:1 to c[..]
            #   * the n_idx coords starting at `axis` address the INDEX tensor
            #     -> load idx_val (the gathered base-axis coordinate)
            #   * trailing base axes map to c[shifted by (n_idx-1)]
            idx_shape = self._shape(node.index.name)
            base_shape = self._shape(node.base.name)
            ax = _norm_axis(node.axis, len(base_shape))
            n_idx = len(idx_shape)
            idx_coord = list(coord[ax : ax + n_idx])
            idx_var = self._emit_value(node.index.name, idx_coord)
            base_stride = self._strides(base_shape)
            terms = []
            bi = 0  # base axis
            ci = 0  # output coord index
            while bi < ax:  # leading base axes
                terms.append(f"(({coord[ci]}) % {base_shape[bi]}) * {base_stride[bi]}")
                bi += 1
                ci += 1
            terms.append(f"{idx_var} * {base_stride[bi]}")  # gathered axis
            bi += 1
            ci += n_idx  # skip the index's coords
            while bi < len(base_shape):  # trailing base axes
                terms.append(f"(({coord[ci]}) % {base_shape[bi]}) * {base_stride[bi]}")
                bi += 1
                ci += 1
            off = " + ".join(terms) if terms else "0"
            v = self.fresh("g")
            self.lines.append(
                f"    {v} = tl.load({node.base.name}_ptr + {off}, mask=mask, other=0)"
            )
            return v
        if isinstance(node, Slice):
            ax = _norm_axis(node.axis, len(coord))
            base_size = self._shape(node.base.name)[ax]
            start = _resolve_bound(node.start, base_size)
            base_coord = list(coord)
            base_coord[ax] = f"({coord[ax]} + {start})"
            return self._emit_value(node.base.name, base_coord)
        if isinstance(node, Unsqueeze):
            ax = _norm_axis(node.axis, len(coord) + 1)  # axis is in the UNsqueezed rank
            base_coord = [c for i, c in enumerate(coord) if i != ax]
            return self._emit_value(node.base.name, base_coord)
        if isinstance(node, Concat):
            ax = _norm_axis(node.axis, len(coord))
            len_a = self._shape(node.a.name)[ax]
            va = self._emit_value(node.a.name, list(coord))
            b_coord = list(coord)
            b_coord[ax] = f"({coord[ax]} - {len_a})"
            vb = self._emit_value(node.b.name, b_coord)
            v = self.fresh("cat")
            self.lines.append(
                f"    {v} = tl.where({coord[ax]} < {len_a}, {va}, {vb})"
            )
            return v
        if isinstance(node, Pointwise):
            return self._emit_pointwise(node, coord)
        raise NotImplementedError(f"multidim node {type(node).__name__}")

    def _emit_pointwise(self, node: Pointwise, coord: list[str]) -> str:
        args = [self._emit_value(a.name, list(coord)) for a in node.args]
        v = self.fresh("p")
        a0 = args[0]
        fn = node.fn
        L = self.lines
        if fn == "cast":
            L.append(f"    {v} = {a0}.to({self._tl_dtype(node.out_dtype)})")
        elif fn == "mul":
            L.append(f"    {v} = {a0} * {args[1]}")
        elif fn == "add":
            L.append(f"    {v} = {a0} + {args[1]}")
        elif fn == "sub":
            L.append(f"    {v} = {a0} - {args[1]}")
        elif fn == "div":
            L.append(f"    {v} = {a0} / {args[1]}")
        elif fn == "neg":
            L.append(f"    {v} = -{a0}")
        elif fn == "abs":
            L.append(f"    {v} = tl.abs({a0})")
        elif fn == "exp":
            L.append(f"    {v} = tl.exp({a0})")
        elif fn == "tanh":
            self._needs_libdevice = True
            L.append(f"    {v} = libdevice.tanh({a0})")
        elif fn == "sigmoid":
            L.append(f"    {v} = tl.sigmoid({a0})")
        elif fn == "sqrt":
            L.append(f"    {v} = tl.sqrt({a0})")
        elif fn == "rsqrt":
            L.append(f"    {v} = tl.rsqrt({a0})")
        elif fn == "silu":
            L.append(f"    {v} = {a0} * tl.sigmoid({a0})")
        elif fn == "gelu":  # tanh approx (libdevice.tanh — Triton 3.7+ has no tl.tanh)
            self._needs_libdevice = True
            L.append(
                f"    {v} = 0.5 * {a0} * (1.0 + libdevice.tanh("
                f"0.7978845608028654 * ({a0} + 0.044715 * {a0} * {a0} * {a0})))"
            )
        elif fn == "min":
            L.append(f"    {v} = tl.minimum({a0}, {args[1]})")
        elif fn == "max":
            L.append(f"    {v} = tl.maximum({a0}, {args[1]})")
        elif fn == "where":
            L.append(f"    {v} = tl.where({a0}, {args[1]}, {args[2]})")
        else:
            raise NotImplementedError(f"multidim pointwise {fn!r}")
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# §5  The launchers: compile (cached, real-file-backed) + launch by pattern
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_CACHE: dict[tuple[int, str, str], Any] = {}
_GEN_DIR: Path | None = None


def _generated_dir() -> Path:
    global _GEN_DIR
    if _GEN_DIR is None:
        import tempfile
        from pathlib import Path as _Path

        _GEN_DIR = _Path(tempfile.gettempdir()) / "xkernels_vkl_gen"
        _GEN_DIR.mkdir(parents=True, exist_ok=True)
    return _GEN_DIR


def _get_kernel(body: MathBody, out_dtype: str, *, pattern: str):
    """Compile (and cache) the ``@triton.jit`` kernel for this IR + dtype + pattern."""
    key = (id(body), out_dtype, pattern)
    if key not in _KERNEL_CACHE:
        if pattern == "tiled_2d":
            gen = _TritonGen(body, out_dtype)
        elif pattern == "rowwise":
            gen = _TritonGenRowwise(body, out_dtype)
        else:
            gen = _TritonGenElementwise(body, out_dtype)
        src = gen.kernel_source()
        path = (
            _generated_dir()
            / f"mathbody_{pattern}_{abs(hash((id(body), out_dtype, pattern)))}.py"
        )
        path.write_text(src)
        ns: dict[str, Any] = {"__name__": path.stem}
        exec(compile(src, str(path), "exec"), ns)  # noqa: S102
        _KERNEL_CACHE[key] = ns["_kernel"]
    return _KERNEL_CACHE[key]


def _symbol_values(body: MathBody, inputs: dict[str, torch.Tensor]) -> dict[str, int]:
    """Map each decl subscript symbol to its concrete dim value.

    Input shapes are the primary binding source. Some valid addressing kernels
    produce an output dimension that is a static slice of an input dimension
    (for example packed ``[M, 2K] -> [M, K]`` activation gates), so output-only
    symbols are filled from the concrete shape of the value stored to that output.
    """
    vals: dict[str, int] = {}
    for name, ref in body.in_decls.items():
        t = inputs.get(name)
        if t is None:
            continue
        for axis, sym in enumerate(ref.subscript):
            vals.setdefault(sym, t.shape[axis])
    for node in body.ir.nodes:
        if not isinstance(node, Store):
            continue
        out_ref = node.ref
        try:
            val_shape = _concrete_shape(body, node.val.name, vals, inputs)
        except KeyError:
            continue
        for axis, sym in enumerate(out_ref.subscript):
            if axis < len(val_shape):
                vals.setdefault(sym, val_shape[axis])
    return vals


def _concrete_shape(
    body: MathBody,
    name: str,
    sym_vals: dict[str, int],
    inputs: dict[str, torch.Tensor],
) -> tuple[int, ...]:
    """Derive a tensor's concrete shape from the math IR and known input symbols."""
    cache: dict[str, tuple[int, ...]] = {}
    producers: dict[str, MathNode | _LitMarker | _DimRefMarker] = {}
    for node in body.ir.nodes:
        if isinstance(node, Load):
            producers[node.ref.name] = node
        elif isinstance(node, Store):
            producers[node.ref.name] = node
        elif isinstance(node, (_LitMarker, _DimRefMarker)):
            producers[node.name] = node
        elif hasattr(node, "out"):
            producers[node.out.name] = node  # type: ignore[attr-defined]

    def resolve_raw(raw: tuple[int | str, ...]) -> tuple[int, ...]:
        return tuple(d if isinstance(d, int) else int(sym_vals[d]) for d in raw)

    def shape_of(n: str) -> tuple[int, ...]:
        if n in cache:
            return cache[n]
        if n in inputs:
            sh = tuple(inputs[n].shape)
        elif n in body.in_decls:
            sh = resolve_raw(body.in_decls[n].shape)
        elif n in body.out_decls and n not in producers:
            sh = resolve_raw(body.out_decls[n].shape)
        else:
            node = producers[n]
            if isinstance(node, (Load, _LitMarker, _DimRefMarker)):
                sh = resolve_raw(body.ir.tensors[n].shape)
            elif isinstance(node, Store):
                sh = shape_of(node.val.name)
            elif isinstance(node, MMA):
                a = shape_of(node.a.name)
                b = shape_of(node.b.name)
                sh = a[:-1] + b[-1:]
            elif isinstance(node, Reduce):
                x = shape_of(node.x.name)
                ax = _norm_axis(node.axis, len(x))
                sh = x[:ax] + (1,) + x[ax + 1 :]
            elif isinstance(node, Slice):
                base = shape_of(node.base.name)
                ax = _norm_axis(node.axis, len(base))
                start = _resolve_bound(node.start, base[ax])
                stop = _resolve_bound(node.stop, base[ax])
                sh = base[:ax] + (stop - start,) + base[ax + 1 :]
            elif isinstance(node, Concat):
                a = shape_of(node.a.name)
                b = shape_of(node.b.name)
                ax = _norm_axis(node.axis, len(a))
                sh = a[:ax] + (a[ax] + b[ax],) + a[ax + 1 :]
            elif isinstance(node, Unsqueeze):
                base = shape_of(node.base.name)
                ax = _norm_axis(node.axis, len(base) + 1)
                sh = base[:ax] + (1,) + base[ax:]
            elif isinstance(node, Gather):
                base = shape_of(node.base.name)
                idx = shape_of(node.index.name)
                ax = _norm_axis(node.axis, len(base))
                sh = base[:ax] + idx + base[ax + 1 :]
            elif isinstance(node, Pointwise):
                arg_shapes = [shape_of(a.name) for a in node.args]
                sh = tuple(torch.broadcast_shapes(*arg_shapes))
            else:  # pragma: no cover - defensive for new IR nodes
                sh = resolve_raw(body.ir.tensors[n].shape)
        cache[n] = sh
        return sh

    return shape_of(name)


# The canonical Triton tile / launch-metaparam knob names (the contract between
# a ``Target.knobs`` declaration and the tiled lowering — docs/brainstorm/10 §5).
# A target may declare any subset; the lowering falls back to defaults for the
# rest. ``num_warps``/``num_stages`` are Triton launch metas (not ``tl.constexpr``);
# ``BLOCK_*`` are kernel ``tl.constexpr`` (Triton recompiles per value, cached).
TRITON_TILE_KNOBS: tuple[str, ...] = ("BLOCK_M", "BLOCK_N", "BLOCK_K")
TRITON_META_KNOBS: tuple[str, ...] = ("num_warps", "num_stages")


def launch(
    body: MathBody,
    inputs: dict[str, torch.Tensor],
    out_dtype: str,
    *,
    pattern: str,
    **knobs: int,
) -> dict[str, torch.Tensor]:
    """Lower + launch the math IR per the launch ``pattern`` (tiled_2d | rowwise).

    ``knobs`` carries the active specialization binding (the schedule IR's current
    ``Knob`` values) — ``BLOCK_M/N/K`` recompile the tiled kernel's ``tl.constexpr``
    tiles; ``num_warps``/``num_stages`` are Triton launch metas. Phase 2.0a/2.0b
    ignored these (hardcoded defaults); Phase 2.2a makes them live, so a
    ``verify(impl_card_id, knobs={...})`` sweep actually retargets the kernel —
    the substrate's ``autotune-knob-sweep`` skill driven by the schedule IR.
    """
    if pattern == "tiled_2d":
        return _launch_tiled_2d(body, inputs, out_dtype, **knobs)
    if pattern == "rowwise":
        return _launch_rowwise(body, inputs, out_dtype, **knobs)
    if pattern == "elementwise":
        if _has_addressing(body):
            return _launch_multidim(body, inputs, out_dtype, **knobs)
        return _launch_elementwise(body, inputs, out_dtype, **knobs)
    raise NotImplementedError(f"launch pattern {pattern!r}")


def _meta_kwargs(knobs: dict[str, int]) -> dict[str, int]:
    """Pull ``num_warps``/``num_stages`` from knobs; omit when unset (Triton auto-picks)."""
    meta: dict[str, int] = {}
    for k in TRITON_META_KNOBS:
        if k in knobs:
            meta[k] = knobs[k]
    return meta


def _output_dtype(body: MathBody, out_name: str, out_dtype: str) -> str:
    """Resolve a per-output dtype from the IR (honors explicit casts like fp8).

    The math IR's Store node carries its stored value's TensorRef; the dtype on
    that ref is what an explicit ``.cast("fp8")`` produced (a literal short
    name), or the ``__OUT_DTYPE__`` sentinel if the body used ``ctx.out_dtype()``
    (resolved to the launch's global out_dtype). This is what lets an op emit
    MIXED-dtype outputs (e.g. per-token-group fp8 quant: ``q`` fp8 + ``scale``
    fp32, issue #57) without a single global output dtype.
    """
    for node in body.ir.nodes:
        if isinstance(node, Store) and node.ref.name == out_name:
            dt = body.ir.tensors[node.val.name].dtype
            return out_dtype if dt == "__OUT_DTYPE__" else dt
    return out_dtype


def _launch_tiled_2d(
    body: MathBody,
    inputs: dict[str, torch.Tensor],
    out_dtype: str,
    **knobs: int,
) -> dict[str, torch.Tensor]:
    """Run the generated tiled Triton kernel; return the outputs (tiled_2d grid).

    Tile sizes come from the knob binding (``BLOCK_M/N/K``) with the Phase 2.0a
    defaults as fallback; ``num_warps``/``num_stages`` are passed as Triton launch
    metas only when the binding sets them.
    """
    kernel = _get_kernel(body, out_dtype, pattern="tiled_2d")
    mma = _find_mma(body.ir.nodes)
    r = _dim_roles(mma, body)
    out_name = next(iter(body.out_decls))
    a, b = mma.a.name, mma.b.name
    dev = next(iter(inputs.values())).device
    M = inputs[a].shape[list(mma.a.subscript).index(r["m"])]
    N = inputs[b].shape[list(mma.b.subscript).index(r["n"])]
    K = inputs[a].shape[list(mma.a.subscript).index(r["k"])]
    dt = to_torch_dtype(out_dtype)
    out = torch.empty(M, N, dtype=dt, device=dev)
    bm = knobs.get("BLOCK_M", _TritonGen.BLOCK_M)
    bn = knobs.get("BLOCK_N", _TritonGen.BLOCK_N)
    bk = knobs.get("BLOCK_K", _TritonGen.BLOCK_K)
    grid = ((M + bm - 1) // bm, (N + bn - 1) // bn)
    kernel[grid](
        inputs[a],
        inputs[a].stride()[0],
        inputs[a].stride()[1],
        inputs[b],
        inputs[b].stride()[0],
        inputs[b].stride()[1],
        out,
        out.stride()[0],
        out.stride()[1],
        M,
        N,
        K,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        **_meta_kwargs(knobs),
    )
    return {out_name: out}


def _launch_rowwise(
    body: MathBody,
    inputs: dict[str, torch.Tensor],
    out_dtype: str,
    **knobs: int,
) -> dict[str, torch.Tensor]:
    """Run the generated row-wise Triton kernel; return the outputs (1D grid=T).

    Allocates outputs at real shape (``[T, d]``), derives the row strides + the
    per-symbol BLOCK constexprs (``next_pow2(d)``) + dim args from the input
    shapes, and launches ``grid=(T,)``.
    """
    kernel = _get_kernel(body, out_dtype, pattern="rowwise")
    gen = _TritonGenRowwise(body, out_dtype)
    gen._collect_symbols()  # rebuild the symbol order the source used
    sym_vals = _symbol_values(body, inputs)
    T = next(t.shape[0] for t in inputs.values() if t.dim() == 2)
    dev = next(iter(inputs.values())).device
    out_tensors: dict[str, torch.Tensor] = {}
    for name, ref in body.out_decls.items():
        rank = len(ref.shape)
        d = sym_vals[ref.subscript[rank - 1]]
        odt = to_torch_dtype(_output_dtype(body, name, out_dtype))
        out_tensors[name] = (
            torch.empty(T, d, dtype=odt, device=dev)
            if rank == 2
            else torch.empty(d, dtype=odt, device=dev)
        )
    args: list[Any] = []
    for name in (*body.in_decls, *body.out_decls):
        t = inputs[name] if name in inputs else out_tensors[name]
        args.append(t)
        ref = body.in_decls.get(name) or body.out_decls[name]
        if len(ref.shape) == 2:
            args.append(t.stride(0))
    block_kwargs: dict[str, int] = {}
    for sym in gen.symbols:  # signature order (deterministic)
        d = sym_vals[sym]
        args.append(d)
        block_kwargs[gen.symbols[sym]["block"]] = _next_pow2(d)
    kernel[(T,)](*args, **block_kwargs, **_meta_kwargs(knobs))
    return out_tensors


def _launch_elementwise(
    body: MathBody,
    inputs: dict[str, torch.Tensor],
    out_dtype: str,
    **knobs: int,
) -> dict[str, torch.Tensor]:
    """Run the generated flat-1D Triton kernel; return the outputs (1D grid).

    Pure-pointwise over the flattened output: one program per ``BLOCK``-sized
    tile of ``numel = prod(output shape)``. Every input/output is addressed as a
    flat 1D buffer (row-major), so all operands must share the output's shape.
    Each output is allocated at its own IR-resolved dtype (``_output_dtype``).
    """
    kernel = _get_kernel(body, out_dtype, pattern="elementwise")
    sym_vals = _symbol_values(body, inputs)
    out0 = next(iter(body.out_decls))
    numel = 1
    for sym in body.out_decls[out0].subscript:
        numel *= sym_vals[sym]
    dev = next(iter(inputs.values())).device
    out_tensors: dict[str, torch.Tensor] = {}
    for name, ref in body.out_decls.items():
        shape = tuple(sym_vals[s] for s in ref.subscript)
        odt = to_torch_dtype(_output_dtype(body, name, out_dtype))
        out_tensors[name] = torch.empty(shape, dtype=odt, device=dev)
    args: list[Any] = []
    for name in (*body.in_decls, *body.out_decls):
        args.append(inputs[name] if name in inputs else out_tensors[name])
    args.append(numel)
    block = knobs.get("BLOCK", _TritonGenElementwise.BLOCK)
    grid = ((numel + block - 1) // block,)
    kernel[grid](*args, BLOCK=block, **_meta_kwargs(knobs))
    return out_tensors


_MULTIDIM_KERNEL_CACHE: dict[tuple[int, str, tuple[int, ...]], Any] = {}


def _launch_multidim(
    body: MathBody,
    inputs: dict[str, torch.Tensor],
    out_dtype: str,
    **knobs: int,
) -> dict[str, torch.Tensor]:
    """Run the generated multi-dim addressing kernel; return the outputs.

    Same flat-tiled grid as elementwise (one program per ``BLOCK``-sized tile of
    the flattened output), but the kernel decomposes each lane's flat offset
    into per-axis coordinates and each addressing node computes its own address
    (``_TritonGenMultiDim``). The output dims are baked as ``tl.constexpr`` (the
    coord-decomposition divides by them), so the cache is keyed by the resolved
    OUTPUT SHAPE — a different shape recompiles, exactly like a ``BLOCK`` change.
    """
    sym_vals = _symbol_values(body, inputs)
    out0 = next(iter(body.out_decls))
    out_shape = tuple(sym_vals[s] for s in body.out_decls[out0].subscript)
    numel = 1
    for s in out_shape:
        numel *= s
    sig = (id(body), out_dtype, out_shape)
    if sig not in _MULTIDIM_KERNEL_CACHE:
        gen = _TritonGenMultiDim(body, out_dtype, sym_vals)
        src = gen.kernel_source()
        path = _generated_dir() / f"mathbody_multidim_{abs(hash(sig))}.py"
        path.write_text(src)
        ns: dict[str, Any] = {"__name__": path.stem}
        exec(compile(src, str(path), "exec"), ns)  # noqa: S102
        _MULTIDIM_KERNEL_CACHE[sig] = ns["_kernel"]
    kernel = _MULTIDIM_KERNEL_CACHE[sig]
    dev = next(iter(inputs.values())).device
    out_tensors: dict[str, torch.Tensor] = {}
    for name, ref in body.out_decls.items():
        shape = tuple(sym_vals[s] for s in ref.subscript)
        odt = to_torch_dtype(_output_dtype(body, name, out_dtype))
        out_tensors[name] = torch.empty(shape, dtype=odt, device=dev)
    args: list[Any] = []
    for name in (*body.in_decls, *body.out_decls):
        args.append(inputs[name] if name in inputs else out_tensors[name])
    args.append(numel)
    block = knobs.get("BLOCK", _TritonGenMultiDim.BLOCK)
    grid = ((numel + block - 1) // block,)
    kernel[grid](*args, BLOCK=block, **_meta_kwargs(knobs))
    return out_tensors
