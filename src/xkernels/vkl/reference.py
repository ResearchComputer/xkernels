# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Run a ``@kernel`` as the auto-reference on torch (docs/brainstorm/10 §6).

Two body forms are supported (the surface's ``launch`` field selects):

  * **Direct torch body** (``launch is None``, the Phase 1 form): the body takes
    the inputs and returns the outputs. ``run_reference`` calls it directly.
  * **Math-IR trace body** (``launch = Launch.tiled_2d() | Launch.rowwise()``):
    the body takes a ``MathBodyCtx`` and builds the doc-10 math IR
    (``MMA``/``Reduce``/``Pointwise``); ``run_reference`` builds the IR once, then
    evaluates it on torch (launch-agnostic — ``MMA`` -> ``torch.matmul`` in fp32,
    ``Reduce`` -> ``torch.sum``). The Triton lowering (``lower/triton.py``)
    dispatches the SAME math IR per launch pattern (Phase 2.0b: both patterns).

The auto-reference callable registered in ``auto.py`` is ``run_reference`` bound
to the spec, so the emitted ``numerics.reference`` path resolves to this.
"""

from __future__ import annotations

from typing import Any

import torch

from ..registry.dtypes import to_short_dtype, to_torch_dtype
from .ir.math import TensorRef
from .surface import KernelSpec


def make_inputs(
    spec: KernelSpec, point: dict[str, Any], *, seed: int = 0, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    """Materialize seeded sweep-point inputs for a kernel (the default generator).

    Mirrors the substrate's input-generator convention: each input tensor's shape
    is read off the sweep point by ``shape_symbols``; dtype from the point; values
    seeded deterministically.

    Integer inputs (e.g. RoPE ``positions``) are generated as random VALID
    INDICES into the table they gather. The index range upper bound is read from
    the point under the key ``P`` (the table's leading dim) by convention — an op
    whose int input gathers a different-size table should pass its own bound in
    the point (or register a custom generator, first-writer-wins).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    dtype = to_torch_dtype(point["dtype"])
    inputs: dict[str, torch.Tensor] = {}
    for name, decl in spec.inputs.items():
        shape = tuple(point[sym] for sym in decl.symbols)
        if decl.dtype and decl.dtype[0].startswith("int"):
            # An index tensor: random valid indices in [0, P) (default P from point).
            upper = int(point.get("P", point.get(decl.symbols[-1], 16) if decl.symbols else 16))
            upper = max(1, upper)
            t = torch.randint(0, upper, shape, generator=g, device=device)
            inputs[name] = t.to(to_torch_dtype(decl.dtype[0]))
        else:
            t = (torch.rand(shape, generator=g, device=device) * 2 - 1).to(dtype)
            inputs[name] = t
    return inputs


# ═══════════════════════════════════════════════════════════════════════════════
# Decl construction for the math-IR body
# ═══════════════════════════════════════════════════════════════════════════════


def _math_decls_for(spec: KernelSpec) -> tuple[dict[str, TensorRef], dict[str, TensorRef]]:
    """Build the {name: TensorRef} dicts the math-IR body wants.

    ``subscript`` = the declared ``shape_symbols`` (Einstein labels); the lowering
    uses them to tile (the contracted dim of an MMA is the K-loop; the reduced
    dim of a row-wise ``Reduce`` is the program-local tile). ``dtype`` is the
    first declared dtype (a representative — the actual point dtype overrides at
    eval/codegen time). ``shape`` = the symbols verbatim (symbolic; bound later).
    """
    in_decls = {
        n: TensorRef(n, d.dtype[0], tuple(d.symbols), tuple(d.symbols))
        for n, d in spec.inputs.items()
    }
    out_decls = {
        n: TensorRef(n, d.dtype[0], tuple(d.symbols), tuple(d.symbols))
        for n, d in spec.outputs.items()
    }
    return in_decls, out_decls


# ═══════════════════════════════════════════════════════════════════════════════
# Body-IR construction (memoized; KernelSpec is frozen so we key by id(spec))
# ═══════════════════════════════════════════════════════════════════════════════

_MATHBODY_CACHE: dict[int, object] = {}


def _mathbody_of(spec: KernelSpec):
    """Build (and memoize) the math IR for a trace body (tiled_2d or rowwise)."""
    cached = _MATHBODY_CACHE.get(id(spec))
    if cached is not None:
        return cached
    from .lower.mathbody import build_body

    in_decls, out_decls = _math_decls_for(spec)
    body = build_body(spec.body, in_decls, out_decls)
    _MATHBODY_CACHE[id(spec)] = body
    return body


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry points
# ═══════════════════════════════════════════════════════════════════════════════


def run_reference(spec: KernelSpec, inputs: dict[str, torch.Tensor]) -> Any:
    """Run the kernel body on torch — the auto-reference output.

    The reference is the ORACLE: it must be EXACT. We disable TF32 / fp16-reduced
    matmul reductions for the duration of the eval so a GPU ``torch.matmul`` is
    TRUE fp32 (matching the math IR's ``accum_dtype=fp32``), not TF32. This is
    load-bearing: the global ``torch.backends.cuda.matmul.allow_tf32`` defaults
    to ``True`` on CUDA, which would silently make the oracle TF32-class — and a
    true-fp32 KERNEL would then "fail" against it by the TF32 rounding gap
    (~4e-3). The kernel runs with whatever the harness/backends allow; only the
    oracle is pinned exact. (Measured on GB10/sm_121: without this, a true-fp32
    triton kernel diverges from the oracle by 4.26e-3.)

    Returns the outputs in the spec's output order.
    """
    import torch

    _tf32 = torch.backends.cuda.matmul.allow_tf32
    _fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        return _run_reference_inner(spec, inputs)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = _tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = _fp16


def _run_reference_inner(spec: KernelSpec, inputs: dict[str, torch.Tensor]) -> Any:
    if spec.launch is None:
        return spec.body(**inputs)
    out_dtype = to_short_dtype(next(iter(inputs.values())).dtype)
    from .lower.mathbody import eval_torch

    body = _mathbody_of(spec)
    out_map = eval_torch(body, inputs, out_dtype)
    return tuple(out_map[name] for name in spec.outputs)


def trace_ir(spec: KernelSpec):
    """Public accessor for the trace body's math IR (``None`` for direct bodies).

    Returns a ``MathBody`` for any trace body (tiled_2d or rowwise). The lowering
    (``lower/triton.py``) dispatches the SAME ``MathBody`` per ``spec.launch.pattern``.
    """
    if spec.launch is None:
        return None
    return _mathbody_of(spec)
