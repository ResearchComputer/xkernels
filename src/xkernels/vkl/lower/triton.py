# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Triton lowering dispatch (docs/brainstorm/04 Ex.1, ``11`` §2).

Entry point: ``lower_to_triton(spec) -> callable``. Dispatches on the spec's
``launch.pattern`` to the matching ``mathbody`` lowering: ``tiled_2d`` (the GEMM
2D-grid + K-loop) or ``rowwise`` (the one-program-per-row reduction, dual_rmsnorm).
Both lower the SAME math IR (``lower/mathbody.py``) — Phase 2.0b collapsed the
Phase 1.5 bespoke row-reduce IR into the doc-10 math IR.

The returned callable is a host launcher that allocates outputs, computes
strides/grid, and launches the generated ``@triton.jit`` kernel. The kernel is
compiled lazily (cached by ``(trace, dtype, pattern)`` — Triton recompiles per
dtype anyway). This is the path that closes the docs/brainstorm/04 Ex.1 loop:
one ``@kernel`` source → a registered Triton callable that ``verify`` can run,
with zero hand-editing of JSON.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ..._backends import Backend
from ..._dispatch import register
from ...registry.dtypes import to_short_dtype
from ..reference import trace_ir
from ..surface import KernelSpec


def lower_to_triton(spec: KernelSpec) -> Callable[..., Any]:
    """Return a host launcher that runs the spec's generated Triton kernel.

    The launcher's signature matches the spec's INPUT order (so it is callable
    like the hand-written ``dual_rmsnorm_triton(x1, w1, x2, w2, eps=...)``).
    Outputs are returned as a tuple in the spec's output order.
    """
    if spec.launch is None:
        raise ValueError(
            f"{spec.id}: no @launch declared — cannot lower to Triton "
            f"(the body is a direct torch computation, not a trace)"
        )
    pattern = spec.launch.pattern
    if pattern not in ("rowwise", "tiled_2d", "elementwise"):
        raise NotImplementedError(
            f"launch pattern {pattern!r} not yet supported "
            f"(Phase 2.0b: 'rowwise' + 'tiled_2d'; 2.0c: 'elementwise')"
        )

    ir = trace_ir(spec)
    input_names = tuple(spec.inputs)
    # The knob names this target declares (the autotune search space). Anything
    # else in **kwargs that isn't an input or a declared knob is silently dropped
    # (verify passes requested-but-unaccepted knobs back as not-applied).
    tgt = spec.targets.get("triton")
    knob_names: set[str] = set(tgt.knobs) if tgt is not None else set()

    def launcher(*args: torch.Tensor, **kwargs: Any) -> Any:
        # Accept EITHER positional (in input order) OR keyword (``fn(**inputs)``,
        # the form ``verify`` / ``dispatch`` / ``generate_inputs`` use). Bind to
        # input names; require contiguous row-major. Knob kwargs (BLOCK_M, ...) are
        # separated out and threaded to the lowering as the active specialization
        # binding — the substrate's ``verify(knobs=...)`` autotune path.
        inputs: dict[str, torch.Tensor] = {}
        knobs: dict[str, int] = {}
        if args:
            for name, val in zip(input_names, args, strict=True):
                inputs[name] = val.contiguous()
        for name, val in kwargs.items():
            if name in spec.inputs:
                inputs[name] = val.contiguous()
            elif name in knob_names:
                knobs[name] = int(val)
        missing = set(spec.inputs) - set(inputs)
        if missing:
            raise TypeError(f"missing required inputs: {sorted(missing)}")
        out_dtype = to_short_dtype(next(iter(inputs.values())).dtype)
        from . import mathbody

        outs = mathbody.launch(ir, inputs, out_dtype, pattern=pattern, **knobs)
        return tuple(outs[name] for name in spec.outputs)

    return launcher


def register_dsl(spec: KernelSpec, backend: str = "triton") -> Callable[..., Any]:
    """Lower the spec to ``backend`` and register it with the dispatch registry.

    After this, ``verify(spec.targets[backend]...)`` and
    ``dispatch(spec.kernel, backend=backend)`` run the DSL-generated callable —
    the docs/brainstorm/04 Ex.1 loop closed: one ``@kernel`` source → a
    registered callable that the unchanged substrate can run, with zero JSON
    hand-editing.

    NB: registering under ``(spec.kernel, backend)`` supersedes any prior
    registration for that pair (the hand-written one). The DSL kernel satisfies
    the SAME op spec + reference, so ``verify``'s correctness gate still holds.
    """
    if backend != "triton":
        raise NotImplementedError(f"backend {backend!r} not yet lowered (Phase 1.5: triton)")
    launcher = lower_to_triton(spec)
    register(spec.kernel, Backend.TRITON)(launcher)
    # Phase 3: record this kernel as a capturable graph node so a @graph body's
    # reference-mode ctx.call can resolve its auto-reference.
    from ..graph import register_graph_node

    register_graph_node(spec)
    # Also wire the input generator so ``verify`` is end-to-end runnable with no
    # hand-editing of ``input_gen.py``'s literal dict. The generator delegates to
    # the spec's ``shape_symbols`` via ``vkl.reference.make_inputs``.
    from ...registry.input_gen import register_input_gen
    from ..reference import make_inputs

    def _gen(point, seed, device):
        return make_inputs(spec, point, seed=seed, device=device)

    register_input_gen(spec.id, _gen)
    return launcher
