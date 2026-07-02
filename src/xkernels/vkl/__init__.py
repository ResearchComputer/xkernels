# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""``xkernels.vkl`` — the contract-native kernel-authoring DSL (docs/brainstorm/).

A higher-level, multi-target authoring layer that sits ABOVE the existing
registry substrate. The contract (Op Spec) remains the product; the DSL is a
*spelling* of it, not a gatekeeper (docs/brainstorm/02 §1, §10 anti-goals).

Phase 1 (CPU-satisfiable, this package) delivers:
  * ``@kernel`` / ``@targets`` authoring surface (``surface.py``)
  * the auto-reference: the ``@kernel`` body run on torch (``reference.py``)
  * the emitter: header → schema-valid Op Spec + Impl Card JSON (``emit.py``)
  * the edit gate: ``SetKnob`` / ``Retile`` with locally-decidable checks (``edits.py``,
    ``gate.py``) — the programmatic autotune primitive

The GPU-gated path (per-program tiling, Triton/CUDA/HIP lowering, ``verify``
end-to-end) is Phase 1.5 / Phase 2 (docs/brainstorm/11 §2). On CPU we test the
contract round-trip + auto-reference equivalence + edit decidability — the three
Phase 1 gates.

Importing this package is side-effect-free (no registry mutation, no emission).
"""
from __future__ import annotations

from . import archdb, auto, cost, edits, emit, gate, override, reference, sweep, tiles
from .cost import (
    GateVerdict,
    Occupancy,
    Roofline,
    occupancy,
    overflows_scratch,
    predict_scratch,
    roofline,
    roofline_gate,
    workload,
)
from .edits import AddStage, MapTo_, Ok, Reject, Retile, SetKnob
from .emit import emit_card, emit_reference_card, emit_spec
from .gate import GateResult, TraceEntry, run_gate
from .graph import (
    CapturedGraph,
    GraphCtx,
    GraphPerf,
    GraphSpec,
    capture,
    graph,
    graph_of,
    measure,
    register_graph_node,
    run_graph,
)
from .ir import (
    MMA,
    CopyAtom,
    Knob,
    Load,
    MapTo,
    MathIR,
    Pointwise,
    Reduce,
    ScheduleIR,
    Stage,
    Store,
    TensorRef,
    Tile,
)
from .lower import cuda as lower_cuda
from .lower import triton as lower_triton
from .lower.cuda import lower_to_cuda, register_dsl_cuda
from .lower.triton import lower_to_triton, register_dsl
from .override import OverrideCheck, check_override_math_ir, emit_override_card
from .reference import make_inputs, run_reference, trace_ir
from .surface import (
    AUTO_REFERENCE,
    KernelSpec,
    Launch,
    Numerics,
    OverrideBody,
    Target,
    TensorDecl,
    kernel,
    launch,
    spec_of,
    targets,
)
from .sweep import SweepResult, autotune, enumerate_configs, schedule_from_card

__all__ = [
    # authoring surface
    "kernel", "targets", "launch", "spec_of", "KernelSpec", "Target", "TensorDecl",
    "Numerics", "Launch", "AUTO_REFERENCE", "OverrideBody",
    # emit (header -> JSON)
    "emit_spec", "emit_card", "emit_reference_card",
    # reference + lowering (body -> torch / Triton)
    "make_inputs", "run_reference", "trace_ir", "lower_to_triton", "register_dsl",
    "lower_triton", "lower_to_cuda", "register_dsl_cuda", "lower_cuda",
    # graphs (Phase 3: capture a composition into one CUDA/HIP graph launch)
    "graph", "graph_of", "GraphSpec", "GraphCtx", "capture", "run_graph",
    "CapturedGraph", "measure", "GraphPerf", "register_graph_node",
    # math IR (frozen oracle)
    "TensorRef", "Load", "Reduce", "MMA", "Pointwise", "Store", "MathIR",
    # schedule IR (editable)
    "Tile", "MapTo", "Stage", "CopyAtom", "Knob", "ScheduleIR",
    # edits + gate
    "SetKnob", "Retile", "MapTo_", "AddStage", "Ok", "Reject",
    "run_gate", "GateResult", "TraceEntry",
    # modules
    "archdb", "auto", "cost", "edits", "emit", "gate", "override", "reference", "sweep", "tiles",
    # autotune sweep (Phase 2.2)
    "autotune", "SweepResult", "enumerate_configs", "schedule_from_card",
    # cost model + Phase 2 roofline gate (Phase 2.3)
    "workload", "predict_scratch", "overflows_scratch", "roofline", "occupancy",
    "roofline_gate", "Roofline", "Occupancy", "GateVerdict",
    # per-target override bodies (Phase 2.1 mechanism)
    "check_override_math_ir", "emit_override_card", "OverrideCheck",
]
