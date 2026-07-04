# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The schedule-IR spine: spec -> structured schedule -> flat binding (docs/brainstorm/09).

This module is what makes the schedule IR the **source of truth in both
directions** (the Phase A closure of the doc-09 agent-editable-IR thesis):

  * **read-out** (``schedule_from_spec``): given a ``KernelSpec`` + arch, build a
    REAL ``ScheduleIR`` - the ``Tile`` / ``MapTo`` / ``Stage`` / ``Knob`` nodes
    that describe what the default lowering actually does. Phase 1's
    ``schedule_from_card`` built a knob-only bag; this carries the structure the
    edit primitives (``Retile`` / ``MapTo_`` / ``AddStage`` / ``SetMapPolicy``)
    operate on. The schedule is a *description* of the lowering's choices,
    read out from the math IR + the launch pattern + the declared knob space.

  * **read-in** (``resolve_binding``): project a (possibly edited) schedule to
    the flat ``{knob_name: value}`` dict the launcher already consumes - PLUS the
    MMA's ``input_precision`` policy flattened in. So an agent path
    (load_schedule -> check_edit -> apply_edit -> resolve_binding -> launch)
    and the substrate path (``verify(knobs=...)`` -> launch) converge on the SAME
    launcher entry. Edits reach silicon because the binding the launcher reads is
    the binding the schedule resolved to.

The one concrete, agent-editable lever wired end-to-end here is ``input_precision``
(an MMA policy stored on ``MapTo``): an fp32 GEMM's ``ieee``->``tf32`` swap is a
``SetMapPolicy`` edit that changes what ``tl.dot`` compiles to. That is the
doc-09 section 8 "map_to step reaches silicon" proof, made real on the Triton
backend. Tile sizes + ``num_stages`` were already knobs; this adds the policy
lever and the structured IR that carries all of them together.

Pure dataclass logic throughout - no torch, no GPU. The launch that consumes the
binding is GPU-gated (``lower/mathbody.py``); building + editing + resolving the
schedule is CPU-doable and is what an agent reasons over.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from . import archdb
from .ir.math import MMA, MathNode, Reduce
from .ir.schedule import Knob, MapTo, ScheduleIR, Stage, Tile

if TYPE_CHECKING:
    # ``KernelSpec`` lives in the torch-bearing ``surface`` module. Imported only
    # under TYPE_CHECKING so this module's pure projections (``resolve_binding``
    # / ``precision_of``) stay CPU-importable: the edit round-trip is reasoned
    # about without a GPU (docs/brainstorm/09 §0). ``trace_ir`` is still imported
    # lazily inside the functions that actually trace, since tracing needs torch.
    from .surface import KernelSpec

# Knob/value names the Triton lowering recognizes as the MMA's input-precision
# policy (flattened from a MapTo node, not a declared specialization knob).
PRECISION_KEY = "input_precision"

# Matrix-engine instruction preference per arch (the "native" ceiling instr).
# Derived from archdb: for a concrete arch we prefer the newest matrix-engine
# family; for the portable "any" target there is no concrete instruction (Triton
# picks at runtime), so None is the honest default.
_NATIVE_INSTR_PREFERENCE: dict[str, str] = {
    "nvidia_sm90": "wgmma",
    "nvidia_sm121": "wgmma",
    "nvidia_sm80": "wmma",
    "amd_cdna3": "mfma",
    "amd_cdna2": "mfma",
}


def _native_matrix_instr(arch: str) -> str | None:
    """The native L5 matrix-engine instruction for ``arch`` (None if portable)."""
    if arch == "any":
        return None
    pref = _NATIVE_INSTR_PREFERENCE.get(arch)
    if pref and pref in archdb.legal_instructions(arch):
        return pref
    # Fall back to any non-fma instruction the arch advertises.
    legal = [i for i in archdb.legal_instructions(arch) if i != "fma"]
    return legal[0] if legal else None


def _find_mma(nodes: tuple[MathNode, ...]) -> MMA | None:
    mmas = [n for n in nodes if isinstance(n, MMA)]
    return mmas[0] if mmas else None


def _find_reduces(nodes: tuple[MathNode, ...]) -> tuple[Reduce, ...]:
    return tuple(n for n in nodes if isinstance(n, Reduce))


# ═══════════════════════════════════════════════════════════════════════════════
# §1  read-out: spec -> structured ScheduleIR
# ═══════════════════════════════════════════════════════════════════════════════


def schedule_from_spec(spec: KernelSpec, arch: str = "any") -> ScheduleIR:
    """Build a structured ``ScheduleIR`` describing the default lowering of ``spec``.

    The nodes mirror what the launch pattern + math IR + declared knobs produce:
      * ``tiled_2d``  -> an output Tile + two streaming Tiles + an L5 MapTo for
        the MMA + two scratch Stages (depth tracks ``num_stages``) + Knobs.
      * ``rowwise``   -> a Reduce schedule node per math Reduce (wave-level, L3)
        + Knobs.
      * ``elementwise``-> Knobs (no tile structure beyond the flat BLOCK).

    The arch selects the MapTo's ``instruction`` (``None`` for the portable
    ``any`` target; the arch-native matrix-engine instr otherwise). Every
    hardware-naming field is ``None`` or a closed enum - never a free literal -
    so the gate can decide legality from the IR alone (docs/brainstorm/10 section 5).
    """
    if spec.launch is None:
        # Direct-torch body (Phase 1 form): no device schedule to describe.
        return _knobs_only(spec)

    pattern = spec.launch.pattern
    if pattern == "tiled_2d":
        return _schedule_tiled_2d(spec, arch)
    if pattern == "rowwise":
        return _schedule_rowwise(spec, arch)
    if pattern == "elementwise":
        return _knobs_only(spec)
    # Unknown pattern: degrade to the knob bag (honest - structure not modeled).
    return _knobs_only(spec)


def _knobs_only(spec: KernelSpec) -> ScheduleIR:
    """The minimal schedule: one Knob node per declared specialization knob."""
    sched = ScheduleIR()
    for name, choices in _declared_knobs(spec):
        if choices:
            sched = sched.with_node(Knob(name=name, value=choices[0], choices=choices))
    return sched


def _declared_knobs(spec: KernelSpec) -> list[tuple[str, tuple[int, ...]]]:
    """The triton target's declared knob space (name -> choices), if any."""
    tgt = spec.targets.get("triton")
    if tgt is None:
        return []
    return [(name, tuple(choices)) for name, choices in tgt.knobs.items()]


def _schedule_tiled_2d(spec: KernelSpec, arch: str) -> ScheduleIR:
    """The GEMM schedule: output + streaming tiles, an L5 MMA map, K-loop stages."""
    from .reference import trace_ir

    nodes = trace_ir(spec).ir.nodes
    mma = _find_mma(nodes)
    sched = _knobs_only(spec)

    # Tiles (symbolic in knob names - resolved at emit by resolve_binding).
    sched = sched.with_node(Tile(id="out", shape=("BLOCK_M", "BLOCK_N"), level="L2"))
    sched = sched.with_node(Tile(id="a_tile", shape=("BLOCK_M", "BLOCK_K"), level="L2"))
    sched = sched.with_node(Tile(id="b_tile", shape=("BLOCK_K", "BLOCK_N"), level="L2"))

    # The MMA -> L5 map (the heavy op). instruction is arch-native or None
    # (portable); precision is the dtype-default (None) until an edit sets it.
    if mma is not None:
        instr = _native_matrix_instr(arch)
        instr_shape = _native_instr_shape(arch, instr)
        sched = sched.with_node(MapTo(
            id="mma0", op_ref=mma.out.name, level="L5",
            instruction=instr, instr_shape=instr_shape, precision=None,
        ))

    # The K-loop streams both operands through scratch at num_stages depth.
    # depth tracks the knob by NAME (symbolic); resolved at emit.
    sched = sched.with_node(Stage(
        id="stage_a", producer_ref="a_tile", space="scratch", depth="num_stages",
    ))
    sched = sched.with_node(Stage(
        id="stage_b", producer_ref="b_tile", space="scratch", depth="num_stages",
    ))
    return sched


def _schedule_rowwise(spec: KernelSpec, arch: str) -> ScheduleIR:
    """The row-reduce schedule: a wave-level (L3) Reduce node per math Reduce."""
    from .ir.schedule import Reduce as SchedReduce
    from .reference import trace_ir

    nodes = trace_ir(spec).ir.nodes
    sched = _knobs_only(spec)
    for i, red in enumerate(_find_reduces(nodes)):
        sched = sched.with_node(SchedReduce(id=f"reduce{i}", op_ref=red.out.name, level="L3"))
    return sched


def _native_instr_shape(arch: str, instr: str | None) -> tuple[int, ...] | None:
    """The native MMA shape for (arch, instr), as a tuple (m, n_hint, k) or None."""
    if instr is None:
        return None
    native = archdb.native_shape(arch, instr)
    if native is None:
        return None
    return (native["m"], native["k"])


# ═══════════════════════════════════════════════════════════════════════════════
# §2  read-in: schedule -> flat binding (the launcher's input)
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_binding(sched: ScheduleIR) -> dict[str, int | str]:
    """Project a schedule to the flat ``{name: value}`` binding the launcher reads.

    Three sources, merged in this precedence (later wins on conflict):

      1. every ``Knob``'s current value (``BLOCK_M``, ``num_stages``, ...) - the
         tile/meta search space, the values an autotune sweep or a ``SetKnob``
         edit set.
      2. resolved ``Stage`` depths - a stage whose ``depth`` is a knob *name*
         contributes nothing extra (the knob already supplied the value); a
         concrete-int depth (from an ``AddStage`` edit) overrides it. Emitted
         under ``num_stages`` so the launcher's existing ``_meta_kwargs`` reads it.
      3. the ``MapTo`` precision - flattened to ``input_precision`` (omitted when
         None, the dtype-default: the lowering then picks per dtype).

    The result is exactly the shape ``verify(knobs=...)`` passes and the launcher
    (``lower_to_triton``) consumes - so the agent path and the substrate path
    converge. Edits reach silicon because this is the binding the launcher reads.
    """
    binding: dict[str, int | str] = {}
    # (1) knobs
    for name, knob in sched.knobs.items():
        binding[name] = knob.value
    # (2) concrete stage depths override the num_stages knob when an AddStage
    # edit pinned a depth (the symbolic "num_stages"-by-name case is already
    # covered by the knob itself, so it is a no-op here).
    for stage in sched.stages():
        if isinstance(stage.depth, int) and "num_stages" not in binding:
            binding["num_stages"] = stage.depth
    # (3) MMA precision (the agent-editable policy lever)
    for m in sched.maps():
        if m.precision is not None:
            binding[PRECISION_KEY] = m.precision
    return binding


def precision_of(sched: ScheduleIR) -> str | None:
    """The schedule's MMA precision policy (None = dtype-default). Convenience."""
    for m in sched.maps():
        if m.precision is not None:
            return m.precision
    return None


__all__ = [
    "schedule_from_spec",
    "resolve_binding",
    "precision_of",
    "PRECISION_KEY",
]
