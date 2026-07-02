# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Edit primitives + the check gate (docs/brainstorm/10 §3, §5).

Each edit is a frozen dataclass with two methods:
  * ``check(ir, arch) -> Result``  — locally decidable preconditions (no running
    code; a pure function of *edit args + current IR + arch*). This is the bet
    that makes an LLM agent a reliable editor (docs/brainstorm/09 §0): the agent
    can predict whether an edit is legal *before* applying it.
  * ``apply(ir) -> ScheduleIR``    — returns a NEW frozen IR (no in-place mutation;
    the ``tuning_trace`` is a chain of immutable snapshots).

Phase 1 implements the two cheapest primitives (``SetKnob``, ``Retile``); the
matrix-engine edits (``MapTo_``, ``AddStage``, ...) land in Phase 2
(docs/brainstorm/11 §2). The oracle property holds by construction: edits
operate on ``ScheduleIR`` only, and the math IR lives on the ``KernelSpec`` — an
edit literally cannot reach it, so the reference cannot drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import archdb
from .ir.schedule import ScheduleIR, Tile

# ═══════════════════════════════════════════════════════════════════════════════
# Result type — the gate's verdict
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Ok:
    """The edit's preconditions hold."""


@dataclass(frozen=True)
class Reject:
    """The edit is rejected with a reason the agent reads (training signal)."""

    reason: str


Result = Ok | Reject


def is_ok(r: Result) -> bool:
    return isinstance(r, Ok)


# ═══════════════════════════════════════════════════════════════════════════════
# The Phase 1 edits
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SetKnob:
    """Bind a specialization knob to a value (docs/brainstorm/10 §5, row 5).

    Precondition: the knob is declared on the IR and ``value ∈ choices``. This
    is the programmatic ``autotune-knob-sweep`` primitive (docs/brainstorm/09 §7).
    """

    name: str
    value: int

    def check(self, ir: ScheduleIR, arch: str) -> Result:
        knob = ir.knobs.get(self.name)
        if knob is None:
            return Reject(f"undeclared knob {self.name!r}")
        if self.value not in knob.choices:
            return Reject(
                f"knob {self.name!r}: {self.value} not in declared choices "
                f"{list(knob.choices)}"
            )
        return Ok()

    def apply(self, ir: ScheduleIR) -> ScheduleIR:
        from .ir.schedule import Knob

        knob = ir.knobs[self.name]
        return ir.with_node(Knob(name=knob.name, value=self.value, choices=knob.choices))


@dataclass(frozen=True)
class Retile:
    """Resize a tile (docs/brainstorm/10 §5, row 1).

    Precondition (STATEFUL — the load-bearing Phase 0 finding): the divisibility
    check only bites once an L5 matrix-engine map is present in the IR. A gate is
    a pure function of *(edit args, current IR, arch)*, so this correctly returns
    ``Ok`` when no L5 engine is mapped yet (no constraint to violate) and rejects
    a tile the mapped engine can't consume once one is.
    """

    tile_id: str
    shape: tuple[int, ...]

    def check(self, ir: ScheduleIR, arch: str) -> Result:
        node = ir.by_id(self.tile_id)
        if node is None:
            return Reject(f"no tile with id {self.tile_id!r}")
        if not isinstance(node, Tile):
            return Reject(f"{self.tile_id!r} is a {type(node).__name__}, not a Tile")
        if not self.shape:
            return Reject(f"retile of {self.tile_id!r}: empty shape")

        # Find a mapped L5 engine; if none, no divisibility constraint applies.
        native = None
        instruction = None
        for m in ir.maps():
            if m.level == "L5" and m.instruction is not None:
                instr_shape = archdb.native_shape(arch, m.instruction)
                if instr_shape is not None:
                    native = instr_shape
                    instruction = m.instruction
                    break
        if native is None:
            return Ok()  # no L5 engine mapped → nothing to divide by

        m_dim = native["m"]
        if self.shape[0] % m_dim != 0:
            return Reject(
                f"tile {self.tile_id!r} M={self.shape[0]} not divisible by "
                f"L5 {instruction} native m={m_dim}"
            )
        return Ok()

    def apply(self, ir: ScheduleIR) -> ScheduleIR:
        node = ir.by_id(self.tile_id)
        assert isinstance(node, Tile)
        return ir.with_node(Tile(id=node.id, shape=self.shape, level=node.level))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 edits (declared, checkable, but outside Phase 1's deliverable scope)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MapTo_:
    """Map a math node onto a hierarchy level + instruction (docs/brainstorm/10 §3).

    Phase 2 primitive (needs matrix-engine lowering). Its check — instruction
    legal for arch, native shape divides the L2 tile — is realized here so the
    gate vocabulary is complete; ``apply`` is Phase 2.
    """

    map_id: str
    op_ref: str
    level: str
    instruction: str
    instr_shape: tuple[int, ...] | None = None

    def check(self, ir: ScheduleIR, arch: str) -> Result:
        if self.instruction not in archdb.legal_instructions(arch):
            return Reject(
                f"instruction {self.instruction!r} not legal for {arch} "
                f"(have {list(archdb.legal_instructions(arch))})"
            )
        native = archdb.native_shape(arch, self.instruction)
        if native is not None:
            m_dim = native["m"]
            for t in ir.tiles():
                if t.level == "L2" and t.shape and t.shape[0] % m_dim != 0:
                    return Reject(
                        f"L2 tile {t.id!r} M={t.shape[0]} not divisible by "
                        f"{self.instruction} native m={m_dim}"
                    )
        return Ok()

    def apply(self, ir: ScheduleIR) -> ScheduleIR:  # pragma: no cover (Phase 2)
        from .ir.schedule import MapTo

        return ir.with_node(MapTo(
            id=self.map_id, op_ref=self.op_ref, level=self.level,  # type: ignore[arg-type]
            instruction=self.instruction, instr_shape=self.instr_shape,
        ))


@dataclass(frozen=True)
class AddStage:
    """Add a pipeline stage (docs/brainstorm/10 §5, scratch row).

    Phase 2 primitive. Its check — scratch fits the arch budget — is realized
    here; ``apply`` is Phase 2.
    """

    stage_id: str
    producer_ref: str
    depth: int
    tile_bytes: int

    def check(self, ir: ScheduleIR, arch: str) -> Result:
        budget = archdb.scratch_budget(arch)
        if budget == 0:
            return Ok()  # 'any' target: no scratch budget to overflow
        # Tentative total: existing scratch (from a cost annotation) + this stage.
        existing = sum(getattr(n, "_bytes", 0) for n in ir.stages())  # Phase 2 annotates
        tentative = existing + self.tile_bytes * self.depth
        if tentative > budget:
            return Reject(
                f"stage {self.stage_id!r}: scratch {tentative} B > {arch} budget "
                f"{budget} B"
            )
        return Ok()

    def apply(self, ir: ScheduleIR) -> ScheduleIR:  # pragma: no cover (Phase 2)
        from .ir.schedule import Stage

        return ir.with_node(Stage(
            id=self.stage_id, producer_ref=self.producer_ref,
            space="scratch", depth=self.depth,
        ))


# Any edit primitive implements this protocol (structural typing; no runtime cost).
EditKind = Literal["set_knob", "retile", "map_to", "add_stage"]
