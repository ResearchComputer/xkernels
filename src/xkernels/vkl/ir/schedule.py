# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The editable schedule IR — the HOW over the L0–L5 hierarchy (docs/brainstorm/10 §2).

Unlike the math IR, these nodes ARE the edit target: the agent (or a tuning
skill) rewrites them via edit primitives (``edits.py``) to chase a roofline.
Every field that names hardware is ``str | None`` or a closed enum — never a
free-form literal. ``instruction="wgmma"`` is legal; ``instruction="my_asm"``
is rejected at the gate. There is no syntax for "32 lanes"; wave size is bound
by the target (``archdb.py`` / ``archs.py``), never remembered by a human.

Fields are frozen for trace immutability (an edit returns a NEW schedule; the
``tuning_trace`` is a chain of snapshots — docs/brainstorm/10 §3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from .math import Dim

# The 6-level hardware hierarchy (docs/brainstorm/08).
#   L0 device | L1 cluster | L2 CTA | L3 wave/warp | L4 lane | L5 matrix engine
Level = Literal["L0", "L1", "L2", "L3", "L4", "L5"]
Space = Literal["register", "scratch", "dsmem", "global", "descriptor"]


@dataclass(frozen=True)
class Tile:
    """A tile of data. Output tiles live at L0/L2; streaming tiles at L2."""

    id: str
    shape: tuple[Dim, ...]  # ints (concrete) or knob names (symbolic)
    level: Literal["L0", "L2"]


@dataclass(frozen=True)
class MapTo:
    """Schedule a math node onto a hierarchy level + (optionally) an instruction.

    ``instruction="wgmma"``/``"mfma"`` → L5 matrix engine; ``"fma"``/None → L4
    scalar FMA (compiler picks). The ``check`` gate verifies the instruction is
    legal for the arch and that the L5 native shape divides the L2 tile.
    """

    id: str
    op_ref: str  # → a math node id (the WHAT being scheduled)
    level: Level
    instruction: str | None = None  # "wgmma" | "mfma" | "fma" | None
    instr_shape: tuple[int, ...] | None = None  # native MMA shape, e.g. (64, 128, 16)


@dataclass(frozen=True)
class Stage:
    """A pipeline stage buffering a Load/Tile in some memory space."""

    id: str
    producer_ref: str  # a Load/Tile it buffers
    space: Space
    depth: int  # pipeline depth (concrete; symbolic knobs resolve before emit)


@dataclass(frozen=True)
class CopyAtom:
    """A copy primitive between two spaces (e.g. global → scratch), vectorized."""

    id: str
    src: Space
    dst: Space
    width: int = 0  # vectorize lanes-wide (0 = auto)
    swizzle: str | None = None  # None | "xor" | "pad" (bank-conflict policy)


@dataclass(frozen=True)
class Reduce:
    """Schedule a math Reduce onto a hierarchy level."""

    id: str
    op_ref: str  # → a math Reduce id
    level: Literal["L3", "L2", "L0"]  # within-wave / within-CTA / cross-CTA


@dataclass(frozen=True)
class Knob:
    """A declared specialization point + its current binding."""

    name: str
    value: int  # current binding; MUST be ∈ choices (gate-enforced)
    choices: tuple[int, ...]  # the declared specialization space


ScheduleNode = Tile | MapTo | Stage | CopyAtom | Reduce | Knob


@dataclass(frozen=True)
class ScheduleIR:
    """An editable schedule: an indexed bag of nodes.

    Edits return a NEW frozen ScheduleIR (via ``replace_node`` / ``with_node``);
    the ``tuning_trace`` is the chain of these snapshots.
    """

    nodes: tuple[ScheduleNode, ...] = ()
    knobs: dict[str, Knob] = field(default_factory=dict)  # name -> Knob (fast lookup)

    def by_id(self, node_id: str) -> ScheduleNode | None:
        for n in self.nodes:
            if getattr(n, "id", None) == node_id or getattr(n, "name", None) == node_id:
                return n
        return None

    def tiles(self) -> tuple[Tile, ...]:
        return tuple(n for n in self.nodes if isinstance(n, Tile))

    def maps(self) -> tuple[MapTo, ...]:
        return tuple(n for n in self.nodes if isinstance(n, MapTo))

    def stages(self) -> tuple[Stage, ...]:
        return tuple(n for n in self.nodes if isinstance(n, Stage))

    def map_nodes_by_id(self) -> dict[str, ScheduleNode]:
        """All nodes indexable by their id/name — for edit lookups."""
        out: dict[str, ScheduleNode] = {}
        for n in self.nodes:
            key = getattr(n, "id", None) or getattr(n, "name", None)
            if key is not None:
                out[key] = n
        return out

    def scratch_total(self) -> int:
        """Sum of scratch bytes across stages — the AddStage budget input.

        Scratch bytes per stage are computed by ``cost.py`` from the stage's tile
        shape × depth; here we only aggregate already-annotated bytes. For the
        Phase 1 gate (which checks a *candidate* depth against a budget), the
        caller passes the candidate bytes explicitly.
        """
        return 0  # Phase 1: scratch accounting lives in the gate's candidate math

    def with_node(self, node: ScheduleNode) -> ScheduleIR:
        """Return a copy with ``node`` added (or replacing the node of equal id)."""
        key = getattr(node, "id", None) or getattr(node, "name", None)
        kept = tuple(
            n for n in self.nodes
            if (getattr(n, "id", None) or getattr(n, "name", None)) != key
        )
        new_nodes = kept + (node,)
        new_knobs = self.knobs
        if isinstance(node, Knob):
            new_knobs = {**self.knobs, node.name: node}
        return replace(self, nodes=new_nodes, knobs=new_knobs)
