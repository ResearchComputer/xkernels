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


# The legal ``input_precision`` policies an MMA may carry (Triton's vocabulary).
# ``None`` = dtype-default (the lowering omits the arg: tensor cores for bf16/fp8;
# CUDA-core FMA for fp32). ``"ieee"`` = true fp32 (no TF32). ``"tf32"`` = tensor-
# float-32 (the sm_80+ fp32 tensor-core mode, ~10x faster, ~1e-3 precision).
# This is the one MMA-level policy the Triton lowering responds to today - the
# agent-editable lever that proves the schedule IR reaches silicon
# (docs/brainstorm/09 section 8 step "map_to").
PRECISION_POLICIES: tuple[str | None, ...] = (None, "ieee", "tf32", "tf32x3")


@dataclass(frozen=True)
class MapTo:
    """Schedule a math node onto a hierarchy level + (optionally) an instruction.

    ``instruction="wgmma"``/``"mfma"`` -> L5 matrix engine; ``"fma"``/None -> L4
    scalar FMA (compiler picks). The ``check`` gate verifies the instruction is
    legal for the arch and that the L5 native shape divides the L2 tile.

    ``precision`` is the MMA's ``input_precision`` policy (None = dtype-default).
    It is the concrete agent-editable lever on the Triton backend: an fp32 GEMM's
    ieee->tf32 swap is a one-field ``SetMapPolicy`` edit that changes what compiles
    (docs/brainstorm/09 section 8). Stored here - the semantic home - not as a
    global knob, because it is a property of *how the MMA is scheduled*.
    """

    id: str
    op_ref: str  # -> a math node id (the WHAT being scheduled)
    level: Level
    instruction: str | None = None  # "wgmma" | "mfma" | "fma" | None
    instr_shape: tuple[int, ...] | None = None  # native MMA shape, e.g. (64, 128, 16)
    precision: str | None = None  # None | "ieee" | "tf32" | "tf32x3" (the MMA policy)


@dataclass(frozen=True)
class Stage:
    """A pipeline stage buffering a Load/Tile in some memory space."""

    id: str
    producer_ref: str  # a Load/Tile it buffers
    space: Space
    depth: int | str  # pipeline depth (concrete int, or a knob name resolved at emit)


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


# The cross-vendor stall vocabulary (docs/library.md §10, normalized). A profile
# parser (``vkl/profile.py``) maps a profiler's raw reason names onto one of these
# before anything routes on it — so the diagnose skills branch on a *closed enum*,
# never on a free-form profiler string. This is the causal signal the skills route
# on (the dominant stall reason), not the compute/mem throughput ratio alone.
StallReason = Literal[
    "memory_latency",  # L1TEX / LG / MIO throttle (ncu); SQ wait-cnt / TCC (rocprof)
    "dependency",      # Wait / scoreboard (ncu); instruction-fetch dependency
    "tensor_pipe",     # Long scoreboard on a compute-bound, idle-matrix-engine kernel
    "vgpr",            # VGPR/SGPR pressure caps resident waves/warps
    "scratch",         # registers spilled to backing memory
    "scheduling",      # latency-bound, no single dominant resource (un-pipelined load)
]


@dataclass(frozen=True)
class ProfileMetrics:
    """One kernel's on-device profile, normalized to the §10 vocabulary.

    The Phase C bridge (issue #74): a profile reports metrics at the *kernel
    symbol* granularity; this carries them in the cross-vendor form the diagnose
    skills route on. Produced by a parser (``vkl/profile.py``) from ncu / rocprof
    text tables; attached to schedule nodes by ``annotate_schedule``.

    Every field is ``... | None`` because a given profiler mode may not collect it
    (ncu ``roof`` vs ``sq`` are separate passes). The diagnose skills' routing
    function (``route``) degrades gracefully on ``None`` — it routes on what IS
    present, defaulting to the safest first probe.
    """

    bottleneck: str            # "compute" | "memory" | "latency"
    profiler: str              # "ncu" | "rocprof" (which tool produced this)
    dominant_stall: str | None = None          # a StallReason, normalized
    dominant_stall_pct: float | None = None    # share of total stall cycles
    achieved_bw_pct: float | None = None       # DRAM throughput % (ncu) / HBM (rocprof)
    compute_throughput_pct: float | None = None  # SM Compute % (ncu)
    tensor_pipe_util_pct: float | None = None    # matrix-engine pipeline utilization %
    ipc_active: float | None = None              # Executed IPC Active (ncu)
    occupancy_fraction: float | None = None      # achieved/peak warps or waves
    duration_us: float | None = None             # kernel duration µs

    def to_dict(self) -> dict[str, object]:
        return {
            "bottleneck": self.bottleneck,
            "profiler": self.profiler,
            "dominant_stall": self.dominant_stall,
            "dominant_stall_pct": self.dominant_stall_pct,
            "achieved_bw_pct": self.achieved_bw_pct,
            "compute_throughput_pct": self.compute_throughput_pct,
            "tensor_pipe_util_pct": self.tensor_pipe_util_pct,
            "ipc_active": self.ipc_active,
            "occupancy_fraction": self.occupancy_fraction,
            "duration_us": self.duration_us,
        }


ScheduleNode = Tile | MapTo | Stage | CopyAtom | Reduce | Knob


@dataclass(frozen=True)
class ScheduleIR:
    """An editable schedule: an indexed bag of nodes.

    Edits return a NEW frozen ScheduleIR (via ``replace_node`` / ``with_node``);
    the ``tuning_trace`` is the chain of these snapshots.
    """

    nodes: tuple[ScheduleNode, ...] = ()
    knobs: dict[str, Knob] = field(default_factory=dict)  # name -> Knob (fast lookup)
    # Phase C (issue #74): profile-derived annotations keyed to node ids. A
    # side-table (not a per-node field) because annotations are MEASURED/derived,
    # not authored — they must not dirty the node dataclasses the edit gate reasons
    # over, and they are re-derived by ``vkl/profile.py::annotate_schedule`` after
    # each MCP replay (the stateless agent loop replays from spec + edits; profile
    # is attached on demand by ``vkl_annotate_profile``, never carried in edits).
    profile: dict[str, ProfileMetrics] = field(default_factory=dict)

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
        """Return a copy with ``node`` added (or replacing the node of equal id).

        Carries the ``profile`` side-table forward (``replace`` preserves
        unspecified fields); edits do not drop annotations, but an agent should
        re-annotate after an edit because the metrics are now stale (the node-key
        projection is cheap and the MCP loop re-derives it each call anyway).
        """
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

    def with_profile(self, profile: dict[str, "ProfileMetrics"]) -> ScheduleIR:
        """Return a copy carrying ``profile`` (node-id -> measured metrics).

        The Phase C entry: ``annotate_schedule`` builds the side-table and returns
        a new IR via this. Frozen + replace keeps the trace-immutable invariant.
        """
        return replace(self, profile=dict(profile))
