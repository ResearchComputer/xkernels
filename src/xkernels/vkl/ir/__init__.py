# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""IR subpackage: frozen math oracle + editable schedule (docs/brainstorm/10 §1–2)."""
from __future__ import annotations

from .math import (
    MMA,
    Load,
    MathIR,
    MathNode,
    Pointwise,
    Reduce,
    Store,
    TensorRef,
)
from .schedule import (
    CopyAtom,
    Knob,
    MapTo,
    ScheduleIR,
    ScheduleNode,
    Stage,
    Tile,
)
from .schedule import (
    Reduce as SchedReduce,
)

__all__ = [
    # math (frozen oracle)
    "TensorRef", "Load", "Reduce", "MMA", "Pointwise", "Store", "MathNode", "MathIR",
    # schedule (editable)
    "Tile", "MapTo", "Stage", "CopyAtom", "SchedReduce", "Knob",
    "ScheduleNode", "ScheduleIR",
]
