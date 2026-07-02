# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The check gate: validate an edit SEQUENCE, return a tuning trace (docs/brainstorm/09 §6, 10 §5).

The gate runs each edit's ``check`` against the current IR state. If an edit
passes, it is applied (producing the next immutable snapshot) and the trace
records the win; if it fails, the IR is unchanged and the trace records the
reject reason — the *training signal* the next agent run reads to skip dead-ends
(the compounding loop, docs/brainstorm/09 §6 step 7).

A trace entry is the token-compact JSON shape from docs/brainstorm/10 §3:
``{step, edit, target, args, check, predicted, measured}`` — one line of intent,
one line of outcome, ~6 entries to a context window.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .edits import Result, is_ok
from .ir.schedule import ScheduleIR


@dataclass(frozen=True)
class TraceEntry:
    """One step of a tuning trace (docs/brainstorm/10 §3)."""

    step: int
    edit: str
    target: str
    args: dict[str, Any]
    check: str  # "ok" | "reject"
    reason: str = ""
    predicted: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step": self.step,
            "edit": self.edit,
            "target": self.target,
            "args": self.args,
            "check": self.check,
        }
        if self.reason:
            d["reason"] = self.reason
        if self.predicted:
            d["predicted"] = self.predicted
        if self.measured:
            d["measured"] = self.measured
        return d


@dataclass(frozen=True)
class GateResult:
    """Outcome of running an edit sequence through the gate."""

    final_ir: ScheduleIR
    trace: tuple[TraceEntry, ...]
    applied: int
    rejected: int


def _edit_kind(edit: Any) -> str:
    return type(edit).__name__.rstrip("_").lower()  # SetKnob->setknob, MapTo_->mapto


def _edit_target(edit: Any) -> str:
    # Every Phase 1/2 edit names its target as its first non-class field.
    for attr in ("name", "tile_id", "map_id", "stage_id", "copy_id", "reduce_id"):
        v = getattr(edit, attr, None)
        if v is not None:
            return str(v)
    return ""


def _edit_args(edit: Any) -> dict[str, Any]:
    """The public args of the edit (everything except the target id)."""
    import dataclasses as dc

    target_field = target_attr_of(edit)
    out: dict[str, Any] = {}
    for f in dc.fields(edit):
        if f.name == target_field:
            continue
        v = getattr(edit, f.name)
        if isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    return out


def target_attr_of(edit: Any) -> str:
    """The dataclass field name that holds the edit's target id."""
    for attr in ("name", "tile_id", "map_id", "stage_id", "copy_id", "reduce_id"):
        if hasattr(edit, attr):
            return attr
    return ""


def run_gate(
    edits: list[Any],
    ir: ScheduleIR,
    arch: str,
    *,
    start_step: int = 1,
) -> GateResult:
    """Run an edit sequence; apply the passing ones, record every verdict.

    Each edit is checked against the IR *as it stands after prior applied edits*
    (stateful — the load-bearing Phase 0 finding: a gate is a pure function of
    *(edit args, current IR, arch)*, so the sequence order matters and a later
    edit may become legal only after an earlier one applied).
    """
    current = ir
    trace: list[TraceEntry] = []
    applied = 0
    rejected = 0
    for i, edit in enumerate(edits):
        step = start_step + i
        result: Result = edit.check(current, arch)
        target = _edit_target(edit)
        if is_ok(result):
            current = edit.apply(current)
            applied += 1
            entry = TraceEntry(
                step=step, edit=_edit_kind(edit), target=target,
                args=_edit_args(edit), check="ok",
            )
        else:
            rejected += 1
            entry = TraceEntry(
                step=step, edit=_edit_kind(edit), target=target,
                args=_edit_args(edit), check="reject", reason=result.reason,  # type: ignore[attr-defined]
            )
        trace.append(entry)
    return GateResult(final_ir=current, trace=tuple(trace), applied=applied, rejected=rejected)
