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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .edits import Result, is_ok
from .ir.schedule import ScheduleIR


@dataclass(frozen=True)
class TraceEntry:
    """One step of a tuning trace (docs/brainstorm/10 §3).

    The ``{edit, predicted, measured, rationale}`` triple (issue #73, track E):
    ``edit`` is what was tried, ``predicted`` is the closed-form cost-model call
    made BEFORE launch, ``measured`` is the on-device outcome (``verify`` perf,
    ms-only first), and ``rationale`` is the free-text reason a later task reads
    to skip a dead-end or reuse a winner. ``reason`` carries the gate's own
    reject string (machine-stable); ``rationale`` carries the agent's human note.
    A trace entry with ``check="reject"`` + a ``rationale`` is the structured
    form of a dead-end the next run avoids without re-deriving it.
    """

    step: int
    edit: str
    target: str
    args: dict[str, Any]
    check: str  # "ok" | "reject"
    reason: str = ""
    rationale: str = ""
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
        if self.rationale:
            d["rationale"] = self.rationale
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


IssueLevel = Literal["error", "warning"]


@dataclass(frozen=True)
class KernelIssue:
    """One contract/preflight issue found on a ``KernelSpec``.

    This is deliberately smaller and more agent-readable than a Python
    traceback. The code is stable enough for routing; the message is for humans.
    """

    level: IssueLevel
    code: str
    message: str
    node: str = ""

    def to_dict(self) -> dict[str, str]:
        d = {"level": self.level, "code": self.code, "message": self.message}
        if self.node:
            d["node"] = self.node
        return d


@dataclass(frozen=True)
class KernelValidation:
    """Top-level VKL preflight result for a kernel contract."""

    passed: bool
    issues: tuple[KernelIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "error_count": sum(1 for i in self.issues if i.level == "error"),
            "warning_count": sum(1 for i in self.issues if i.level == "warning"),
        }


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
    predict: Callable[[ScheduleIR, str], dict[str, Any]] | None = None,
) -> GateResult:
    """Run an edit sequence; apply the passing ones, record every verdict.

    Each edit is checked against the IR *as it stands after prior applied edits*
    (stateful — the load-bearing Phase 0 finding: a gate is a pure function of
    *(edit args, current IR, arch)*, so the sequence order matters and a later
    edit may become legal only after an earlier one applied).

    ``predict`` (issue #73) is an optional closed-form hook
    ``(applied_schedule, arch) -> dict`` that fills each OK entry's ``predicted``
    field — the cost-model call made *before* the on-device measure. It keeps the
    gate decoupled from ``cost.py`` (no import cycle) while letting the
    predicted half of the {predicted, measured, rationale} triple flow into the
    trace. Rejects get no prediction (the edit did not apply, so there is no
    resulting schedule to cost).
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
            predicted = predict(current, arch) if predict is not None else {}
            entry = TraceEntry(
                step=step, edit=_edit_kind(edit), target=target,
                args=_edit_args(edit), check="ok", predicted=predicted,
            )
        else:
            rejected += 1
            entry = TraceEntry(
                step=step, edit=_edit_kind(edit), target=target,
                args=_edit_args(edit), check="reject", reason=result.reason,  # type: ignore[attr-defined]
            )
        trace.append(entry)
    return GateResult(final_ir=current, trace=tuple(trace), applied=applied, rejected=rejected)


_SUPPORTED_POINTWISE = {
    "cast",
    "mul",
    "add",
    "div",
    "rsqrt",
    "sub",
    "neg",
    "abs",
    "exp",
    "tanh",
    "sigmoid",
    "sqrt",
    "silu",
    "gelu",
    "min",
    "max",
    "where",
}

_TRITON_KNOBS_BY_PATTERN = {
    "tiled_2d": {"BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"},
    "rowwise": {"num_warps", "num_stages"},
    "elementwise": {"BLOCK", "num_warps", "num_stages"},
}


def validate_kernel(spec: Any, *, arch: str = "any") -> KernelValidation:
    """Validate a VKL kernel before any device compile/launch.

    This is the contract-level gate for the DSL path. It checks the parts that
    are CPU-decidable:

    * emitted Op Spec / reference card / target cards are schema-valid;
    * constraints are in the decidable mini-language;
    * trace bodies build a math IR and store the declared outputs;
    * reductions/MMAs use ``numerics.reduce_dtype``;
    * the launch pattern can lower the math-node family it is given;
    * declared Triton knobs are names the launcher actually consumes.

    It complements ``run_gate``: ``run_gate`` validates an edit sequence against
    an existing schedule, while this function validates the kernel contract that
    produced the schedule in the first place.
    """

    issues: list[KernelIssue] = []

    def add(level: IssueLevel, code: str, message: str, node: str = "") -> None:
        issues.append(KernelIssue(level, code, message, node))

    _validate_emitted_artifacts(spec, add)
    _validate_constraints(spec, add)
    _validate_target_knobs(spec, add)

    if getattr(spec, "launch", None) is None:
        if getattr(spec, "targets", None):
            add(
                "warning",
                "direct_body_with_targets",
                "kernel has backend targets but no @launch pattern, "
                "so no device schedule is described",
            )
        return KernelValidation(not any(i.level == "error" for i in issues), tuple(issues))

    try:
        from .reference import trace_ir

        body = trace_ir(spec)
    except Exception as exc:  # pragma: no cover - exact exception depends on body code
        add("error", "trace_failed", f"trace body failed to build math IR: {exc}")
        return KernelValidation(False, tuple(issues))

    if body is None:
        add("error", "missing_trace", "trace_ir returned None for a launched kernel")
        return KernelValidation(False, tuple(issues))

    _validate_math_ir(spec, body, add)
    _validate_launch_compat(spec, body, add)

    return KernelValidation(not any(i.level == "error" for i in issues), tuple(issues))


def _validate_emitted_artifacts(spec: Any, add: Any) -> None:
    from .emit import emit_card, emit_reference_card, emit_spec

    try:
        from ..registry.schemas import validate_impl_card, validate_op_spec
    except Exception as exc:  # pragma: no cover - registry import failures are environment issues
        add("error", "schema_validator_unavailable", f"could not import schema validators: {exc}")
        return

    try:
        validate_op_spec(emit_spec(spec))
    except Exception as exc:
        add("error", "op_spec_schema", f"emitted Op Spec is not schema-valid: {exc}")
    try:
        validate_impl_card(emit_reference_card(spec))
    except Exception as exc:
        add("error", "reference_card_schema", f"emitted reference card is not schema-valid: {exc}")
    for name, target in getattr(spec, "targets", {}).items():
        try:
            validate_impl_card(emit_card(spec, target))
        except Exception as exc:
            add("error", "impl_card_schema", f"emitted {name!r} card is not schema-valid: {exc}")


def _validate_constraints(spec: Any, add: Any) -> None:
    try:
        from ..registry.constraints import validate_decidable
    except Exception as exc:  # pragma: no cover
        add(
            "warning",
            "constraint_validator_unavailable",
            f"could not import constraint validator: {exc}",
        )
        return
    for constraint in getattr(spec, "constraints", ()):
        try:
            validate_decidable(constraint)
        except Exception as exc:
            add(
                "error",
                "undecidable_constraint",
                f"constraint {constraint!r} is not decidable: {exc}",
            )


def _validate_target_knobs(spec: Any, add: Any) -> None:
    launch = getattr(spec, "launch", None)
    pattern = getattr(launch, "pattern", None)
    if pattern is None:
        return
    allowed = _TRITON_KNOBS_BY_PATTERN.get(pattern)
    if allowed is None:
        add("error", "unknown_launch_pattern", f"unknown launch pattern {pattern!r}")
        return
    triton = getattr(spec, "targets", {}).get("triton")
    if triton is None:
        return
    for knob in getattr(triton, "knobs", {}):
        if knob not in allowed:
            add(
                "error",
                "unsupported_knob",
                f"triton knob {knob!r} is not consumed by the {pattern!r} launcher "
                f"(allowed: {sorted(allowed)})",
            )


def _validate_math_ir(spec: Any, body: Any, add: Any) -> None:
    from .ir.math import MMA, Pointwise, Reduce, Store

    output_names = tuple(getattr(n.ref, "name", "") for n in body.ir.nodes if isinstance(n, Store))
    declared = tuple(getattr(spec, "outputs", {}).keys())
    missing = [name for name in declared if name not in output_names]
    extra = [name for name in output_names if name not in declared]
    dupes = sorted({name for name in output_names if output_names.count(name) > 1})
    for name in missing:
        add("error", "missing_store", f"declared output {name!r} is never stored", name)
    for name in extra:
        add("error", "extra_store", f"body stores undeclared output {name!r}", name)
    for name in dupes:
        add("error", "duplicate_store", f"output {name!r} is stored more than once", name)

    reduce_dtype = getattr(getattr(spec, "numerics", None), "reduce_dtype", None)
    for node in body.ir.nodes:
        if isinstance(node, (MMA, Reduce)):
            if reduce_dtype is None:
                add(
                    "error",
                    "missing_reduce_dtype",
                    f"{type(node).__name__} accumulates in {node.accum_dtype!r}, "
                    "but numerics.reduce_dtype is unset",
                    getattr(getattr(node, "out", None), "name", ""),
                )
            elif node.accum_dtype != reduce_dtype:
                add(
                    "error",
                    "accum_dtype_mismatch",
                    f"{type(node).__name__} accumulates in {node.accum_dtype!r}, "
                    f"but numerics.reduce_dtype is {reduce_dtype!r}",
                    getattr(getattr(node, "out", None), "name", ""),
                )
        if isinstance(node, Pointwise) and node.fn not in _SUPPORTED_POINTWISE:
            add(
                "error",
                "unsupported_pointwise",
                f"pointwise op {node.fn!r} is not lowerable",
                node.out.name,
            )
        if isinstance(node, Store):
            ref = body.ir.tensors.get(node.val.name)
            if ref is not None and len(ref.shape) != len(node.ref.shape):
                add(
                    "error",
                    "store_rank_mismatch",
                    f"store to {node.ref.name!r} has rank {len(ref.shape)}, "
                    f"but output rank is {len(node.ref.shape)}",
                    node.ref.name,
                )


def _validate_launch_compat(spec: Any, body: Any, add: Any) -> None:
    from .ir.math import MMA, Concat, Gather, Load, Pointwise, Reduce, Slice, Store, Unsqueeze

    pattern = spec.launch.pattern
    nodes = body.ir.nodes
    if pattern == "tiled_2d":
        mmas = [n for n in nodes if isinstance(n, MMA)]
        if len(mmas) != 1:
            add(
                "error",
                "tiled_2d_mma_count",
                f"tiled_2d lowering requires exactly one MMA; found {len(mmas)}",
            )
        unsupported = (Reduce, Gather, Slice, Concat, Unsqueeze)
        for node in nodes:
            if isinstance(node, unsupported):
                add(
                    "error",
                    "unsupported_tiled_2d_node",
                    f"tiled_2d lowering cannot lower {type(node).__name__}",
                    getattr(getattr(node, "out", None), "name", ""),
                )
        return

    if pattern == "rowwise":
        for node in nodes:
            if isinstance(node, MMA):
                add(
                    "error",
                    "unsupported_rowwise_mma",
                    "rowwise lowering cannot lower MMA",
                    node.out.name,
                )
            if isinstance(node, Reduce):
                rank = len(node.x.shape)
                axis = node.axis % rank if rank else node.axis
                if axis != rank - 1:
                    add(
                        "error",
                        "rowwise_reduce_axis",
                        "rowwise lowering only reduces the last axis; "
                        f"got axis {node.axis} of rank {rank}",
                        node.out.name,
                    )
            if isinstance(node, (Gather, Slice, Concat)):
                add(
                    "error",
                    "unsupported_rowwise_addressing",
                    f"rowwise lowering cannot lower {type(node).__name__}; "
                    "use elementwise or hand path",
                    getattr(getattr(node, "out", None), "name", ""),
                )
        if not any(isinstance(n, Reduce) for n in nodes):
            add("warning", "rowwise_without_reduce", "rowwise launch has no Reduce nodes")
        return

    if pattern == "elementwise":
        for node in nodes:
            if isinstance(node, (MMA, Reduce)):
                add(
                    "error",
                    "unsupported_elementwise_node",
                    f"elementwise lowering cannot lower {type(node).__name__}",
                    getattr(getattr(node, "out", None), "name", ""),
                )
        return

    # Unknown patterns are caught by target-knob validation; keep this defensive.
    for node in nodes:
        if not isinstance(node, (Load, Pointwise, Store)):
            add(
                "warning",
                "unknown_pattern_node",
                f"unknown pattern {pattern!r} with node {type(node).__name__}",
            )
