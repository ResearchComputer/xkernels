# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Project a ``KernelSpec`` to Op Spec / Impl Card JSON (docs/brainstorm/10 §0).

This is the contract-native thesis made literal: every IR object lowers to
fields the schema ALREADY knows (``arch.family``, ``arch.requires``,
``arch.wave_size``, ``arch.scratch.kind``, ``specialization_knobs``, ``backend``,
``perf.roofline``). The emitter does not invent contract vocabulary; it
*produces* the existing vocabulary from the richer editable representation.

The output MUST be accepted as-is by the real validators
(``validate_op_spec`` / ``validate_impl_card``) and ingestors
(``op_spec_from_doc`` / ``ImplCard.from_doc``) — that is the Phase 1 round-trip
test (docs/brainstorm/11 §2, §4). If the emitter ever needs a field the schema
rejects, the IR is wrong, not the schema (docs/brainstorm/10 §0).
"""
from __future__ import annotations

import datetime as _dt
from typing import Any

from .ir.schedule import ScheduleIR
from .surface import KernelSpec, Target

# The DSL emitter's fixed provenance (authored_by="dsl" — the enum value the
# Phase 1 schema edit added). Cards are contract-identical to hand-written ones;
# this just records provenance for the compounding loop.
_DSL_SOURCE = "xkernels.vkl"  # default source_path prefix for emitted kernels


def emit_spec(spec: KernelSpec) -> dict[str, Any]:
    """Project a KernelSpec header to a schema-valid Op Spec dict."""
    return {
        "id": spec.id,
        "name": spec.name,
        "version": spec.version,
        "kernel": spec.kernel,
        "op": {
            "signature": spec.signature,
            "canonical_op": spec.canonical_op,
            "fusions": list(spec.fusions),
        },
        "inputs": {k: v.to_dict() for k, v in spec.inputs.items()},
        "outputs": {k: v.to_dict() for k, v in spec.outputs.items()},
        "constraints": list(spec.constraints),
        "preconditions": list(spec.preconditions),
        "numerics": spec.numerics.to_dict(spec.reference_path),
        "shape_sweep": spec.shape_sweep,
        "composes_with": [],
    }


def emit_card(
    spec: KernelSpec,
    target: Target,
    *,
    schedule: ScheduleIR | None = None,
    created: str | None = None,
    source_path: str | None = None,
    tuning_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Project a (KernelSpec, Target[, ScheduleIR]) to a schema-valid Impl Card dict.

    The card's ``specialization_knobs`` come from the target's declared knob
    choices (the autotune space); when a ``schedule`` is given, its ``Knob``
    bindings are recorded as the *current* point but the declared space is what
    the card advertises (the search space, not one search result).
    """
    backend = target.backend
    card_id = f"{spec.short_name}.{backend}@{spec.version}"
    knobs_decl = _emit_knobs(target, schedule)
    provenance: dict[str, Any] = {
        "authored_by": "dsl",
        "created": created or _now_iso(),
        "source_path": source_path or f"{_DSL_SOURCE}:{spec.short_name}",
    }
    if tuning_trace is not None:
        provenance["tuning_trace"] = tuning_trace

    return {
        "id": card_id,
        "implements": spec.id,
        "backend": backend,
        "arch": {
            "family": target.arch,
            "requires": list(target.requires),
            "wave_size": target.wave_size,
            "scratch": {"kind": target.scratch_kind, "bytes": 0},
        },
        "specialization_knobs": knobs_decl,
        "perf": {
            "regime": target.regime,
            "roofline": target.roofline,
            "measured": [],
        },
        "uses_primitives": [],
        "supersedes": [],
        "provenance": provenance,
    }


def emit_reference_card(
    spec: KernelSpec,
    *,
    created: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    """Project a KernelSpec to its reference Impl Card (backend="reference").

    Every DSL op's body IS the auto-reference (``docs/brainstorm/02`` §1), so
    every DSL op has a reference card for free — the backend-neutral oracle the
    substrate's invariants expect (every op: one spec + a reference card + per-
    backend cards). Mirrors the hand ``*.reference.card.json`` shape exactly.
    """
    card_id = f"{spec.short_name}.reference@{spec.version}"
    return {
        "id": card_id,
        "implements": spec.id,
        "backend": "reference",
        "arch": {
            "family": "any",
            "requires": [],
            "wave_size": 0,
            "scratch": {"kind": "none", "bytes": 0},
        },
        "specialization_knobs": {},
        "perf": {
            "regime": "pure-torch auto-reference: the @kernel body on torch.",
            "roofline": "latency_bound",
            "measured": [],
        },
        "uses_primitives": [],
        "supersedes": [],
        "provenance": {
            "authored_by": "dsl",
            "created": created or _now_iso(),
            "source_path": source_path or f"{_DSL_SOURCE}:{spec.short_name}",
        },
    }


def _emit_knobs(target: Target, schedule: ScheduleIR | None) -> dict[str, dict[str, Any]]:
    """The specialization_knobs space: declared choices from the target.

    Knobs declared on the target (``knobs={name: choices}``) become the card's
    ``{name: {type:"int", choices:[...]}}``. The schema's knob sub-schema allows
    a namespaced ``_doc`` (additionalProperties is open there), so we annotate.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, choices in target.knobs.items():
        out[name] = {
            "type": "int",
            "choices": list(choices),
            "_doc": f"vkl-declared autotune knob for {target.backend}/{target.arch}",
        }
    # If a schedule carries Knob nodes with a wider/finer space, merge them in
    # (the schedule is the editable truth; the target is the declared default).
    if schedule is not None:
        for k in schedule.knobs.values():
            if k.name not in out:
                out[k.name] = {"type": "int", "choices": list(k.choices)}
    return out


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
