# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Persisted tuning-trace records — the cross-task compounding store (issue #73).

A record is the ``{edit, predicted, measured, rationale}`` triple (track E of the
vkl roadmap, ``meta/docs/design/vkl.md`` §11) keyed by
``(op, arch, shape, dtype, edit)``. It is the *persisted* form of a
``gate.TraceEntry``: an in-memory trace dies with the task that produced it, but a
persisted record lets the **next** task ask "did a prior task already try this
edit, what did it predict, what did it measure, and why?" — and skip the dead-end
or reuse the winner. That is the cross-task compounding that makes the *next*
task cheaper (``meta/docs/library.md`` §6.2, the explicit point of the loop).

Storage mirrors ``registry/outcomes.py``: an append-only JSONL log at
``registry/tuning_traces.jsonl`` — plain, greppable, vendor-neutral (the bottom
consumption tier). The latest record for a given
``(op, arch, shape, dtype, edit)`` key wins (replace-on-write, like
``record_measurement``), so re-tuning a point updates rather than drowns the
signal. External/untrusted callers are read + verify only; write-back into the
shared corpus requires a server-side-rerun ``source`` (same stance as
write-back/outcomes, ``§8.4``/``§11``).

Staging (per the issue):

  * the **predicted** half is closed-form (``vkl.cost``) and CPU-doable; it is
    filled from the cost model when omitted.
  * the **measured** half (``ms``/``tflops``/``achieved_bw_pct``) comes from
    ``verify`` and is GPU-gated. The store accepts **ms-only first** (the
    ``perf.ms`` ``verify`` already returns) and absorbs the richer metrics once
    track C (profile feedback, #74) feeds them.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..registry.loader import registry_root

_TRACE_FIELDS = {"ms", "tflops", "achieved_bw_pct"}


def _traces_path() -> Path:
    """The JSONL trace store (one ``TraceEntry``-shaped record per line)."""
    return registry_root() / "tuning_traces.jsonl"


def _canonical_edit(edit: dict[str, Any]) -> str:
    """A stable short key for an edit dict, so two identical edits match.

    ``{"kind": "set_knob", "name": "num_stages", "value": 3}`` ->
    ``"set_knob{name=num_stages,value=3}"``. Sorted by key so field order in the
    dict does not fragment the key. This is what makes "skip the already-tuned
    point" a lookup, not a search: a later task recomputes the same canonical key
    and finds the prior record.
    """
    items = sorted((k, v) for k, v in edit.items() if k != "kind")
    body = ",".join(f"{k}={v}" for k, v in items)
    return f"{edit.get('kind', '?')}{{{body}}}"


def _record_key(
    op: str, arch: str, shape: dict[str, Any], dtype: str, edit: dict[str, Any]
) -> tuple:
    return (
        op,
        arch,
        json.dumps(shape, sort_keys=True),
        dtype,
        _canonical_edit(edit),
    )


def _shape_of(point: dict[str, Any] | None, shape: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the spatial shape, tolerating either an explicit ``shape`` dict or a
    flat cost-model ``point`` (the convention ``vkl.cost.roofline`` uses: the point
    IS the shape dims plus a ``dtype`` key)."""
    if shape:
        return dict(shape)
    if point:
        return {k: v for k, v in point.items() if k != "dtype"}
    return {}


def _dtype_of(point: dict[str, Any] | None, dtype: str | None) -> str:
    if dtype:
        return dtype
    if point and "dtype" in point:
        return str(point["dtype"])
    return ""


def record_trace(
    op: str,
    arch: str,
    edit: dict[str, Any],
    *,
    shape: dict[str, Any] | None = None,
    dtype: str | None = None,
    point: dict[str, Any] | None = None,
    applied_edits: list[dict[str, Any]] | None = None,
    check: str = "ok",
    reason: str = "",
    predicted: dict[str, Any] | None = None,
    measured: dict[str, Any] | None = None,
    rationale: str = "",
    source: str | None = None,
    trust: str = "trusted",
) -> dict[str, Any]:
    """Persist one ``{edit, predicted, measured, rationale}`` record (issue #73).

    Args:
        op: the op id (e.g. ``"gemm_bf16@1.0.0"``). Required — the key's first leg.
        arch: the target arch (e.g. ``"amd_cdna3"``). Required.
        edit: the edit dict that was tried, e.g.
            ``{"kind": "set_knob", "name": "num_stages", "value": 3}``.
        shape / dtype: the measurement point's shape + dtype. Either pass them
            directly or via ``point`` (``{"shape": {...}, "dtype": "bf16", ...}``).
        applied_edits: the prior edit sequence ``edit`` was proposed on top of
            (the agent's replayed state). Recorded so the predicted half can be
            re-derived and so the key disambiguates the same edit under different
            prefixes.
        check: ``"ok"`` (the edit applied) or ``"reject"`` (the gate rejected it).
        reason: the gate's reject string when ``check == "reject"`` (the
            machine-stable training signal).
        predicted: the closed-form cost-model call. If omitted and a ``point`` or
            ``shape`` is given, it is computed from ``vkl.cost`` (CPU-doable).
        measured: the on-device outcome — any subset of ``{ms, tflops,
            achieved_bw_pct}`` from ``verify`` (ms-only first; GPU-gated).
        rationale: free-text agent note — the *human* reason a later task cites.
        source: a reproducible run id (e.g. ``verify``'s ``artifacts.run_id``).
            Strongly recommended; required for an external write.
        trust: ``"trusted"`` (this runtime) or ``"external"``. External writes
            require a server-side-rerun ``source`` prefix (same stance as
            ``record_measurement``).

    Returns the recorded record. The latest record for a given
    ``(op, arch, shape, dtype, edit)`` key replaces any prior one.
    """
    if not op or not arch:
        raise ValueError("record_trace requires `op` and `arch` (the key legs, §6.2)")
    if check not in ("ok", "reject"):
        raise ValueError(f"check must be 'ok' or 'reject', got {check!r}")
    if trust == "external" and not (source or "").startswith("server-rerun:"):
        raise ValueError(
            "external write-back requires a server-side-rerun source "
            "(prefix 'server-rerun:'); use verify() to obtain one (§11)"
        )

    shape_resolved = _shape_of(point, shape)
    dtype_resolved = _dtype_of(point, dtype)
    measured = {k: v for k, v in (measured or {}).items() if k in _TRACE_FIELDS and v is not None}
    predicted = predicted or {}

    record: dict[str, Any] = {
        "op": op,
        "arch": arch,
        "shape": shape_resolved,
        "dtype": dtype_resolved,
        "edit": edit,
        "edit_key": _canonical_edit(edit),
        "applied_edits": list(applied_edits or []),
        "check": check,
        "reason": reason,
        "predicted": predicted,
        "measured": measured,
        "rationale": rationale,
        "source": source or "",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    path = _traces_path()
    key = _record_key(op, arch, shape_resolved, dtype_resolved, edit)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _record_key(
                    rec.get("op", ""), rec.get("arch", ""),
                    rec.get("shape", {}), rec.get("dtype", ""), rec.get("edit", {}),
                ) == key:
                    continue  # drop the superseded record for this key
                existing.append(rec)
    existing.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in existing:
            f.write(json.dumps(rec, default=str, sort_keys=True))
            f.write("\n")
    return record


def prior_traces(
    op: str,
    arch: str,
    *,
    shape: dict[str, Any] | None = None,
    dtype: str | None = None,
) -> list[dict[str, Any]]:
    """Read prior trace records for ``(op, arch[, shape, dtype])`` (issue #73).

    The cross-task retrieval an agent does at the start of a task: "what has
    already been tried at this point, and why?" Pass ``shape``/``dtype`` to scope
    to one measurement point; omit them to get every prior record for the
    ``(op, arch)`` pair (e.g. to seed a fresh sweep with the dead-ends to avoid).

    Returns records newest-last (append order). Each record is the full
    ``{edit, check, reason, predicted, measured, rationale, source}`` triple.
    """
    path = _traces_path()
    if not path.exists():
        return []
    shape_json = json.dumps(shape, sort_keys=True) if shape is not None else None
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("op") != op or rec.get("arch") != arch:
                continue
            if shape_json is not None and json.dumps(
                rec.get("shape", {}), sort_keys=True
            ) != shape_json:
                continue
            if dtype is not None and rec.get("dtype", "") != dtype:
                continue
            out.append(rec)
    return out
