"""Compounding write-back (meta/docs/library.md §6.2).

A successful ``(op, arch, shape, dtype) -> knobs -> perf`` tuple is appended to
the matching card's ``perf.measured``. Next time, retrieval/autotune skips the
search. This is the whole point of a library: marginal cost trends down.

Invariants enforced (§2.4): every measurement must cite a reproducible ``source``
run id and an ``arch``; un-sourced/arch-less entries are rejected. External
(untrusted) callers are read + verify only — write-back into the shared corpus
requires a server-side-rerun ``source`` (open question §11).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loader import get_card, registry_root, reset_cache
from .models import Measurement


def _card_path(card_id: str) -> Path:
    short = card_id.split("@", 1)[0]  # e.g. "dual_rmsnorm.triton"
    p = registry_root() / "impls" / f"{short}.card.json"
    if not p.exists():
        raise FileNotFoundError(f"no card file for {card_id!r} at {p}")
    return p


def record_measurement(
    impl_card_id: str,
    arch: str,
    shape: dict[str, int],
    dtype: str,
    source: str,
    *,
    knobs: dict[str, Any] | None = None,
    tflops: float | None = None,
    achieved_bw_pct: float | None = None,
    ms: float | None = None,
    trust: str = "trusted",
) -> dict[str, Any]:
    """Append a measurement to a card's ``perf.measured`` and persist it.

    Args:
        impl_card_id: e.g. "dual_rmsnorm.triton@1.0.0".
        arch, shape, dtype, knobs: the measurement point.
        source: a reproducible run id (e.g. ``verify``'s ``artifacts.run_id``).
            Required — un-sourced numbers are dropped (§2.4).
        trust: ``"trusted"`` (this runtime) or ``"external"``. External writes are
            rejected unless they carry a server-side-rerun ``source`` marker.

    Returns a summary dict; raises ``ValueError`` on invariant violation.
    """
    if not source:
        raise ValueError("measurement must cite a reproducible `source` run id (§2.4)")
    if not arch:
        raise ValueError("measurement must cite an `arch` (§2.4)")
    if trust == "external" and not source.startswith("server-rerun:"):
        raise ValueError(
            "external write-back requires a server-side-rerun source "
            "(prefix 'server-rerun:'); use verify() to obtain one (§11)"
        )

    path = _card_path(impl_card_id)
    with path.open() as f:
        doc = json.load(f)

    measured = doc.setdefault("perf", {}).setdefault("measured", [])
    # Replace any existing entry for the same (arch, shape, dtype, knobs) point.
    knobs = knobs or {}
    point_key = (arch, json.dumps(shape, sort_keys=True), dtype, json.dumps(knobs, sort_keys=True))
    measured = [
        m for m in measured
        if (m.get("arch"), json.dumps(m.get("shape", {}), sort_keys=True),
            m.get("dtype"), json.dumps(m.get("knobs", {}), sort_keys=True)) != point_key
    ]
    entry = {"arch": arch, "shape": shape, "dtype": dtype, "knobs": knobs, "source": source}
    if tflops is not None:
        entry["tflops"] = tflops
    if achieved_bw_pct is not None:
        entry["achieved_bw_pct"] = achieved_bw_pct
    if ms is not None:
        entry["ms"] = ms
    measured.append(entry)
    doc["perf"]["measured"] = measured

    with path.open("w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    reset_cache()

    return {"impl_card_id": impl_card_id, "recorded": entry, "total_measurements": len(measured)}


def measurement_view(impl_card_id: str) -> list[Measurement]:
    """Read-only view of a card's measurements."""
    return list(get_card(impl_card_id).measured)
