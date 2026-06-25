"""Skill outcome records + the governance loop (docs/library.md §7.3).

Every agent run that uses a skill emits an **outcome record**; these roll up into
each skill's ``metrics`` (success_rate, median_iterations, regression_count) and
feed the continuous loop: score → promote → revise → split/merge → deprecate.

This is the third compounding loop (§7.4): cards accumulate tunings; skills
accumulate outcome records; provenance links them. A skill that needs many
iterations is "expensive" even if it eventually works.

Storage is an append-only JSONL log at ``registry/skill_outcomes.jsonl`` — plain,
greppable, vendor-neutral (the bottom consumption tier). External/untrusted
callers are read + verify only; write-back here is for the integrated runtime.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .loader import registry_root

_VALID_RESULTS = {"success", "partial", "fail"}


def _outcomes_path() -> Path:
    return registry_root() / "skill_outcomes.jsonl"


def record_outcome(
    skill_id: str,
    version: str,
    task_signature: str,
    result: str,
    *,
    iterations: int = 0,
    run_id: str | None = None,
    final_tflops_vs_regime: float | None = None,
    failure_mode: str | None = None,
    trust: str = "trusted",
) -> dict[str, Any]:
    """Append a skill outcome record (§7.3).

    Args:
        skill_id: e.g. "tune-for-cdna@1.0.0".
        version: skill version used (must match the skill's frontmatter).
        task_signature: a stable string identifying the task (op+arch+shape+dtype).
        result: "success" | "partial" | "fail".
        iterations: number of skill-loop iterations to the result (cheaper = better).
        run_id: reproducible run id (e.g. a ``verify`` run_id) — strongly recommended.
        final_tflops_vs_regime: achieved perf as a fraction of the arch roofline regime.
        failure_mode: short tag when result != success (feeds the revise step §7.3.3).
        trust: "trusted" (this runtime) or "external". External writes are rejected
            (write-back is for the integrated runtime; §8.4 / §11).

    Returns the recorded record. Raises ``ValueError`` on an invalid result or an
    untrusted external write.
    """
    if result not in _VALID_RESULTS:
        raise ValueError(f"result must be one of {_VALID_RESULTS}, got {result!r}")
    if trust == "external":
        raise ValueError(
            "external callers may not write skill outcomes (§8.4); "
            "outcomes are recorded by the integrated runtime only"
        )
    record = {
        "skill_id": skill_id,
        "version": version,
        "task_signature": task_signature,
        "result": result,
        "iterations": int(iterations),
        "run_id": run_id,
        "final_tflops_vs_regime": final_tflops_vs_regime,
        "failure_mode": failure_mode,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _outcomes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def all_outcomes(skill_id: str | None = None) -> list[dict[str, Any]]:
    """Read all outcome records, optionally filtered by ``skill_id``."""
    path = _outcomes_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if skill_id is None or rec.get("skill_id") == skill_id:
            out.append(rec)
    return out


def skill_metrics(skill_id: str) -> dict[str, Any]:
    """Roll outcome records into a skill's metrics (§7.3.1).

    Returns {uses, success_rate, median_iterations, regression_count,
    failure_modes: {...}, versions: [...]}.
    ``regression_count`` counts fails on task_signatures that previously succeeded
    (the §7.3.3 revise trigger).
    """
    records = all_outcomes(skill_id)
    if not records:
        return {
            "skill_id": skill_id, "uses": 0, "success_rate": None,
            "median_iterations": None, "regression_count": 0,
            "failure_modes": {}, "versions": [],
        }
    uses = len(records)
    successes = sum(1 for r in records if r["result"] == "success")
    iters = [r["iterations"] for r in records if r.get("iterations") is not None]
    # regression: a fail on a task_signature with a prior success
    seen_success: set[str] = set()
    regression_count = 0
    failure_modes: dict[str, int] = {}
    # records are append-order == chronological
    for r in records:
        sig = r["task_signature"]
        if r["result"] == "success":
            seen_success.add(sig)
        elif r["result"] == "fail":
            if sig in seen_success:
                regression_count += 1
            fm = r.get("failure_mode") or "unspecified"
            failure_modes[fm] = failure_modes.get(fm, 0) + 1
    return {
        "skill_id": skill_id,
        "uses": uses,
        "success_rate": round(successes / uses, 3),
        "median_iterations": statistics.median(iters) if iters else None,
        "regression_count": regression_count,
        "failure_modes": failure_modes,
        "versions": sorted({r["version"] for r in records}),
    }


def reset_outcomes() -> None:
    """Clear the outcome log (test helper; never call in production)."""
    path = _outcomes_path()
    if path.exists():
        path.unlink()
