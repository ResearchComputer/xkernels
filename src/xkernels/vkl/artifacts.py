# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Emit and drift-check registry JSON materialized from VKL sources."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .emit import emit_card, emit_reference_card, emit_spec
from .override import emit_override_card
from .surface import KernelSpec, spec_of

REPO_ROOT = Path(__file__).resolve().parents[3]
VKL_REFERENCE_PREFIX = "xkernels.vkl.auto:"


@dataclass(frozen=True)
class Artifact:
    """One generated registry artifact and its target path."""

    path: Path
    expected: dict[str, Any]
    current: dict[str, Any] | None

    @property
    def is_drifted(self) -> bool:
        return self.current != self.expected


@dataclass(frozen=True)
class Drift:
    """A missing or stale generated artifact."""

    path: Path
    reason: str


def managed_short_names(root: Path = REPO_ROOT) -> list[str]:
    """VKL-managed op names discovered from checked-in Op Specs.

    A spec is managed when its reference path points at the VKL auto-reference
    namespace. This keeps hand-owned specs that merely have a VKL sibling, such
    as ``dual_rmsnorm``, outside this drift gate.
    """
    out: list[str] = []
    for path in sorted((root / "registry/ops").glob("*.spec.json")):
        doc = _read_json(path)
        reference = str(doc.get("numerics", {}).get("reference", ""))
        if reference.startswith(VKL_REFERENCE_PREFIX):
            out.append(reference.removeprefix(VKL_REFERENCE_PREFIX))
    return out


def spec_for_short_name(short_name: str) -> KernelSpec:
    """Resolve a VKL example by registry short name."""
    from . import examples

    try:
        fn = getattr(examples, short_name)
    except AttributeError as e:
        raise KeyError(f"no VKL example named {short_name!r}") from e
    return spec_of(fn)


def emit_registry_artifacts(
    short_names: list[str] | None = None,
    *,
    root: Path = REPO_ROOT,
) -> list[Artifact]:
    """Return generated Op Spec / Impl Card artifacts for VKL-managed ops."""
    names = short_names if short_names is not None else managed_short_names(root)
    artifacts: list[Artifact] = []
    for short_name in names:
        spec = spec_for_short_name(short_name)
        artifacts.append(_artifact(root / f"registry/ops/{short_name}.spec.json", emit_spec(spec)))
        artifacts.append(
            _artifact(
                root / f"registry/impls/{short_name}.reference.card.json",
                emit_reference_card(spec),
            )
        )
        for backend, target in spec.targets.items():
            artifacts.append(
                _artifact(
                    root / f"registry/impls/{short_name}.{backend}.card.json",
                    emit_card(spec, target),
                )
            )
        for override in spec.overrides:
            artifacts.append(
                _artifact(
                    root / f"registry/impls/{short_name}.{override.backend}.card.json",
                    emit_override_card(spec, override),
                )
            )
    return artifacts


def check_registry_artifacts(
    short_names: list[str] | None = None,
    *,
    root: Path = REPO_ROOT,
) -> list[Drift]:
    """Return drift records for missing or stale generated artifacts."""
    drifts: list[Drift] = []
    for artifact in emit_registry_artifacts(short_names, root=root):
        if artifact.current is None:
            drifts.append(Drift(artifact.path, "missing"))
        elif artifact.is_drifted:
            drifts.append(Drift(artifact.path, "content drift"))
    for short_name in short_names if short_names is not None else managed_short_names(root):
        sweep = root / f"registry/shape_sweeps/{short_name}.sweep.json"
        if not sweep.exists():
            drifts.append(Drift(sweep, "missing shape sweep"))
    return drifts


def write_registry_artifacts(
    short_names: list[str] | None = None,
    *,
    root: Path = REPO_ROOT,
) -> list[Path]:
    """Write generated artifacts and return the paths that changed."""
    changed: list[Path] = []
    for artifact in emit_registry_artifacts(short_names, root=root):
        if artifact.is_drifted:
            artifact.path.parent.mkdir(parents=True, exist_ok=True)
            artifact.path.write_text(_canonical_json(artifact.expected))
            changed.append(artifact.path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write", "list"))
    parser.add_argument(
        "short_names",
        nargs="*",
        help="Optional VKL op short names to limit scope.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Defaults to the current xkernels checkout.",
    )
    args = parser.parse_args(argv)
    names = args.short_names or None
    root = args.root.resolve()

    if args.command == "list":
        for name in managed_short_names(root):
            print(name)
        return 0
    if args.command == "write":
        changed = write_registry_artifacts(names, root=root)
        for path in changed:
            print(path.relative_to(root))
        return 0

    drifts = check_registry_artifacts(names, root=root)
    for drift in drifts:
        print(f"{drift.path.relative_to(root)}: {drift.reason}", file=sys.stderr)
    return 1 if drifts else 0


def _artifact(path: Path, emitted: dict[str, Any]) -> Artifact:
    current = _read_json(path) if path.exists() else None
    expected = _preserve_materialized_state(emitted, current)
    return Artifact(path=path, expected=expected, current=current)


def _preserve_materialized_state(
    emitted: dict[str, Any],
    current: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep mutable card state that is not owned by the VKL header/body."""
    if current is None or "backend" not in emitted:
        return emitted

    expected = json.loads(json.dumps(emitted))
    current_provenance = current.get("provenance", {})
    expected_provenance = expected.setdefault("provenance", {})
    if "created" in current_provenance:
        expected_provenance["created"] = current_provenance["created"]
    if "tuning_trace" in current_provenance:
        expected_provenance["tuning_trace"] = current_provenance["tuning_trace"]
    if "skill_used" in current_provenance:
        expected_provenance["skill_used"] = current_provenance["skill_used"]
    if "perf" in current and "measured" in current["perf"]:
        expected.setdefault("perf", {})["measured"] = current["perf"]["measured"]
    return expected


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canonical_json(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
