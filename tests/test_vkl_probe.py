# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CI enforcement for the vkl Phase 0 probe (docs/brainstorm/11 §7).

Loads ``scripts/vkl_probe.py`` as a module (it is not on the package path) and
asserts every probe check passes. If a check fails here, the contract-native
thesis is weaker than docs/brainstorm/02 claims — see docs/brainstorm/06 §D for
the scope-correction to run *before* building the lowering.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROBE_PATH = _REPO_ROOT / "scripts" / "vkl_probe.py"


@pytest.fixture(scope="module")
def probe():
    """Load the probe script as an importable module.

    Registered in ``sys.modules`` before exec so the ``@dataclass`` field-type
    resolution (which looks up ``sys.modules[cls.__module__]`` for stringized
    annotations) succeeds.
    """
    import sys
    spec = importlib.util.spec_from_file_location("vkl_probe", _PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vkl_probe"] = module
    spec.loader.exec_module(module)
    return module


def test_probe_module_loads(probe):
    """Sanity: the probe script is importable and exposes the four checks."""
    for name in ("check_roundtrip_spec", "check_cards_ingest",
                 "check_edit_decidability", "check_schema_finding", "run_all", "main"):
        assert hasattr(probe, name), f"probe missing {name!r}"


def _all_results(probe):
    """Flatten (section, name, passed, detail) for every probe check."""
    for section_name, results in probe.run_all():
        for name, passed, detail in results:
            yield section_name, name, passed, detail


def test_probe_runs_clean(probe):
    """Every probe check passes. On failure, report the section + detail."""
    failures = [
        f"[{sec}] {name} — {detail}"
        for sec, name, passed, detail in _all_results(probe)
        if not passed
    ]
    assert not failures, "vkl probe failures:\n  " + "\n  ".join(failures)


# --- per-section guards (so a regression points at the right thesis clause) ----

def test_roundtrip_spec(probe):
    """Section A: header -> emit -> validate -> ingest is byte-identical."""
    results = probe.check_roundtrip_spec()
    assert results, "section A produced no checks"
    bad = [n for n, p, _ in results if not p]
    assert not bad, f"round-trip failures: {bad}"


def test_cards_ingest(probe):
    """Section B: all three Impl Cards (triton/cuda/hip) pass schema + from_doc."""
    results = probe.check_cards_ingest()
    assert len(results) == 3, f"expected 3 cards, got {len(results)}"
    bad = [n for n, p, _ in results if not p]
    assert not bad, f"card ingest failures: {bad}"


def test_edit_decidability(probe):
    """Section C: every edit precondition is locally decidable, accept or reject."""
    results = probe.check_edit_decidability()
    # 4 accepted + 4 rejected = 8 checks; each must pass (rejections pass by being rejected).
    assert len(results) >= 8, f"expected >=8 edit checks, got {len(results)}"
    bad = [n for n, p, _ in results if not p]
    assert not bad, f"edit-decidability failures: {bad}"


def test_schema_finding(probe):
    """Section D: the closed schema rejects the namespaced extensions, as predicted."""
    results = probe.check_schema_finding()
    assert len(results) == 4, f"expected 4 schema findings, got {len(results)}"
    bad = [n for n, p, _ in results if not p]
    assert not bad, f"schema-finding failures: {bad}"


def test_main_exits_zero(probe):
    """The CLI entrypoint returns 0 when all checks pass."""
    assert probe.main() == 0
