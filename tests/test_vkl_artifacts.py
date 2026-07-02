# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Drift gate for registry JSON materialized from VKL sources."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from xkernels.vkl.artifacts import (
    check_registry_artifacts,
    emit_registry_artifacts,
    managed_short_names,
)

_REPO = Path(__file__).resolve().parents[1]


def test_vkl_managed_artifacts_do_not_drift():
    drifts = check_registry_artifacts(root=_REPO)
    assert drifts == []


def test_vkl_artifacts_are_marked_as_dsl_generated():
    names = managed_short_names(_REPO)
    assert "temperature_softmax" in names
    artifacts = emit_registry_artifacts(["temperature_softmax"], root=_REPO)
    spec = next(a.expected for a in artifacts if a.path.name == "temperature_softmax.spec.json")
    card = next(
        a.expected for a in artifacts if a.path.name == "temperature_softmax.triton.card.json"
    )
    assert spec["provenance"] == {
        "authored_by": "dsl",
        "source_path": "xkernels.vkl:temperature_softmax",
    }
    assert card["provenance"]["authored_by"] == "dsl"
    assert card["provenance"]["source_path"] == "xkernels.vkl:temperature_softmax"


def test_vkl_artifact_checker_detects_content_drift(tmp_path):
    for dirname in ("ops", "impls", "shape_sweeps"):
        src = _REPO / f"registry/{dirname}"
        dst = tmp_path / f"registry/{dirname}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)

    spec_path = tmp_path / "registry/ops/temperature_softmax.spec.json"
    doc = json.loads(spec_path.read_text())
    doc["name"] = "drifted"
    spec_path.write_text(json.dumps(doc, indent=2) + "\n")

    drifts = check_registry_artifacts(["temperature_softmax"], root=tmp_path)
    assert [(d.path.name, d.reason) for d in drifts] == [
        ("temperature_softmax.spec.json", "content drift")
    ]
