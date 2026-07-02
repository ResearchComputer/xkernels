# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 1 gate C: the DSL-authored dual_rmsnorm faithfully spells the hand-written spec.

The strongest form of "the header is a spelling of the contract": the emitted
Op Spec agrees with the hand-written ``registry/ops/dual_rmsnorm.spec.json`` on
every CONTRACT field (constraints, numerics, tensor decls, canonical_op,
fusions, shape_sweep). The id matches too — the DSL authors the SAME op.

Fields that legitimately differ: ``numerics.reference`` (the DSL owns its own
auto-reference path) and ``perf.measured`` (the DSL card starts empty; the
hand card has real benchmarks). These are asserted, not hand-waved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from xkernels.registry.models import ImplCard, op_spec_from_doc
from xkernels.vkl import emit_card, emit_spec, spec_of
from xkernels.vkl.examples import dual_rmsnorm

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def emitted_spec():
    return emit_spec(spec_of(dual_rmsnorm))


@pytest.fixture(scope="module")
def hand_spec():
    return json.loads((_REPO / "registry/ops/dual_rmsnorm.spec.json").read_text())


@pytest.fixture(scope="module")
def hand_card():
    return json.loads((_REPO / "registry/impls/dual_rmsnorm.triton.card.json").read_text())


def test_same_op_id(emitted_spec, hand_spec):
    """The DSL authors the SAME op (same id), not a sibling."""
    assert emitted_spec["id"] == hand_spec["id"]
    assert emitted_spec["kernel"] == hand_spec["kernel"]


def test_contract_fields_match(emitted_spec, hand_spec):
    """The header projects to identical contract fields (the spelling claim)."""
    for field in ("constraints", "preconditions", "shape_sweep", "op", "inputs", "outputs"):
        assert emitted_spec[field] == hand_spec[field], (
            f"contract field {field!r} drifted"
        )


def test_numerics_match_except_reference(emitted_spec, hand_spec):
    """Tolerances + reduce_dtype match; only the reference import path differs (by design)."""
    e_num, h_num = emitted_spec["numerics"], hand_spec["numerics"]
    for field in ("rtol", "atol", "reduce_dtype", "cross_backend_rtol", "by_dtype", "notes"):
        assert e_num[field] == h_num[field], f"numerics.{field} drifted"
    # The reference path is the ONE field that differs — and it's the DSL's own path.
    assert e_num["reference"] == "xkernels.vkl.auto:dual_rmsnorm"
    assert h_num["reference"] == "xkernels.ops.norm.reference:dual_rmsnorm_ref"


def test_emitted_card_matches_hand_card_shape():
    """The emitted triton card matches the hand card's arch/backend/roofline shape."""
    spec = spec_of(dual_rmsnorm)
    emitted = emit_card(spec, spec.targets["triton"])
    hand = json.loads((_REPO / "registry/impls/dual_rmsnorm.triton.card.json").read_text())
    for field in ("implements", "backend"):
        assert emitted[field] == hand[field]
    assert emitted["arch"] == hand["arch"]  # family=any, requires=[], wave_size=0, scratch
    assert emitted["perf"]["roofline"] == hand["perf"]["roofline"]
    # The DSL card starts with empty measurements (the hand card has real ones).
    assert emitted["perf"]["measured"] == []
    assert len(hand["perf"]["measured"]) >= 1
    # And the DSL card records its provenance honestly.
    assert emitted["provenance"]["authored_by"] == "dsl"
    assert hand["provenance"]["authored_by"] == "human"


def test_emitted_artifacts_round_trip_through_real_validators(emitted_spec):
    """The emitted spec + card pass the real schema validators + dataclass ingest."""
    from xkernels.registry.schemas import validate_impl_card, validate_op_spec

    spec = spec_of(dual_rmsnorm)
    validate_op_spec(emitted_spec)
    op_spec_from_doc(emitted_spec)
    card = emit_card(spec, spec.targets["triton"])
    validate_impl_card(card)
    ImplCard.from_doc(card)
