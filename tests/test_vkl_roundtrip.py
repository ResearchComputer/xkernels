# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 1 gate A: ``@kernel`` header → emit → validate → ingest (docs/brainstorm/11 §4).

The round-trip is the spine of the contract-native thesis: the header is a
*spelling* of the Op Spec, so emitting it and re-ingesting must be stable and
schema-valid against the REAL validators (not mocks). Also asserts the
auto-reference path resolves.
"""
from __future__ import annotations

import json

import pytest

from xkernels.registry.models import ImplCard, op_spec_from_doc
from xkernels.registry.schemas import validate_impl_card, validate_op_spec
from xkernels.vkl import auto, emit_card, emit_spec, spec_of
from xkernels.vkl.examples import dual_rmsnorm


@pytest.fixture(scope="module")
def spec():
    return spec_of(dual_rmsnorm)


def test_spec_emits_schema_valid_and_ingests(spec):
    doc = emit_spec(spec)
    validate_op_spec(doc)  # raises on any schema violation
    op = op_spec_from_doc(doc)
    assert op.id == spec.id
    assert op.kernel == spec.kernel
    assert op.canonical_op == spec.canonical_op
    assert op.constraints == spec.constraints
    assert op.numerics.reduce_dtype == spec.numerics.reduce_dtype


def test_card_emits_schema_valid_and_ingests(spec):
    for backend, target in spec.targets.items():
        doc = emit_card(spec, target)
        validate_impl_card(doc)  # raises on any schema violation
        card = ImplCard.from_doc(doc)
        assert card.implements == spec.id
        assert card.backend.value == backend
        assert card.arch.family == target.arch
        assert card.provenance["authored_by"] == "dsl"  # the Phase 1 enum value


def test_emit_is_stable(spec):
    """Emitting twice produces byte-identical JSON (canonical projection)."""
    doc1 = emit_spec(spec)
    doc2 = emit_spec(spec)
    assert json.dumps(doc1, sort_keys=True) == json.dumps(doc2, sort_keys=True)
    # same for the card (modulo the timestamp, which we pin)
    card1 = emit_card(spec, spec.targets["triton"], created="2026-06-30T00:00:00+00:00")
    card2 = emit_card(spec, spec.targets["triton"], created="2026-06-30T00:00:00+00:00")
    assert json.dumps(card1, sort_keys=True) == json.dumps(card2, sort_keys=True)


def test_auto_reference_path_resolves(spec):
    """The emitted numerics.reference path resolves to a runnable reference.

    For a DIRECT body the reference IS the body; for a TRACE body (like
    dual_rmsnorm) the reference is a wrapper that builds the IR + evaluates on
    torch. Either way, resolving the path yields a callable that takes inputs
    and returns the outputs.
    """
    assert spec.reference_path == "xkernels.vkl.auto:dual_rmsnorm"
    ref = auto.get_auto("dual_rmsnorm")
    assert callable(ref)
    # Running it on seeded inputs returns the spec's outputs in order.
    from xkernels.vkl import make_inputs

    inputs = make_inputs(spec, {"dtype": "fp32", "T": 4, "d1": 8, "d2": 6}, seed=0)
    out = ref(**inputs)
    assert len(out) == len(spec.outputs)


def test_emitted_constraints_are_decidable(spec):
    """Every emitted constraint is in the decidable subset (§2.4, reject-before-compile)."""
    from xkernels.registry.constraints import validate_decidable

    for c in spec.constraints:
        validate_decidable(c)  # raises UndecidableConstraintError if not decidable
