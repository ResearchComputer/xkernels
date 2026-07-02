# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 1 gate D: edit preconditions are locally decidable (docs/brainstorm/11 §4).

Each edit primitive has ACCEPTED cases (preconditions pass) and REJECTED cases
(preconditions fail with a reason). The reject reasons are asserted — they are
the training signal an agent reads to skip dead-ends (docs/brainstorm/10 §5).

The load-bearing property (the Phase 0 finding): gates are STATEFUL — a
``Retile``'s divisibility check only bites once an L5 matrix-engine map is
present in the IR. So each test hands the gate the IR state where its
precondition applies.
"""
from __future__ import annotations

import pytest

from xkernels.vkl import (
    AddStage,
    Knob,
    MapTo,
    MapTo_,
    Ok,
    Reject,
    Retile,
    ScheduleIR,
    SetKnob,
    Tile,
    run_gate,
)

# --- fixtures: a GEMM-ish schedule on sm_90 WITH an L5 wgmma map present -------

@pytest.fixture
def sched_with_l5():
    """Schedule with an L5 wgmma map — so Retile divisibility bites."""
    return ScheduleIR(
        nodes=(
            Tile(id="out", shape=(128, 128), level="L2"),
            MapTo(id="mma0", op_ref="mma_0", level="L5",
                  instruction="wgmma", instr_shape=(64, 128, 16)),
        ),
        knobs={"num_stages": Knob(name="num_stages", value=2, choices=(2, 3, 4))},
    )


@pytest.fixture
def sched_no_l5():
    """Schedule with NO L5 map — Retile divisibility does not apply yet."""
    return ScheduleIR(
        nodes=(Tile(id="out", shape=(128, 128), level="L2"),),
        knobs={"num_stages": Knob(name="num_stages", value=2, choices=(2, 3, 4))},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SetKnob
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetKnob:
    def test_accept_in_choices(self, sched_with_l5):
        assert isinstance(SetKnob("num_stages", 3).check(sched_with_l5, "nvidia_sm90"), Ok)

    def test_reject_out_of_choices(self, sched_with_l5):
        r = SetKnob("num_stages", 5).check(sched_with_l5, "nvidia_sm90")
        assert isinstance(r, Reject) and "not in declared choices" in r.reason

    def test_reject_undeclared(self, sched_with_l5):
        r = SetKnob("block_m", 64).check(sched_with_l5, "nvidia_sm90")
        assert isinstance(r, Reject) and "undeclared knob" in r.reason

    def test_apply_binds_value(self, sched_with_l5):
        ir = SetKnob("num_stages", 3).apply(sched_with_l5)
        assert ir.knobs["num_stages"].value == 3
        # immutable: original IR unchanged
        assert sched_with_l5.knobs["num_stages"].value == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Retile (STATEFUL — the Phase 0 finding)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetile:
    def test_accept_divisible_when_l5_present(self, sched_with_l5):
        # 256 % 64 == 0 → ok
        assert isinstance(Retile("out", (256, 128)).check(sched_with_l5, "nvidia_sm90"), Ok)

    def test_reject_not_divisible_when_l5_present(self, sched_with_l5):
        # 96 % 64 != 0 → reject (the gate bites because an L5 map is present)
        r = Retile("out", (96, 128)).check(sched_with_l5, "nvidia_sm90")
        assert isinstance(r, Reject) and "not divisible by L5 wgmma native m=64" in r.reason

    def test_accept_anything_when_no_l5(self, sched_no_l5):
        """No L5 map → no divisibility constraint → 96 is fine (stateful gate)."""
        assert isinstance(Retile("out", (96, 128)).check(sched_no_l5, "nvidia_sm90"), Ok)

    def test_reject_missing_tile(self, sched_with_l5):
        r = Retile("nope", (64,)).check(sched_with_l5, "nvidia_sm90")
        assert isinstance(r, Reject) and "no tile with id" in r.reason

    def test_reject_wrong_arch_native(self, sched_with_l5):
        """mfma native m=32 on cdna3: 96 % 32 == 0 -> ok; 50 % 32 != 0 -> reject.

        Uses a cdna3-coherent schedule (mfma L5 map) — mixing a wgmma map with
        cdna3 is incoherent (wgmma isn't legal there); MapTo_'s check catches
        that, Retile's check just does divisibility against the mapped engine.
        """
        cdna3_sched = ScheduleIR(
            nodes=(
                Tile(id="out", shape=(128, 128), level="L2"),
                MapTo(id="mma0", op_ref="mma_0", level="L5",
                      instruction="mfma", instr_shape=(32, 128, 16)),
            ),
        )
        # 96 % 32 == 0 -> ok
        assert isinstance(Retile("out", (96, 128)).check(cdna3_sched, "amd_cdna3"), Ok)
        # 50 % 32 != 0 -> reject
        r = Retile("out", (50, 128)).check(cdna3_sched, "amd_cdna3")
        assert isinstance(r, Reject) and "m=32" in r.reason

    def test_apply_resizes_tile(self, sched_with_l5):
        ir = Retile("out", (256, 128)).apply(sched_with_l5)
        assert ir.by_id("out").shape == (256, 128)
        assert sched_with_l5.by_id("out").shape == (128, 128)  # immutable


# ═══════════════════════════════════════════════════════════════════════════════
# MapTo_ + AddStage (Phase 2 primitives, but check is realized now)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMapTo:
    def test_reject_wrong_vendor_instruction(self, sched_with_l5):
        """mfma is not legal on sm_90 (vendor honesty, docs/brainstorm/10 §5 row 4)."""
        r = MapTo_("mma1", "mma_0", "L5", "mfma").check(sched_with_l5, "nvidia_sm90")
        assert isinstance(r, Reject) and "not legal for nvidia_sm90" in r.reason

    def test_accept_legal_instruction(self, sched_no_l5):
        assert isinstance(
            MapTo_("mma1", "mma_0", "L5", "wgmma").check(sched_no_l5, "nvidia_sm90"), Ok
        )


class TestAddStage:
    def test_reject_scratch_overflow(self):
        """Stage overflows the sm_90 scratch budget (228K)."""
        ir = ScheduleIR()  # empty
        big = AddStage("s", "load_a", depth=4, tile_bytes=60 * 1024)  # 240K > 228K
        r = big.check(ir, "nvidia_sm90")
        assert isinstance(r, Reject) and "budget" in r.reason

    def test_accept_fits(self):
        ir = ScheduleIR()
        ok = AddStage("s", "load_a", depth=2, tile_bytes=32 * 1024)  # 64K < 228K
        assert isinstance(ok.check(ir, "nvidia_sm90"), Ok)

    def test_any_target_has_no_budget(self):
        """The portable 'any' target has no scratch budget → never overflows."""
        ir = ScheduleIR()
        big = AddStage("s", "load_a", depth=100, tile_bytes=10 ** 9)
        assert isinstance(big.check(ir, "any"), Ok)


# ═══════════════════════════════════════════════════════════════════════════════
# The gate runs a SEQUENCE and produces a trace (the compounding artifact)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateSequence:
    def test_mixed_sequence_applies_and_records(self, sched_with_l5):
        edits = [
            SetKnob("num_stages", 3),          # ok
            SetKnob("num_stages", 5),          # reject
            Retile("out", (256, 128)),         # ok
            Retile("out", (96, 128)),          # reject (stateful: L5 present)
        ]
        res = run_gate(edits, sched_with_l5, "nvidia_sm90")
        assert res.applied == 2 and res.rejected == 2
        assert len(res.trace) == 4
        # the final IR reflects the two applied edits
        assert res.final_ir.knobs["num_stages"].value == 3
        assert res.final_ir.by_id("out").shape == (256, 128)
        # the trace records reject reasons (training signal)
        reasons = [t.reason for t in res.trace if t.check == "reject"]
        assert any("not in declared choices" in r for r in reasons)
        assert any("not divisible by L5" in r for r in reasons)

    def test_trace_is_json_serializable(self, sched_with_l5):
        """The trace goes into provenance.tuning_trace — must be JSON-serializable."""
        import json

        res = run_gate([SetKnob("num_stages", 3)], sched_with_l5, "nvidia_sm90")
        blob = json.dumps([t.to_dict() for t in res.trace])
        assert json.loads(blob)[0]["edit"] == "setknob"
        assert json.loads(blob)[0]["check"] == "ok"

    def test_stateful_ordering(self, sched_no_l5, sched_with_l5):
        """Retile(96) is rejected AFTER a MapTo(wgmma) but accepted before it."""
        # sequence 1: retile(96) then map(wgmma) — retile ok (no L5 yet)
        r1 = run_gate([Retile("out", (96, 128)), MapTo_("m", "mma_0", "L5", "wgmma")],
                      sched_no_l5, "nvidia_sm90")
        assert r1.trace[0].check == "ok"
        # sequence 2: map(wgmma) then retile(96) — retile rejected (L5 now present)
        r2 = run_gate([MapTo_("m", "mma_0", "L5", "wgmma"), Retile("out", (96, 128))],
                      sched_no_l5, "nvidia_sm90")
        assert r2.trace[1].check == "reject"
