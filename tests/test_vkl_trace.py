# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase E (issue #73): the persisted tuning_trace — cross-task compounding.

The in-memory ``gate.TraceEntry`` dies with the task that produced it; the
persisted ``tuning_traces.jsonl`` store lets the **next** task read what was
already tried (``{edit, predicted, measured, rationale}`` keyed by
``(op, arch, shape, dtype, edit)``) and skip the dead-end or reuse the winner.
That is the whole point of the §6.2 loop: marginal cost trends down across tasks.

These tests demonstrate the two cross-task wins the issue's acceptance criterion
#4 names, both CPU-satisfiable (the PREDICTED half is closed-form; the
reject-reason half is the gate's own decidability):

  * **reject-dead-end avoidance** — task 1 hits a scratch-overflow reject on
    cdna3 and records it with a rationale; task 2 loads the schedule for the same
    point, sees the dead-end, and skips re-proposing the edit (citing the prior
    rationale instead of re-deriving the overflow).
  * **predicted-optimal reuse** — task 1 records the edit with the best
    predicted roofline; task 2 retrieves it and reuses the point rather than
    re-searching.

The MEASURED half (real ``verify`` ms) is GPU-gated; the cross-task *mechanism*
(persist + retrieve + cite) is what these tests pin, and it is fully CPU-doable.
"""
from __future__ import annotations

import json

import pytest

from xkernels.mcp_server import _dispatch
from xkernels.vkl import TraceEntry, prior_traces, record_trace, run_gate
from xkernels.vkl import trace as _trace
from xkernels.vkl.ir.schedule import Knob, MapTo, ScheduleIR, Tile

# ─── isolation: never touch the real registry/tuning_traces.jsonl ─────────────


@pytest.fixture
def tmp_trace_store(tmp_path, monkeypatch):
    """Point the trace store at a tmp path so tests never pollute the corpus."""
    target = tmp_path / "tuning_traces.jsonl"
    monkeypatch.setattr(_trace, "_traces_path", lambda: target)
    return target


@pytest.fixture
def sched_with_l5():
    """A minimal schedule with an L5 wgmma map + a declared knob (for run_gate)."""
    return ScheduleIR(
        nodes=(
            Tile(id="out", shape=(128, 128), level="L2"),
            MapTo(id="mma0", op_ref="mma_0", level="L5",
                  instruction="wgmma", instr_shape=(64, 128, 16)),
        ),
        knobs={"num_stages": Knob(name="num_stages", value=2, choices=(2, 3, 4))},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §1  TraceEntry carries the {edit, predicted, measured, rationale} triple
# ═══════════════════════════════════════════════════════════════════════════════


class TestTraceEntryTriple:
    def test_rationale_field_round_trips(self):
        te = TraceEntry(
            step=1, edit="setknob", target="BLOCK_M", args={"value": 128},
            check="ok", rationale="best predicted roofline at this point",
            predicted={"bottleneck": "compute"},
            measured={"ms": 0.42},
        )
        d = te.to_dict()
        assert d["rationale"] == "best predicted roofline at this point"
        assert d["predicted"]["bottleneck"] == "compute"
        assert d["measured"]["ms"] == 0.42
        json.dumps(d)  # JSON-serializable (it goes into the persisted store)

    def test_rationale_omitted_when_empty(self):
        te = TraceEntry(step=1, edit="x", target="t", args={}, check="ok")
        assert "rationale" not in te.to_dict()

    def test_run_gate_predict_hook_fills_predicted(self, sched_with_l5):
        """run_gate's predict hook populates the predicted half of ok entries."""
        from xkernels.vkl import SetKnob

        def predict(_sched, _arch):
            return {"scratch_bytes": 1234, "overflows_scratch": False}

        res = run_gate(
            [SetKnob("num_stages", 3)], sched_with_l5, "nvidia_sm90", predict=predict
        )
        ok_entries = [t for t in res.trace if t.check == "ok"]
        assert ok_entries and ok_entries[0].predicted["scratch_bytes"] == 1234
        # rejects get no prediction (no resulting schedule to cost)
        res2 = run_gate(
            [SetKnob("num_stages", 99)], sched_with_l5, "nvidia_sm90", predict=predict
        )
        rej = [t for t in res2.trace if t.check == "reject"][0]
        assert rej.predicted == {}


# ═══════════════════════════════════════════════════════════════════════════════
# §2  record_trace persistence + retrieval (the store)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRecordAndRetrieve:
    def test_record_then_retrieve_by_point(self, tmp_trace_store):
        edit = {"kind": "set_knob", "name": "BLOCK_M", "value": 128}
        rec = record_trace(
            "gemm_bf16@1.0.0", "nvidia_sm90", edit,
            shape={"M": 1024, "N": 1024, "K": 512}, dtype="bf16",
            check="ok", rationale="predicted-optimal compute-bound point",
            predicted={"bottleneck": "compute"},
        )
        assert rec["edit_key"] == "set_knob{name=BLOCK_M,value=128}"
        got = prior_traces("gemm_bf16@1.0.0", "nvidia_sm90",
                           shape={"M": 1024, "N": 1024, "K": 512}, dtype="bf16")
        assert len(got) == 1
        assert got[0]["rationale"] == "predicted-optimal compute-bound point"
        assert got[0]["predicted"]["bottleneck"] == "compute"

    def test_latest_record_for_a_key_wins(self, tmp_trace_store):
        edit = {"kind": "set_knob", "name": "num_stages", "value": 3}
        shape, dtype = {"M": 512, "N": 512, "K": 512}, "bf16"
        # task 1 records a guess; task 2 re-tunes the SAME point — latest wins
        record_trace("gemm_bf16@1.0.0", "nvidia_sm90", edit,
                     shape=shape, dtype=dtype, rationale="first guess",
                     measured={"ms": 1.0})
        record_trace("gemm_bf16@1.0.0", "nvidia_sm90", edit,
                     shape=shape, dtype=dtype, rationale="re-tuned winner",
                     measured={"ms": 0.5})
        got = prior_traces("gemm_bf16@1.0.0", "nvidia_sm90", shape=shape, dtype=dtype)
        assert len(got) == 1  # dedup, not append
        assert got[0]["rationale"] == "re-tuned winner"
        assert got[0]["measured"]["ms"] == 0.5

    def test_retrieval_scopes_by_shape_and_dtype(self, tmp_trace_store):
        edit = {"kind": "set_knob", "name": "BLOCK_M", "value": 64}
        record_trace("gemm_bf16@1.0.0", "nvidia_sm90", edit,
                     shape={"M": 1024, "N": 1024, "K": 512}, dtype="bf16")
        record_trace("gemm_bf16@1.0.0", "nvidia_sm90", edit,
                     shape={"M": 4096, "N": 4096, "K": 512}, dtype="bf16")
        # a different point sees nothing; the (op, arch) pair sees both
        assert prior_traces("gemm_bf16@1.0.0", "nvidia_sm90",
                            shape={"M": 2048, "N": 2048, "K": 512}, dtype="bf16") == []
        assert len(prior_traces("gemm_bf16@1.0.0", "nvidia_sm90")) == 2

    def test_external_write_requires_server_rerun_source(self, tmp_trace_store):
        with pytest.raises(ValueError, match="server-side-rerun"):
            record_trace("gemm_bf16@1.0.0", "nvidia_sm90",
                         {"kind": "set_knob", "name": "BLOCK_M", "value": 128},
                         shape={"M": 512, "N": 512, "K": 512}, dtype="bf16",
                         trust="external", source="verify:abc")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  The cross-task wins (acceptance criterion #4) — both CPU-satisfiable
# ═══════════════════════════════════════════════════════════════════════════════


class TestCrossTaskCompounding:
    def test_task2_avoids_reject_deadend_task1_hit(self, tmp_trace_store):
        """The reject-reason dead-end path: task 1 records a scratch-overflow
        reject on cdna3; task 2 loads the schedule, sees the dead-end with its
        rationale, and skips re-proposing the edit (cites the prior rationale
        instead of re-deriving the overflow)."""
        shape, dtype = {"M": 512, "N": 512, "K": 512}, "bf16"
        edit = {  # depth×tile_bytes > cdna3's 64K LDS budget -> gate rejects
            "kind": "add_stage", "stage_id": "s_extra",
            "producer_ref": "a_tile", "depth": 2, "tile_bytes": 40000,
        }

        # ── Task 1: check the edit, observe the reject, record the dead-end ──
        verdict = _dispatch("vkl_check_edit", {
            "spec_id": "gemm_bf16", "arch": "amd_cdna3", "edit": edit,
        })
        assert verdict["ok"] is False
        assert "scratch" in verdict["reason"] and "budget" in verdict["reason"]
        _dispatch("record_trace", {
            "spec_id": "gemm_bf16", "arch": "amd_cdna3", "edit": edit,
            "shape": shape, "dtype": dtype,
            "check": "reject", "reason": verdict["reason"],
            "rationale": "depth=2 x 40KB overflows cdna3 64K LDS — skip this stage",
        })

        # ── Task 2: load the schedule for the SAME point — the dead-end is there ──
        view = _dispatch("vkl_load_schedule", {
            "spec_id": "gemm_bf16", "arch": "amd_cdna3", "shape": shape, "dtype": dtype,
        })
        prior = view["prior_traces"]
        deadends = [p for p in prior if p["check"] == "reject"]
        assert len(deadends) == 1
        assert deadends[0]["edit"] == edit
        assert "overflow" in deadends[0]["rationale"].lower()
        # Task 2 cites the prior rationale rather than re-deriving the dead-end:
        # it would NOT re-propose this edit because the prior record says reject.
        assert any(_trace._canonical_edit(edit) == p["edit_key"] for p in deadends)

    def test_task2_reuses_predicted_optimal_task1_found(self, tmp_trace_store):
        """The predicted-optimal reuse path: task 1 records the edit with the
        best predicted roofline at a point; task 2 retrieves it and reuses the
        point rather than re-searching. Fully CPU-doable (the predicted half is
        closed-form); the live MEASURED-optimal variant is the GPU gate."""
        point = {"M": 1024, "N": 1024, "K": 512, "dtype": "bf16"}
        edit = {"kind": "set_knob", "name": "BLOCK_M", "value": 128}

        # ── Task 1: record the predicted-optimal edit (auto-predicted by the
        #    cost model — the agent did not even need to call read_cost) ──
        rec = _dispatch("record_trace", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90", "edit": edit,
            "point": point, "check": "ok",
            "rationale": "highest predicted roofline (compute-bound) — reuse, do not re-search",
        })
        # the predicted half was auto-filled from the closed-form cost model
        assert rec["predicted"], "predicted half should be auto-filled (CPU-doable)"
        assert "scratch_bytes" in rec["predicted"]

        # ── Task 2: load the schedule for the SAME point — the winner is there ──
        view = _dispatch("vkl_load_schedule", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "shape": {"M": 1024, "N": 1024, "K": 512}, "dtype": "bf16",
        })
        winners = [p for p in view["prior_traces"] if p["check"] == "ok"]
        assert len(winners) == 1
        assert winners[0]["edit"] == edit
        assert "reuse" in winners[0]["rationale"].lower()
        assert winners[0]["predicted"]  # cited prior prediction

    def test_cold_start_point_has_no_prior_traces(self, tmp_trace_store):
        """A brand-new point sees an empty prior_traces list (no crash, no noise)."""
        view = _dispatch("vkl_load_schedule", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "shape": {"M": 8192, "N": 8192, "K": 1024}, "dtype": "bf16",
        })
        assert view["prior_traces"] == []
        # the schedule itself is still returned (prior_traces is additive)
        assert "nodes" in view and "binding" in view
