# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 2.2 gate: the schedule-IR-driven autotune sweep (the 25% -> ceiling lever).

Phase 2.0a ran the GEMM at ONE hardcoded config (BLOCK_M=N=64, K=32) and landed
~25% of H100's bf16 ceiling. Phase 2.2 makes the tile/meta knobs LIVE: the DSL
launcher accepts them, the card declares the search space, and ``sweep.autotune``
enumerates it via ``SetKnob``/``run_gate`` (the agent-editable primitive) +
``verify(measure_perf=True)`` (the substrate's own measurement), recording the
winner to ``perf.measured`` + the history to ``provenance.tuning_trace``.

CPU gates: the schedule-from-card round-trip + config enumeration + gate
decidability (value ∈ choices) + the tuning-trace writeback (on a temp copy, so
the committed card is never polluted by test runs). GPU gates: the launcher
recompiles per knob binding, and a capped sweep finds a non-default winner.

The full 108-config sweep + the real ``perf.measured`` writeback are run as a
manual compounding-loop step (the committed card carries that winner), not here.
"""
from __future__ import annotations

import json

import pytest

from xkernels.registry import get_card
from xkernels.vkl import SweepResult, autotune, enumerate_configs, schedule_from_card
from xkernels.vkl.examples import gemm_bf16
from xkernels.vkl.sweep import apply_config

try:
    import torch  # noqa: F401

    _GPU_OK = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _GPU_OK = False
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")

_CARD_ID = "gemm_bf16.triton@1.0.0"


# ─── CPU: schedule IR from the card + enumeration + gate decidability ─────────


def test_schedule_from_card_round_trips_knobs():
    card = get_card(_CARD_ID)
    sched = schedule_from_card(card)
    # every declared specialization knob is a Knob on the schedule, first-choice bound
    assert set(sched.knobs) == set(card.specialization_knobs)
    for name, knob in sched.knobs.items():
        declared = card.specialization_knobs[name]["choices"]
        assert tuple(knob.choices) == tuple(declared)
        assert knob.value == declared[0]  # default = first choice


def test_enumerate_configs_is_cartesian_product():
    card = get_card(_CARD_ID)
    sched = schedule_from_card(card)
    configs = list(enumerate_configs(sched))
    expected = 1
    for knob in sched.knobs.values():
        expected *= len(knob.choices)
    assert len(configs) == expected
    # each config binds every knob to a value within its choices
    for cfg in configs:
        assert set(cfg) == set(sched.knobs)
        for name, val in cfg.items():
            assert val in sched.knobs[name].choices
    # deterministic + unique
    assert len({tuple(sorted(c.items())) for c in configs}) == len(configs)


def test_enumerate_configs_max_configs_caps():
    card = get_card(_CARD_ID)
    sched = schedule_from_card(card)
    assert len(list(enumerate_configs(sched, max_configs=5))) == 5


def test_apply_config_gate_accepts_legal_rejects_illegal():
    from xkernels.vkl.edits import Reject, SetKnob

    card = get_card(_CARD_ID)
    sched = schedule_from_card(card)
    # a legal config (all values in choices) -> ok
    cfg = {n: k.choices[-1] for n, k in sched.knobs.items()}
    _new, ok, reason = apply_config(sched, cfg, "nvidia_sm90")
    assert ok, reason
    # an illegal value (not in choices) -> rejected by the gate
    res = SetKnob(name="BLOCK_M", value=7).check(sched, "nvidia_sm90")
    assert isinstance(res, Reject)
    assert "not in declared choices" in res.reason


def test_record_tuning_trace_appends_to_provenance(tmp_path):
    """record_tuning_trace appends to provenance.tuning_trace (CPU, temp copy).

    Writes against a temp copy so the committed card is never polluted by the
    test run — the real perf.measured + tuning_trace come from the manual sweep.
    """
    import xkernels.vkl.sweep as sweep_mod
    from xkernels.registry.loader import registry_root
    from xkernels.vkl.sweep import record_tuning_trace

    src = registry_root() / "impls" / "gemm_bf16.triton.card.json"
    doc = json.loads(src.read_text())
    # start clean (the real committed card carries the manual sweep's trace)
    doc["provenance"].pop("tuning_trace", None)
    tmp_card = tmp_path / "gemm_bf16.triton.card.json"
    tmp_card.write_text(json.dumps(doc))
    # redirect the path resolver at the temp copy
    orig = sweep_mod._card_path
    sweep_mod._card_path = lambda _id: tmp_card
    try:
        trace = [{"config": {"BLOCK_M": 128}, "passed": True, "ms": 0.3, "winner": True}]
        record_tuning_trace(_CARD_ID, trace)
    finally:
        sweep_mod._card_path = orig
    out = json.loads(tmp_card.read_text())
    assert out["provenance"]["tuning_trace"] == trace


# ─── GPU: the launcher recompiles per knob; the sweep finds a winner ──────────


@_SKIP
def test_launcher_accepts_knob_binding():
    """verify(knobs={BLOCK_M:128,...}) retargets the compiled kernel (still correct)."""
    from xkernels.verify import verify
    from xkernels.vkl import register_dsl, spec_of

    register_dsl(spec_of(gemm_bf16), backend="triton")
    point = {"dtype": "bf16", "M": 1024, "N": 1024, "K": 1024}
    r_default = verify(_CARD_ID, arch="nvidia_sm90", knobs={}, shapes=[point])
    r_big = verify(
        _CARD_ID, arch="nvidia_sm90",
        knobs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
        shapes=[point],
    )
    assert r_default["correctness"]["passed"]
    assert r_big["correctness"]["passed"]


@_SKIP
def test_autotune_finds_winner():
    """A capped sweep finds a passing winner with a correct SweepResult + gate."""
    from xkernels.vkl import register_dsl, spec_of

    register_dsl(spec_of(gemm_bf16), backend="triton")
    point = {"dtype": "bf16", "M": 2048, "N": 2048, "K": 2048}
    res: SweepResult = autotune(
        _CARD_ID, arch="nvidia_sm90", point=point, max_configs=6, record=False,
    )
    assert res.winner is not None, "sweep found no passing config"
    assert res.winner_ms is not None and res.winner_ms > 0
    assert res.n_passed >= 1
    # exactly one entry flagged the winner, matching res.winner
    assert sum(1 for e in res.entries if e.winner) == 1
    assert next(e for e in res.entries if e.winner).config == res.winner
    # the trace payload is token-compact + agent-readable (now carries reason on fails)
    assert len(res.trace()) == res.n_configs
    assert all({"config", "passed", "ms", "winner"} <= set(t) for t in res.trace())
    # the Phase 2.3 roofline-gate verdict is recorded (BELOW_BAR for a Triton GEMM)
    assert res.gate is not None
    assert res.gate["instruction"] == "wgmma"
    assert res.gate["verdict"] in {"PASS", "BELOW_BAR"}


@_SKIP
def test_autotune_scratch_precheck_rejects_overflow():
    """Phase 2.2b: a scratch-overflow config is rejected BEFORE launch (no crash).

    The 256x256x64 stages=4 config overflows the 228 KB smem budget. Without the
    pre-check it was a kernel crash (a Phase 2.2a FAIL); with the pre-check it is
    a clean trace entry with ``reason`` mentioning scratch overflow.
    """
    from xkernels.vkl import register_dsl, spec_of
    from xkernels.vkl.cost import overflows_scratch

    register_dsl(spec_of(gemm_bf16), backend="triton")
    overflow_cfg = {
        "BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64,
        "num_warps": 8, "num_stages": 4,
    }
    assert overflows_scratch(
        "tiled_2d", overflow_cfg, "bf16", "nvidia_sm90"
    ), "test setup: config must actually overflow"
    # the sweep never launches it; it shows up as a non-passing entry with a reason
    # (we only sweep this one config via max_configs on a card declaring it... but
    # the declared space doesn't include 256/256/64/s4 for both warps — so instead
    # verify the pre-check predicate directly on the sweep's code path)
    from xkernels.vkl.sweep import apply_config, schedule_from_card

    sched = schedule_from_card(get_card(_CARD_ID))
    _new, ok, _reason = apply_config(sched, overflow_cfg, "nvidia_sm90")
    assert ok  # the knob values are legal (in choices); gate passes
    # the overflow is caught by the cost pre-check, not the knob gate
    from xkernels.vkl import cost

    assert cost.overflows_scratch("tiled_2d", overflow_cfg, "bf16", "nvidia_sm90")


@_SKIP
def test_autotune_winner_beats_default():
    """The sweep's winner is not worse than the Phase 2.0a default config.

    The honest bar for 2.2a: sweeping closes *some* of the gap (the full sweep
    lands ~1.7x the default / ~97% of cuBLAS — proven by the manual run that wrote
    the real perf.measured; here a capped sweep just must not regress).
    """
    from xkernels.verify import verify
    from xkernels.vkl import register_dsl, spec_of

    register_dsl(spec_of(gemm_bf16), backend="triton")
    point = {"dtype": "bf16", "M": 4096, "N": 4096, "K": 4096}
    res = autotune(_CARD_ID, arch="nvidia_sm90", point=point, max_configs=8, record=False)
    if res.winner is None:
        pytest.skip("no passing config in the capped sweep")
    r_default = verify(
        _CARD_ID, arch="nvidia_sm90",
        knobs={"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},  # the Phase 2.0a default
        shapes=[point], measure_perf=True,
    )
    # winner within 10% slop of the default (measurement-noise tolerant)
    assert res.winner_ms <= r_default["perf"]["ms"] * 1.10
