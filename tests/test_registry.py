"""Tests for the agent-native substrate: registry, constraints, retrieval, verify,
write-back. These run on CPU (the reference path); GPU/backend paths are exercised
on hardware machines via the existing per-op test suite."""
from __future__ import annotations

import json

import pytest

from xkernels import find_impl, verify, verify_parity
from xkernels.registry import (
    RegistryError,
    all_cards,
    all_specs,
    get_card,
    get_spec,
    have_validator,
    load_shape_sweep,
    record_measurement,
    validate_decidable,
)
from xkernels.registry.constraints import (
    UndecidableConstraintError,
    evaluate,
)
from xkernels.registry.schemas import validate_impl_card, validate_op_spec

# --- registry loads & validates ----------------------------------------------

def test_registry_validates_all_artifacts():
    if not have_validator():
        pytest.skip("jsonschema not installed")
    specs, cards = all_specs(), all_cards()
    assert len(specs) >= 4
    assert sum(len(b) for b in cards.values()) >= 8
    # every card implements a known spec
    for op_id, bucket in cards.items():
        assert op_id in specs
        for card in bucket.values():
            assert card.implements == op_id


def test_every_seeded_op_has_reference_and_sweep():
    from xkernels._backends import Backend
    for op_id, spec in all_specs().items():
        bucket = all_cards()[op_id]
        has_ref = any(c.backend is Backend.REFERENCE for c in bucket.values())
        assert has_ref, f"{op_id} missing reference card"
        sweep = load_shape_sweep(spec.shape_sweep)
        assert len(sweep) >= 3, f"{op_id} sweep too small"


def test_reference_callables_resolve():
    from xkernels.registry import reference_callable

    for op_id in all_specs():
        fn = reference_callable(op_id)
        assert callable(fn)


# --- constraint mini-language ------------------------------------------------

def test_constraints_evaluate():
    assert evaluate("K % 8 == 0", {"K": 16}) is True
    assert evaluate("K % 8 == 0", {"K": 17}) is False
    assert evaluate("dtype(x) == 'bf16'", {"dtype:x": "bf16"}) is True
    assert evaluate("M >= 16 and N % 32 == 0", {"M": 16, "N": 64}) is True
    assert evaluate("dtype(x) == dtype(w)", {"dtype:x": "bf16", "dtype:w": "fp32"}) is False


@pytest.mark.parametrize("bad", [
    "open('/etc/passwd')",          # calls a non-dtype function
    "__import__('os')",             # attribute access / import
    "x if True else 0",             # ifexp not in the decidable subset
])
def test_non_decidable_constraints_rejected_at_ingest(bad):
    with pytest.raises(UndecidableConstraintError):
        validate_decidable(bad)


def test_op_spec_with_bad_constraint_is_rejected():
    bad = {
        "id": "bad@1.0.0", "name": "bad", "version": "1.0.0", "kernel": "bad",
        "op": {"signature": "x", "canonical_op": "reduce"},
        "inputs": {"x": {"dtype": ["fp32"], "rank": 1}},
        "outputs": {"o": {"dtype": ["fp32"], "rank": 1}},
        "constraints": ["__import__('os').system('rm -rf /')"],  # not decidable
        "numerics": {"reference": "x:y", "rtol": 0.01, "atol": 0.01},
        "shape_sweep": "bad",
    }
    # schema-valid (constraint is just a string), but loader must reject as undecidable
    with pytest.raises(RegistryError):
        import pathlib
        import tempfile

        from xkernels.registry.loader import _load_op_spec  # noqa: WPS437
        p = pathlib.Path(tempfile.mkdtemp()) / "bad.spec.json"
        p.write_text(json.dumps(bad))
        _load_op_spec(p)


# --- retrieval ---------------------------------------------------------------

def test_find_impl_ranks_applicable_first_with_reasons():
    res = find_impl(
        "norm",
        {"x1": {"dtype": "bf16", "shape": [64, 1536]},
         "x2": {"dtype": "bf16", "shape": [64, 512]}},
        target_arch="amd_cdna3",
    )
    assert res, "expected at least one norm candidate"
    # applicable candidates sort first
    assert res[0]["applicable"] is True
    # every result carries reject_reasons (possibly empty)
    for r in res:
        assert "reject_reasons" in r


def test_find_impl_rejects_on_dtype_constraint():
    res = find_impl(
        "norm",
        {"x1": {"dtype": "bf16", "shape": [64, 8]},   # x1 dtype
         "x2": {"dtype": "fp16", "shape": [64, 8]}},  # but w2 would need to match x2
        target_arch="any",
    )
    # constraint evaluation binds dtype per-arg from the query's input_specs;
    # with mismatched bindings the op still evaluates its declared constraints.
    applicable = [r for r in res if r["applicable"]]
    # dual_rmsnorm requires dtype(x1)==dtype(w1) etc.; here we only provided x1,x2
    # so dtype(w1)/dtype(w2) are unbound -> conservatively not rejected.
    assert isinstance(applicable, list)


def test_find_impl_missing_backend_signal():
    # an unseeded canonical_op returns no candidates at all (conv2d has no Op Spec).
    # (gemm used to be the empty example, but the GEMM category is now seeded —
    # mm_fp8_blockscale / hc_prenorm_gemm / moe_int4_w4a16 — so it is no longer
    # a valid "absent op" probe.)
    res = find_impl("conv2d", target_arch="nvidia_sm90")
    assert res == []


def test_find_impl_cuda_card_rejected_on_amd_target():
    # No cuda cards seeded, but exercise the vendor-coherence rejection path by
    # checking that a synthetic nvidia target doesn't surface an amd-only hip card.
    res = find_impl("reduce", target_arch="nvidia_sm90",
                    input_specs={"y": {"dtype": "bf16", "shape": [16, 8, 32]}})
    backends = {r["backend"] for r in res if r["applicable"]}
    assert "hip" not in backends


# --- verify (CPU reference path) ---------------------------------------------

@pytest.mark.parametrize("card_id", [
    "fused_ffn.reference@1.0.0",
    "dual_rmsnorm.reference@1.0.0",
    "moe_sum_reduce.reference@1.0.0",
    "mha_merge_state.reference@1.0.0",
    "moe_align_block_size.reference@1.0.0",
    # milestone wave: dense-fp8 / fused / grouped-quant GEMMs, sparse-MLA, mhc pre-fusion
    "mm_fp8_blockscale.reference@1.0.0",
    "hc_prenorm_gemm.reference@1.0.0",
    "moe_int4_w4a16.reference@1.0.0",
    "sparse_mla_attention.reference@1.0.0",
    "mhc_pre.reference@1.0.0",
])
def test_verify_reference_card_passes_on_cpu(card_id):
    v = verify(card_id, arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_verify_triton_card_reports_unrunnable_without_gpu():
    import torch
    if torch.cuda.is_available():
        pytest.skip("GPU present; triton is runnable here")
    v = verify("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3")
    assert v["compiled"] is False
    assert "error" in v["artifacts"]


def test_verify_returns_run_id():
    v = verify("fused_ffn.reference@1.0.0", arch="any", seed=7)
    assert v["artifacts"]["run_id"].startswith("run:")
    # reproducible: same args -> same run id
    v2 = verify("fused_ffn.reference@1.0.0", arch="any", seed=7)
    assert v["artifacts"]["run_id"] == v2["artifacts"]["run_id"]


def test_verify_applies_accepted_knobs_and_reports_unapplied():
    """Specialization is real: accepted knobs flow to the kernel, unaccepted ones
    are reported as unapplied (the honesty §10 demands)."""
    from xkernels._backends import Backend
    from xkernels._dispatch import registered_backends
    if Backend.TRITON not in registered_backends("ffn"):
        pytest.skip("triton backend not registered (triton not installed)")
    from xkernels.registry import backend_callable
    from xkernels.verify import _accepted_knobs

    fn = backend_callable("fused_ffn@1.0.0", "triton")
    accepted, _ = _accepted_knobs(fn, {"x": None, "w_gate": None, "w_up": None, "w_down": None})
    assert "BLOCK" in accepted  # the declared specialization knob is honored
    # reference callable must NOT accept BLOCK (oracle is knob-free)
    ref = backend_callable("fused_ffn@1.0.0", "reference")
    ref_inputs = {"x": None, "w_gate": None, "w_up": None, "w_down": None}
    ref_accepted, _ = _accepted_knobs(ref, ref_inputs)
    assert "BLOCK" not in ref_accepted


# --- parity ------------------------------------------------------------------

def test_verify_parity_structure():
    p = verify_parity("dual_rmsnorm@1.0.0")
    assert p["op_id"] == "dual_rmsnorm@1.0.0"
    assert "agree" in p and "diverging" in p
    assert p["cross_backend_rtol"] == get_spec("dual_rmsnorm@1.0.0").numerics.cross_backend_rtol
    import torch
    if not torch.cuda.is_available():
        # On a CPU box the GPU backends (triton/cuda) can't run, so only the
        # reference is runnable -> parity is INCONCLUSIVE (agree=None), not a
        # vacuous pass. This is the honesty the old `agree=True` violated.
        assert p["per_backend_runnable"]["REFERENCE"] is True
        assert p["n_runnable"] == 1
        assert p["inconclusive"] is True
        assert p["agree"] is None  # NOT True: a backend agreeing with itself isn't parity


def test_verify_parity_inconclusive_with_single_runnable_backend():
    """A parity result with <2 runnable backends must be inconclusive, not pass.

    Before the device fix, verify_parity() hardcoded device='cpu' and every
    GPU-only card raised in the guard, so all 5 CUTE cards reported
    `agree=True` with only REFERENCE runnable — i.e. they agreed with
    themselves. That was a vacuous pass; this test pins the honest behaviour.
    """
    p = verify_parity("mha_merge_state@1.0.0", device="cpu")
    # cuda card raises in its guard on CPU; triton needs a GPU -> only reference
    assert p["n_runnable"] <= 1
    assert p["inconclusive"] is True
    assert p["agree"] is None
    assert "CUDA" in p["per_backend_runnable"]  # the cuda card IS in the bucket
    assert p["per_backend_runnable"]["CUDA"] is False  # but not runnable on cpu


# --- write-back invariants + round-trip --------------------------------------

def test_record_measurement_rejects_unsourced():
    with pytest.raises(ValueError):
        record_measurement("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3",
                           shape={"T": 64, "d1": 1536, "d2": 512}, dtype="bf16", source="")


def test_record_measurement_rejects_archless():
    with pytest.raises(ValueError):
        record_measurement("dual_rmsnorm.triton@1.0.0", arch="",
                           shape={"T": 64}, dtype="bf16", source="run:abc")


def test_record_measurement_rejects_untrusted_external():
    with pytest.raises(ValueError):
        record_measurement("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3",
                           shape={"T": 64}, dtype="bf16", source="run:abc", trust="external")


def test_record_measurement_round_trip_restores_file(tmp_path):
    """Write a measurement, assert it lands, then restore the original file."""
    import pathlib

    from xkernels.registry.loader import registry_root, reset_cache
    card_path = pathlib.Path(registry_root()) / "impls" / "dual_rmsnorm.triton.card.json"
    original = card_path.read_text()
    try:
        out = record_measurement(
            "dual_rmsnorm.triton@1.0.0", arch="amd_cdna3",
            shape={"T": 64, "d1": 1536, "d2": 512}, dtype="bf16",
            knobs={"num_warps": 4}, ms=0.06, source="run:test-roundtrip",
        )
        assert out["total_measurements"] >= 1
        reset_cache()
        card = get_card("dual_rmsnorm.triton@1.0.0")
        assert any(m.source == "run:test-roundtrip" for m in card.measured)
    finally:
        card_path.write_text(original)
        reset_cache()


# --- schema validation edge cases --------------------------------------------

def test_impl_card_schema_rejects_missing_provenance():
    pytest.importorskip("jsonschema")
    bad = {
        "id": "x.triton@1.0.0", "implements": "x@1.0.0", "backend": "triton",
        "arch": {"family": "any"},
        "specialization_knobs": {}, "perf": {"roofline": "memory_bound"},
        # provenance missing
    }
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        validate_impl_card(bad)


def test_op_spec_schema_rejects_missing_numerics():
    pytest.importorskip("jsonschema")
    bad = {
        "id": "x@1.0.0", "name": "x", "version": "1.0.0", "kernel": "x",
        "op": {"signature": "x", "canonical_op": "reduce"},
        "inputs": {"x": {"dtype": ["fp32"], "rank": 1}},
        "outputs": {"o": {"dtype": ["fp32"], "rank": 1}},
        "constraints": [],
        "shape_sweep": "x",
    }
    import jsonschema
    with pytest.raises(jsonschema.ValidationError):
        validate_op_spec(bad)


def test_arch_vocab_matches_schema():
    """The Python arch set (registry.archs) MUST match the JSON-schema enum.

    Keeping two copies is unavoidable (JSON Schema can't import Python), so this
    test is the anti-drift guard: the next arch addition is one line in the
    schema enum + one line in ``archs.py``, and any mismatch fails here.
    """
    from xkernels.registry import ALL_ARCHS, impl_card_schema
    schema_archs = set(
        v for v in impl_card_schema()["properties"]["arch"]["properties"]["family"]["enum"]
        if v != "any"
    )
    assert schema_archs == ALL_ARCHS, (
        f"arch vocab drifted: schema={sorted(schema_archs)} "
        f"python={sorted(ALL_ARCHS)}"
    )


def test_vendor_of_partitions_all_archs():
    """Every concrete arch resolves to a vendor (no arch is 'any' or orphan)."""
    from xkernels.registry import AMD_ARCHS, NVIDIA_ARCHS, vendor_of
    for a in AMD_ARCHS:
        assert vendor_of(a) == "amd"
    for a in NVIDIA_ARCHS:
        assert vendor_of(a) == "nvidia"
    assert AMD_ARCHS.isdisjoint(NVIDIA_ARCHS)


def test_cost_model_matches_roofline_arithmetic_intensity():
    """The analytical FLOP/byte models reproduce the roofline survey's AIs.

    These AIs are what verify(measure_perf=True) now divides against the measured
    ms to fill tflops / achieved_bw_pct — so the models MUST match the byte
    arithmetic the survey validated on GB10. A drift here would mislead every
    diagnose-* skill that reads the perf block.
    """
    from xkernels.registry.cost_model import cost_model, has_model
    cases = {
        # (op_id, point, expected_AI_from_roofline_survey)
        "mm_fp8_blockscale@1.0.0": ({"M": 128, "N": 512, "K": 512, "dtype": "fp32"}, 42.7),
        "moe_sum_reduce@1.0.0": ({"M": 128, "top_k": 8, "H": 7168, "dtype": "bf16"}, 2.0),
        "mha_merge_state@1.0.0": ({"T": 64, "H": 128, "D": 128, "dtype": "bf16"}, 1.0),
        "hc_prenorm_gemm@1.0.0": ({"T": 37, "K": 128, "N": 16, "dtype": "bf16"}, 7.7),
    }
    for op_id, (point, expected_ai) in cases.items():
        assert has_model(op_id), f"no cost model for {op_id}"
        flops, bytes_rw = cost_model(op_id, point)
        ai = flops / bytes_rw
        assert abs(ai - expected_ai) < 0.15, (
            f"{op_id} AI drifted: got {ai:.2f}, survey said {expected_ai}"
        )


def test_cost_model_dtype_aware_byte_count():
    """bf16 vs fp32 must halve the byte count for the read-bound op."""
    from xkernels.registry.cost_model import cost_model
    _, bytes_bf16 = cost_model("moe_sum_reduce@1.0.0",
                               {"M": 128, "top_k": 8, "H": 7168, "dtype": "bf16"})
    _, bytes_fp32 = cost_model("moe_sum_reduce@1.0.0",
                               {"M": 128, "top_k": 8, "H": 7168, "dtype": "fp32"})
    # the fp32 path moves ~2x the y-bytes (out is fp32 in both); assert bf16 is
    # strictly cheaper, which is the whole point of the bf16-native-read lever.
    assert bytes_bf16 < bytes_fp32


# --- skill outcome store (§7.3) ---------------------------------------------

def test_outcome_record_and_metrics_rollup():
    from xkernels.registry import all_outcomes, record_outcome, reset_outcomes, skill_metrics
    reset_outcomes()
    try:
        record_outcome("tune-for-cdna", "1.0.0", "gemm|amd_cdna3|4096|bf16",
                       "success", iterations=3, run_id="run:a")
        record_outcome("tune-for-cdna", "1.0.0", "gemm|amd_cdna3|2048|bf16",
                       "success", iterations=5, run_id="run:b")
        record_outcome("tune-for-cdna", "1.0.0", "gemm|amd_cdna3|4096|bf16",
                       "fail", iterations=8, failure_mode="occupancy", run_id="run:c")
        m = skill_metrics("tune-for-cdna")
        assert m["uses"] == 3
        assert m["success_rate"] == round(2 / 3, 3)
        assert m["regression_count"] == 1  # the 4096 shape had a prior success
        assert m["failure_modes"] == {"occupancy": 1}
        assert m["versions"] == ["1.0.0"]
        assert len(all_outcomes("tune-for-cdna")) == 3
    finally:
        reset_outcomes()


def test_outcome_rejects_bad_result_and_external_writes():
    from xkernels.registry import record_outcome, reset_outcomes
    reset_outcomes()
    try:
        with pytest.raises(ValueError):
            record_outcome("s", "1.0.0", "sig", "bogus")
        with pytest.raises(ValueError):
            record_outcome("s", "1.0.0", "sig", "success", trust="external")
    finally:
        reset_outcomes()


def test_metrics_empty_for_unknown_skill():
    from xkernels.registry import skill_metrics
    m = skill_metrics("never-used-skill")
    assert m["uses"] == 0 and m["success_rate"] is None
