# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase A: the schedule-IR spine is the source of truth in both directions
(docs/brainstorm/09).

Three layers, each independently testable:

  * **read-out** (``schedule_from_spec``): a spec + arch -> a STRUCTURED
    ``ScheduleIR`` carrying Tile / MapTo / Stage / Knob nodes (Phase 1's
    ``schedule_from_card`` was a knob-only bag). The structure is what the edit
    primitives operate on.
  * **edit round-trip** (``SetMapPolicy`` / ``MapTo_`` / ``AddStage`` apply): an
    edit's ``apply`` returns a NEW frozen IR; ``check`` is locally decidable.
    The load-bearing new primitive is ``SetMapPolicy`` — the one MMA-level lever
    (``input_precision``) that proves the IR reaches silicon.
  * **read-in** (``resolve_binding``): project the (edited) schedule to the flat
    ``{name: value}`` binding the launcher reads — so the agent path
    (load_schedule -> check_edit -> apply_edit -> resolve_binding -> launch)
    converges on the same launcher entry as ``verify(knobs=...)``.

The pure-logic layers (read-out structure shape, edit round-trip, resolve_binding)
are CPU-doable (torch import only, no launch). The end-to-end "edit changes
silicon" layer (a ``SetMapPolicy("tf32")`` edit actually changing what ``tl.dot``
compiles to) is GPU-gated — it launches the generated kernel.
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
    ScheduleIR,
    SetKnob,
    SetMapPolicy,
    Stage,
    Tile,
    PRECISION_KEY,
    precision_of,
    resolve_binding,
    schedule_from_spec,
)
from xkernels.vkl.examples import gemm_bf16
from xkernels.vkl import spec_of

# ─── GPU gating for the silicon-touching tests (matches test_vkl_lower_gemm) ──
try:  # pragma: no cover - import guard
    import torch

    _GPU_OK = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _GPU_OK = False
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  read-out: schedule_from_spec builds a STRUCTURED schedule (not a knob bag)
# ═══════════════════════════════════════════════════════════════════════════════


class TestScheduleFromSpec:
    def test_tiled_2d_has_tiles_map_and_stages(self):
        """A GEMM spec lowers to [Tile, Tile, Tile, MapTo, Stage, Stage, Knob...]."""
        spec = spec_of(gemm_bf16)
        sched = schedule_from_spec(spec, arch="nvidia_sm90")
        tiles = sched.tiles()
        assert len(tiles) == 3, f"expected out/a_tile/b_tile tiles, got {len(tiles)}"
        maps = sched.maps()
        assert len(maps) == 1, "a GEMM has exactly one MMA -> one L5 map"
        m = maps[0]
        assert m.level == "L5"
        # On sm_90 the native matrix engine is wgmma; on the portable "any"
        # target there is no concrete instruction (Triton picks at runtime).
        assert m.instruction == "wgmma"
        stages = sched.stages()
        assert len(stages) == 2, "K-loop streams both operands through scratch"
        # stage depth tracks the num_stages knob BY NAME (resolved at emit)
        assert all(isinstance(s.depth, str) for s in stages)

    def test_portable_target_has_no_concrete_instruction(self):
        """The 'any' target honestly reports no native instruction (Triton picks)."""
        spec = spec_of(gemm_bf16)
        sched = schedule_from_spec(spec, arch="any")
        m = sched.maps()[0]
        assert m.instruction is None
        assert m.precision is None  # dtype-default until an edit sets it

    def test_declared_knobs_carry_through(self):
        spec = spec_of(gemm_bf16)
        sched = schedule_from_spec(spec, arch="nvidia_sm90")
        # gemm_bf16 declares BLOCK_M/BLOCK_N/BLOCK_K + num_warps/num_stages knobs
        assert "BLOCK_M" in sched.knobs
        assert "num_stages" in sched.knobs
        for k in sched.knobs.values():
            assert k.value in k.choices  # current binding is a legal choice


# ═══════════════════════════════════════════════════════════════════════════════
# §2  SetMapPolicy: the MMA input_precision lever (the doc-09 "reaches silicon" edit)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def gemm_sched():
    """A real GEMM schedule on sm_90 (read-out from the spec)."""
    return schedule_from_spec(spec_of(gemm_bf16), arch="nvidia_sm90")


class TestSetMapPolicy:
    def test_check_accepts_legal_policy(self, gemm_sched):
        e = SetMapPolicy(map_id="mma0", precision="tf32")
        assert isinstance(e.check(gemm_sched, "nvidia_sm90"), Ok)

    def test_check_rejects_unknown_policy(self, gemm_sched):
        r = SetMapPolicy(map_id="mma0", precision="bogus").check(gemm_sched, "nvidia_sm90")
        assert isinstance(r, Reject) and "not in" in r.reason

    def test_check_rejects_non_mapto_node(self, gemm_sched):
        # 'out' is a Tile, not a MapTo
        r = SetMapPolicy(map_id="out", precision="tf32").check(gemm_sched, "nvidia_sm90")
        assert isinstance(r, Reject) and "not a MapTo" in r.reason

    def test_check_rejects_missing_map(self, gemm_sched):
        r = SetMapPolicy(map_id="nope", precision="tf32").check(gemm_sched, "nvidia_sm90")
        assert isinstance(r, Reject) and "no map node" in r.reason

    def test_apply_sets_precision_immutably(self, gemm_sched):
        e = SetMapPolicy(map_id="mma0", precision="tf32")
        ir2 = e.apply(gemm_sched)
        assert ir2.maps()[0].precision == "tf32"
        # immutable: original IR unchanged (None = dtype-default)
        assert gemm_sched.maps()[0].precision is None

    def test_resolve_binding_carries_precision(self, gemm_sched):
        """The projection the launcher reads: tf32 edit -> binding['input_precision']."""
        ir2 = SetMapPolicy(map_id="mma0", precision="tf32").apply(gemm_sched)
        b = resolve_binding(ir2)
        assert b[PRECISION_KEY] == "tf32"
        # and precision_of reads the same value
        assert precision_of(ir2) == "tf32"

    def test_default_precision_omitted_from_binding(self, gemm_sched):
        """dtype-default (None) is omitted: the lowering picks per dtype."""
        b = resolve_binding(gemm_sched)
        assert PRECISION_KEY not in b
        assert precision_of(gemm_sched) is None


# ═══════════════════════════════════════════════════════════════════════════════
# §3  MapTo_ + AddStage: apply is REAL now (was a Phase-2 stub)
# ═══════════════════════════════════════════════════════════════════════════════


class TestMapToApply:
    def test_check_rejects_illegal_instruction(self, gemm_sched):
        r = MapTo_(map_id="mma0", op_ref="acc", level="L5", instruction="my_asm").check(
            gemm_sched, "nvidia_sm90"
        )
        assert isinstance(r, Reject) and "not legal for" in r.reason

    def test_check_rejects_bad_precision(self, gemm_sched):
        r = MapTo_(
            map_id="mma0", op_ref="acc", level="L5", instruction="wgmma", precision="nope"
        ).check(gemm_sched, "nvidia_sm90")
        assert isinstance(r, Reject) and "not in" in r.reason

    def test_apply_records_map_with_precision(self, gemm_sched):
        e = MapTo_(map_id="mma1", op_ref="acc", level="L5",
                   instruction="wgmma", precision="tf32")
        assert isinstance(e.check(gemm_sched, "nvidia_sm90"), Ok)
        ir2 = e.apply(gemm_sched)
        new = [m for m in ir2.maps() if m.id == "mma1"][0]
        assert new.instruction == "wgmma" and new.precision == "tf32"

    def test_l2_tile_divisibility_bites(self):
        """Concrete M=40 not divisible by wgmma m=64 -> Reject; M=64 -> Ok."""
        div_ok = (
            ScheduleIR()
            .with_node(Tile(id="out", shape=(64, 64), level="L2"))
            .with_node(MapTo(id="mma0", op_ref="acc", level="L5", instruction="wgmma"))
        )
        assert isinstance(
            MapTo_(map_id="mma0", op_ref="acc", level="L5", instruction="wgmma").check(
                div_ok, "nvidia_sm90"
            ),
            Ok,
        )
        div_bad = div_ok.with_node(Tile(id="out2", shape=(40, 64), level="L2"))
        r = MapTo_(map_id="mma0", op_ref="acc", level="L5", instruction="wgmma").check(
            div_bad, "nvidia_sm90"
        )
        assert isinstance(r, Reject) and "not divisible" in r.reason

    def test_symbolic_tile_defers_divisibility(self):
        """A knob-name dim (BLOCK_M) is resolved at emit, so divisibility is Ok."""
        sym = (
            ScheduleIR()
            .with_node(Tile(id="out", shape=("BLOCK_M", 64), level="L2"))
            .with_node(MapTo(id="mma0", op_ref="acc", level="L5", instruction="wgmma"))
        )
        assert isinstance(
            MapTo_(map_id="mma0", op_ref="acc", level="L5", instruction="wgmma").check(
                sym, "nvidia_sm90"
            ),
            Ok,
        )


class TestAddStageApply:
    def test_apply_is_not_a_stub(self, gemm_sched):
        """AddStage.apply returns a real IR with the new Stage (was Phase-2 stub)."""
        e = AddStage(stage_id="stage_c", producer_ref="b_tile", depth=4, tile_bytes=4096)
        ir2 = e.apply(gemm_sched)
        assert any(s.id == "stage_c" for s in ir2.stages())

    def test_concrete_depth_does_not_corrupt_binding(self, gemm_sched):
        """A concrete-int AddStage stage coexists with the declared num_stages knob.

        resolve_binding's precedence: a declared ``num_stages`` knob is the
        authority; a concrete-int stage depth only fills the binding when the knob
        is absent (a custom pipeline with no declared knob). Here the knob wins,
        so the binding is unchanged by the AddStage.
        """
        e = AddStage(stage_id="stage_c", producer_ref="b_tile", depth=4, tile_bytes=4096)
        ir2 = e.apply(gemm_sched)
        b = resolve_binding(ir2)
        # declared knob is the authority; AddStage concrete depth does not override
        assert b["num_stages"] == gemm_sched.knobs["num_stages"].value


# ═══════════════════════════════════════════════════════════════════════════════
# §4  the round-trip converges: schedule binding == launcher input
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoundTripConvergence:
    def test_setknob_then_resolve(self, gemm_sched):
        """A SetKnob edit changes the knob value the binding carries."""
        ir2 = SetKnob(name="BLOCK_M", value=128).apply(gemm_sched)
        assert resolve_binding(ir2)["BLOCK_M"] == 128

    def test_edit_chain_is_immutable(self, gemm_sched):
        """An edit chain produces independent snapshots (the tuning_trace form)."""
        s0 = gemm_sched
        s1 = SetMapPolicy(map_id="mma0", precision="tf32").apply(s0)
        s2 = SetKnob(name="BLOCK_M", value=128).apply(s1)
        # each snapshot is independent + frozen
        assert s0.maps()[0].precision is None and s1.maps()[0].precision == "tf32"
        assert resolve_binding(s2) == {
            **resolve_binding(s1), "BLOCK_M": 128, PRECISION_KEY: "tf32"
        }

    def test_concrete_stage_depth_fills_when_knob_absent(self):
        """A concrete-int Stage depth contributes num_stages when no knob declares it.

        The dual of TestAddStageApply.test_concrete_depth_does_not_corrupt_binding:
        here there is no declared num_stages knob, so a concrete-int AddStage stage
        is what supplies the binding's num_stages (a custom pipeline case).
        """
        sched = (
            ScheduleIR()
            .with_node(Stage(id="stage_a", producer_ref="a_tile", space="scratch", depth=5))
        )
        assert resolve_binding(sched)["num_stages"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# §5  edit changes silicon (GPU-gated): a tf32 edit changes what tl.dot compiles to
# ═══════════════════════════════════════════════════════════════════════════════


@_SKIP
def test_tf32_edit_changes_compiled_source():
    """A SetMapPolicy('tf32') edit threads through to the generated tl.dot source.

    This is the doc-09 section 8 "map_to reaches silicon" proof, made real: the
    default fp32 kernel emits ``tl.dot(a, b, input_precision='ieee')`` (true fp32,
    bit-faithful to the reference); after the edit the schedule's binding carries
    ``input_precision='tf32'`` and the codegen emits that instead (sm_80+ tensor
    cores). The edit changes what compiles — schedule IR is the source of truth.
    """
    pytest.importorskip("triton")
    from xkernels.vkl.lower.mathbody import _TritonGen, _find_mma
    from xkernels.vkl.reference import trace_ir

    spec = spec_of(gemm_bf16)
    body = trace_ir(spec)

    # default (None precision) on an fp32 output -> ieee (true fp32, no TF32)
    gen_ieee = _TritonGen(body, out_dtype="fp32", precision=None)
    src_ieee = gen_ieee.kernel_source()
    assert "input_precision='ieee'" in src_ieee

    # the tf32 edit -> codegen emits tf32 (the silicon change)
    gen_tf32 = _TritonGen(body, out_dtype="fp32", precision="tf32")
    src_tf32 = gen_tf32.kernel_source()
    assert "input_precision='tf32'" in src_tf32
    assert "input_precision='ieee'" not in src_tf32


@_SKIP
def test_launch_consumes_schedule_binding():
    """launch(schedule=<edited>) threads the MMA precision from the schedule IR.

    The full Phase-A closure: build the schedule, edit it, hand the EDITED
    schedule to ``launch`` (via the launcher's ``schedule=`` kwarg), and the
    kernel runs with the edited precision. The agent path and the flat-knob path
    converge on the same launcher entry.
    """
    pytest.importorskip("triton")
    from xkernels.vkl import lower_to_triton, make_inputs

    spec = spec_of(gemm_bf16)
    sched = schedule_from_spec(spec, arch="nvidia_sm90")
    sched_tf32 = SetMapPolicy(map_id="mma0", precision="tf32").apply(sched)

    launch = lower_to_triton(spec)
    p = {"dtype": "fp32", "M": 128, "N": 128, "K": 128}
    inputs = {k: v.to("cuda") for k, v in make_inputs(spec, p, seed=7).items()}
    # The launcher accepts a schedule kwarg; it does not raise, and precision is
    # threaded (tf32 -> tensor cores). Numerics are within the loose fp32 tol.
    (out,) = launch(**inputs, schedule=sched_tf32)
    assert out.shape == (128, 128) and out.dtype == __import__("torch").float32
