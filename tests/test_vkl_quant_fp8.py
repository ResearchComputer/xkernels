# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for per-token-group fp8 quantization (issue #57) + reduce_max/mixed-dtype.

The headline checks: the body is **bit-exact with the existing hand helper**
(``ops/gemm/reference.py:per_token_group_quant_fp8``) over identical grouped
operands — the strongest proof that the DSL ``reduce_max`` + ``abs`` + clamp +
fp8-cast primitives compose into the real quantization math. Also pins the
mixed-dtype output (``q`` fp8 + ``scale`` fp32), the per-output dtype resolution,
and the CPU verify gate.
"""
from __future__ import annotations

import pytest
import torch

from xkernels import find_impl, verify
from xkernels.ops.gemm.reference import (
    per_block_quant_fp8 as hand_block_quant,
)
from xkernels.ops.gemm.reference import (
    per_token_group_quant_fp8 as hand_quant,
)
from xkernels.registry.models import op_spec_from_doc
from xkernels.registry.schemas import validate_impl_card, validate_op_spec
from xkernels.vkl import (
    emit_card,
    emit_reference_card,
    emit_spec,
    make_inputs,
    run_reference,
    spec_of,
)
from xkernels.vkl.examples import per_block_quant_fp8, per_token_group_quant_fp8

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def test_emit_schema_valid_mixed_dtype_outputs():
    spec = spec_of(per_token_group_quant_fp8)
    op = emit_spec(spec)
    validate_op_spec(op)
    op_spec_from_doc(op)
    validate_impl_card(emit_reference_card(spec))
    tri = emit_card(spec, spec.targets["triton"])
    validate_impl_card(tri)
    # the mixed-dtype contract is encoded: q fp8, scale fp32
    assert spec.outputs["q"].dtype == ("fp8",)
    assert spec.outputs["scale"].dtype == ("fp32",)
    assert spec.launch.pattern == "rowwise"


def test_emit_schema_valid_per_block_quant():
    spec = spec_of(per_block_quant_fp8)
    op = emit_spec(spec)
    validate_op_spec(op)
    op_spec_from_doc(op)
    validate_impl_card(emit_reference_card(spec))
    tri = emit_card(spec, spec.targets["triton"])
    validate_impl_card(tri)
    assert op["op"]["canonical_op"] == "quantize"
    assert spec.outputs["q"].dtype == ("fp8",)
    assert spec.outputs["scale"].dtype == ("fp32",)


@pytest.mark.parametrize("dt", list(_DTYPES))
def test_body_bit_exact_with_hand_helper(dt):
    """DSL body == the existing hand per_token_group_quant_fp8, over identical grouped x."""
    spec = spec_of(per_token_group_quant_fp8)
    M, K, block = 4, 256, 128
    G = M * (K // block)
    ins = make_inputs(spec, {"dtype": dt, "G": G, "B": block}, device="cpu")
    q_dsl, scale_dsl = run_reference(spec, ins)
    assert q_dsl.dtype == torch.float8_e4m3fn
    assert scale_dsl.dtype == torch.float32
    # rebuild the SAME x as [M, K] (grouped row-major) for the hand helper
    x_mk = ins["x"].reshape(M, K // block, block).reshape(M, K)
    q_hand_mk, scale_hand = hand_quant(x_mk, block=block, fp8_dtype=torch.float8_e4m3fn)
    q_hand = q_hand_mk.reshape(M, K // block, block).reshape(G, block)
    assert torch.equal(q_dsl, q_hand), f"{dt}: fp8 q drifted from hand helper"
    assert torch.equal(scale_dsl, scale_hand.reshape(G)), f"{dt}: scale drifted from hand helper"


@pytest.mark.parametrize("dt", list(_DTYPES))
def test_per_block_body_bit_exact_with_hand_helper(dt):
    """DSL full-tile block quant == hand per_block_quant_fp8 over identical blocks."""
    spec = spec_of(per_block_quant_fp8)
    block = 16
    nt, kt = 2, 3
    N, K = nt * block, kt * block
    G, B = nt * kt, block * block
    ins = make_inputs(spec, {"dtype": dt, "G": G, "B": B}, device="cpu")
    q_dsl, scale_dsl = run_reference(spec, ins)
    assert q_dsl.dtype == torch.float8_e4m3fn
    assert scale_dsl.dtype == torch.float32

    w = ins["x"].reshape(nt, kt, block, block).permute(0, 2, 1, 3).reshape(N, K)
    q_hand, scale_hand = hand_block_quant(w, block=block, fp8_dtype=torch.float8_e4m3fn)
    q_hand_grouped = q_hand.reshape(nt, block, kt, block).permute(0, 2, 1, 3).reshape(G, B)
    assert torch.equal(q_dsl, q_hand_grouped), f"{dt}: fp8 q drifted from hand helper"
    assert torch.equal(scale_dsl, scale_hand.reshape(G)), f"{dt}: scale drifted from hand helper"


def test_verify_reference_card_passes_on_cpu():
    v = verify("per_token_group_quant_fp8.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True
    assert v["correctness"]["max_abs_err"] == 0.0  # reference-vs-itself floor


def test_per_block_verify_reference_card_passes_on_cpu():
    v = verify("per_block_quant_fp8.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_find_impl_surfaces_quant_under_reduce():
    res = find_impl("reduce", {"x": {"dtype": "bf16", "shape": [8, 128]}}, target_arch="amd_cdna3")
    ids = {r["impl_card_id"] for r in res}
    assert "per_token_group_quant_fp8.triton@1.0.0" in ids
    assert any(r["applicable"] for r in res if "per_token_group" in r["impl_card_id"])


def test_find_impl_surfaces_per_block_quantize():
    res = find_impl(
        "quantize",
        {"x": {"dtype": "bf16", "shape": [4, 256]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "per_block_quant_fp8.triton@1.0.0" in ids
    assert any(r["applicable"] for r in res if "per_block_quant_fp8" in r["impl_card_id"])
