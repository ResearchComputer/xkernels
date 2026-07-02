# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for the gated-activation ops (issue #67) + the ``elementwise`` launch.

``silu_and_mul`` / ``gelu_and_mul`` are the first users of ``Launch.elementwise``
— pure pointwise, no reduction — and of the ``silu``/``gelu`` math-IR primitives
added alongside them. This test pins:

  * emit is schema-valid (spec + reference + triton cards round-trip validators);
  * the body is bit-exact with an independent fp32-nonlinearity-then-cast
    formulation (the body IS the oracle, so this checks the IR encodes the intent);
  * verify(rmsnorm... )  -> verify(reference) passes on CPU; find_impl surfaces
    them as ``activation`` candidates.
"""
from __future__ import annotations

import os

import pytest
import torch

from xkernels import find_impl, verify
from xkernels.registry.models import op_spec_from_doc
from xkernels.registry.schemas import validate_impl_card, validate_op_spec
from xkernels.vkl import (
    emit_card,
    emit_reference_card,
    emit_spec,
    make_inputs,
    run_reference,
    spec_of,
    trace_ir,
)
from xkernels.vkl.examples import (
    gelu_and_mul,
    packed_gelu_and_mul,
    packed_silu_and_mul,
    rmsnorm,
    silu_and_mul,
)
from xkernels.vkl.lower.mathbody import _symbol_values

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _act(kind: str):
    """The independent fp32 nonlinearity the body should be bit-exact with."""
    if kind == "silu":
        return lambda g: g * torch.sigmoid(g)
    return lambda g: 0.5 * g * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))


@pytest.mark.parametrize(
    "name,mod",
    [
        ("silu", silu_and_mul),
        ("gelu", gelu_and_mul),
        ("silu", packed_silu_and_mul),
        ("gelu", packed_gelu_and_mul),
    ],
)
def test_emit_schema_valid(name, mod):
    spec = spec_of(mod)
    op = emit_spec(spec)
    validate_op_spec(op)
    op_spec_from_doc(op)
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))
    assert spec.launch.pattern == "elementwise"


@pytest.mark.parametrize("name,mod", [("silu", silu_and_mul), ("gelu", gelu_and_mul)])
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_body_bit_exact_with_fp32_nonlinearity(name, mod, dt):
    """Body == act(gate.float()) * up.float(), cast to out dtype (bit-exact)."""
    spec = spec_of(mod)
    ins = make_inputs(spec, {"dtype": dt, "M": 5, "K": 7}, device="cpu")
    out = run_reference(spec, ins)[0]
    assert out.dtype == _DTYPES[dt]
    gf, uf = ins["gate"].float(), ins["up"].float()
    naive = (_act(name)(gf) * uf).to(_DTYPES[dt])
    assert torch.equal(out, naive), f"{name}_and_mul {dt}: body drifted from fp32 nonlinearity"


@pytest.mark.parametrize(
    "name,mod",
    [("silu", packed_silu_and_mul), ("gelu", packed_gelu_and_mul)],
)
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_packed_body_bit_exact_with_fp32_nonlinearity(name, mod, dt):
    """Packed body == act(x[:, :K].float()) * x[:, K:].float(), cast to output dtype."""
    spec = spec_of(mod)
    ins = make_inputs(spec, {"dtype": dt, "M": 5, "K": 7, "twoK": 14}, device="cpu")
    out = run_reference(spec, ins)[0]
    assert out.dtype == _DTYPES[dt]
    gate, up = ins["x"][:, :7].float(), ins["x"][:, 7:].float()
    naive = (_act(name)(gate) * up).to(_DTYPES[dt])
    assert torch.equal(out, naive), f"packed_{name}_and_mul {dt}: body drifted"


def test_packed_symbol_values_infer_output_half_width():
    """Generated Triton launch sizing can bind output-only K from x[M, twoK]."""
    spec = spec_of(packed_silu_and_mul)
    ins = make_inputs(spec, {"dtype": "bf16", "M": 5, "K": 7, "twoK": 14}, device="cpu")
    vals = _symbol_values(trace_ir(spec), ins)
    assert vals["M"] == 5
    assert vals["twoK"] == 14
    assert vals["K"] == 7


def test_rowwise_symbol_values_keep_reduce_dim_for_broadcast():
    """Shape inference must model rowwise reductions as keepdim=True intermediates."""
    spec = spec_of(rmsnorm)
    ins = make_inputs(spec, {"dtype": "bf16", "T": 5, "d": 7}, device="cpu")
    vals = _symbol_values(trace_ir(spec), ins)
    assert vals["T"] == 5
    assert vals["d"] == 7


@pytest.mark.parametrize(
    "cid",
    [
        "silu_and_mul.reference@1.0.0",
        "gelu_and_mul.reference@1.0.0",
        "packed_silu_and_mul.reference@1.0.0",
        "packed_gelu_and_mul.reference@1.0.0",
    ],
)
def test_verify_reference_card_passes_on_cpu(cid):
    v = verify(cid, arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_find_impl_surfaces_activations():
    res = find_impl(
        "activation",
        {"gate": {"dtype": "bf16", "shape": [64, 128]},
         "up": {"dtype": "bf16", "shape": [64, 128]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "silu_and_mul.reference@1.0.0" in ids
    assert "gelu_and_mul.triton@1.0.0" in ids
    assert any(r["applicable"] for r in res if "silu_and_mul" in r["impl_card_id"])


def test_find_impl_surfaces_packed_activations():
    res = find_impl(
        "activation",
        {"x": {"dtype": "bf16", "shape": [64, 256]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "packed_silu_and_mul.reference@1.0.0" in ids
    assert "packed_gelu_and_mul.triton@1.0.0" in ids
    assert any(
        r["applicable"] for r in res if r["impl_card_id"] == "packed_silu_and_mul.triton@1.0.0"
    )


def test_find_impl_rejects_odd_packed_width():
    res = find_impl(
        "activation",
        {"x": {"dtype": "bf16", "shape": [64, 255]}},
        target_arch="amd_cdna3",
    )
    packed = [r for r in res if r["impl_card_id"] == "packed_silu_and_mul.triton@1.0.0"]
    assert packed and packed[0]["applicable"] is False
    assert "constraint violated: 'twoK % 2 == 0'" in packed[0]["reject_reasons"]


def test_triton_card_honestly_uncompiled_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present; triton is runnable here")
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        pytest.skip("TRITON_INTERPRET=1 runs triton on CPU; the card is runnable here")
    assert verify("silu_and_mul.triton@1.0.0", arch="amd_cdna3")["compiled"] is False
