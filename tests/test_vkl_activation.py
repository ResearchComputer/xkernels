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
)
from xkernels.vkl.examples import gelu_and_mul, silu_and_mul

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _act(kind: str):
    """The independent fp32 nonlinearity the body should be bit-exact with."""
    if kind == "silu":
        return lambda g: g * torch.sigmoid(g)
    return lambda g: 0.5 * g * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))


@pytest.mark.parametrize("name,mod", [("silu", silu_and_mul), ("gelu", gelu_and_mul)])
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


@pytest.mark.parametrize("cid", ["silu_and_mul.reference@1.0.0", "gelu_and_mul.reference@1.0.0"])
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


def test_triton_card_honestly_uncompiled_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present; triton is runnable here")
    assert verify("silu_and_mul.triton@1.0.0", arch="amd_cdna3")["compiled"] is False
