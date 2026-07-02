# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for ``rmsnorm`` (issue #66) — a DSL-ONLY op (no hand counterpart).

Unlike ``test_vkl_emit_dual_rmsnorm`` (which cross-checks the DSL body against a
hand-written spec/reference), rmsnorm is authored SOLELY in the DSL: the
``@kernel`` body IS the auto-reference and the contract. This test pins the
three properties that make that safe:

  * **emit is schema-valid** — the emitted spec + cards round-trip the real
    validators (validate_op_spec / validate_impl_card) and dataclass ingest.
  * **the body is bit-exact with the independent math** — run_reference matches a
    naive fp32 RMSNorm (var = mean(x^2) in fp32; out = x*rsqrt(var+eps)*w) over
    identical seeded inputs, for fp32/bf16/fp16. The body is the oracle, so this
    is the load-bearing check that the math IR encodes the intended numerics.
  * **the substrate gate passes** — verify(rmsnorm.reference) is compiled/passed/
    deterministic on CPU (the @kernel decorator auto-wires REFERENCE dispatch +
    the input gen), and find_impl surfaces it as a ``norm`` candidate.
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
from xkernels.vkl.examples import rmsnorm

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


@pytest.fixture(scope="module")
def spec():
    return spec_of(rmsnorm)


def test_emit_is_schema_valid(spec):
    op = emit_spec(spec)
    validate_op_spec(op)
    op_spec_from_doc(op)
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))


@pytest.mark.parametrize("dt", list(_DTYPES))
def test_body_is_bit_exact_with_naive_fp32_rmsnorm(spec, dt):
    """The auto-reference body == an independent fp32 RMSNorm, over identical bits."""
    tdt = _DTYPES[dt]
    ins = make_inputs(spec, {"dtype": dt, "T": 64, "d": 128}, device="cpu")
    out = run_reference(spec, ins)[0]
    assert out.shape == (64, 128) and out.dtype == tdt
    # independent reference over the SAME seeded x, w
    xf = ins["x"].float()
    var = (xf * xf).sum(1, keepdim=True) / xf.shape[1]
    naive = (xf * torch.rsqrt(var + 1e-6)).to(tdt) * ins["w"].to(tdt)
    assert torch.equal(out.float(), naive.float())  # bit-exact


def test_verify_reference_card_passes_on_cpu():
    v = verify("rmsnorm.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_find_impl_surfaces_rmsnorm_as_norm_candidate():
    res = find_impl(
        "norm",
        {"x": {"dtype": "bf16", "shape": [64, 4096]},
         "w": {"dtype": "bf16", "shape": [4096]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "rmsnorm.reference@1.0.0" in ids
    assert "rmsnorm.triton@1.0.0" in ids
    assert any(r["applicable"] for r in res if "rmsnorm" in r["impl_card_id"])


def test_triton_card_honestly_uncompiled_without_gpu():
    """No GPU -> the triton card is not yet wired/compiled (register_dsl is the GPU step)."""
    if torch.cuda.is_available():
        pytest.skip("GPU present; triton is runnable here")
    v = verify("rmsnorm.triton@1.0.0", arch="amd_cdna3")
    assert v["compiled"] is False
