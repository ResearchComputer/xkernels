# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for temperature softmax, the deterministic prefix of #69/#70.

Top-k/top-p and RNG sampling are deliberately not added to the math IR. This
test pins the expressible piece those issue families share: stable row-wise
softmax with a rank-1 per-row temperature input broadcast across the vocab axis.
"""
from __future__ import annotations

import pytest
import torch

from xkernels.registry.models import op_spec_from_doc
from xkernels.registry.schemas import validate_impl_card, validate_op_spec
from xkernels.vkl import (
    emit_card,
    emit_reference_card,
    emit_spec,
    lower_to_triton,
    run_reference,
    spec_of,
    trace_ir,
)
from xkernels.vkl.examples import temperature_softmax
from xkernels.vkl.lower.mathbody import _TritonGenRowwise

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_GPU_OK = torch.cuda.is_available()
_SKIP_GPU = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")
_DEV = "cuda" if _GPU_OK else "cpu"


@pytest.fixture(scope="module")
def spec():
    return spec_of(temperature_softmax)


def test_emit_schema_valid(spec):
    op = emit_spec(spec)
    validate_op_spec(op)
    op_spec_from_doc(op)
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))
    assert spec.launch.pattern == "rowwise"


@pytest.mark.parametrize("dt", list(_DTYPES))
def test_body_matches_stable_temperature_softmax(spec, dt):
    g = torch.Generator(device="cpu").manual_seed(123)
    logits = (torch.randn(5, 37, generator=g) * 3.0).to(_DTYPES[dt])
    temperatures = torch.rand(5, generator=g, dtype=torch.float32) + 0.25

    probs = run_reference(spec, {"logits": logits, "temperatures": temperatures})[0]

    scaled = logits.float() / temperatures.unsqueeze(1)
    shifted = scaled - scaled.amax(dim=1, keepdim=True)
    expected = torch.exp(shifted)
    expected = expected / expected.sum(dim=1, keepdim=True)

    assert probs.dtype == torch.float32
    assert torch.equal(probs, expected)
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(5), rtol=0.0, atol=1e-6)


def test_rowwise_codegen_broadcasts_per_row_temperature(spec):
    body = trace_ir(spec)
    src = _TritonGenRowwise(body, "fp32").kernel_source()
    assert "tl.load(temperatures_ptr + row)" in src
    assert "tl.max(" in src
    assert "tl.sum(" in src


@_SKIP_GPU
@pytest.mark.parametrize("dt", ["fp32", "bf16"])
def test_triton_softmax_matches_run_reference_on_gpu(spec, dt):
    pytest.importorskip("triton")
    launch = lower_to_triton(spec)
    g = torch.Generator(device="cpu").manual_seed(456)
    logits = (torch.randn(7, 63, generator=g) * 2.0).to(_DTYPES[dt]).to(_DEV)
    temperatures = (torch.rand(7, generator=g, dtype=torch.float32) + 0.25).to(_DEV)

    (out,) = launch(logits=logits, temperatures=temperatures)
    (ref,) = run_reference(spec, {"logits": logits, "temperatures": temperatures})
    tol = spec.numerics.by_dtype[dt]
    torch.testing.assert_close(
        out,
        ref,
        rtol=float(tol["rtol"]),
        atol=float(tol["atol"]),
    )
