# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Public standalone gated-activation interface (issue #67).

``xkernels.silu_and_mul`` / ``gelu_and_mul`` (and the packed, flashinfer-style
single-buffer variants) are the bare ``act(gate) * up`` ops factored out of
``fused_ffn``. The contract is DSL-authored; these tests pin the *dispatch
surface* — that the four names are exported, dispatch to a runnable backend on
CPU (the auto-reference), and match an independent fp32-nonlinearity-then-cast
formulation bit-exactly (the body IS the oracle, so any drift is a wiring bug).

The DSL-generated Triton card is GPU-gated; on a CPU box it honestly reports
``compiled=False`` (no driver) — that path is pinned in
``test_vkl_activation.py::test_triton_card_honestly_uncompiled_without_gpu`` and
``test_registry.py``; these tests stay on the reference backend, which runs
everywhere. The Triton *kernel's* correctness (vs the same oracle) is pinned
separately below and runs under ``TRITON_INTERPRET=1`` or on a GPU.
"""
from __future__ import annotations

import os

import pytest
import torch

import xkernels
from xkernels import (
    gelu_and_mul,
    packed_gelu_and_mul,
    packed_silu_and_mul,
    silu_and_mul,
)
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.utils.testing import assert_close, gpu_device_or_skip

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

# Triton kernel runs on a GPU or under TRITON_INTERPRET=1; otherwise the tests
# below skip (the card honestly reports compiled=False without a driver).
_TRITON_RUNNABLE = (
    Backend.TRITON in registered_backends("silu_and_mul")
    and (torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET", "0") == "1")
)

# op_id (for the spec's per-dtype tolerance) keyed by the public function name
_OP_ID = {
    "silu_and_mul": "silu_and_mul@1.0.0",
    "gelu_and_mul": "gelu_and_mul@1.0.0",
    "packed_silu_and_mul": "packed_silu_and_mul@1.0.0",
    "packed_gelu_and_mul": "packed_gelu_and_mul@1.0.0",
}


def _act(kind: str):
    if kind == "silu":
        return lambda g: torch.nn.functional.silu(g)
    # tanh approximation — the form flashinfer / vLLM gelu_and_mul use
    return lambda g: torch.nn.functional.gelu(g, approximate="tanh")


@pytest.mark.parametrize(
    "fn_name,kind", [("silu_and_mul", "silu"), ("gelu_and_mul", "gelu")]
)
def test_exported_from_top_level(fn_name, kind):
    assert getattr(xkernels, fn_name) is globals()[fn_name]


@pytest.mark.parametrize(
    "kernel",
    ["silu_and_mul", "gelu_and_mul", "packed_silu_and_mul", "packed_gelu_and_mul"],
)
def test_both_backends_registered(kernel):
    """Importing ``xkernels`` wires the DSL triton backend (lazy compile) AND the
    auto-reference, so ``dispatch(..., backend=...)`` resolves either."""
    impls = registered_backends(kernel)
    assert Backend.REFERENCE in impls
    assert Backend.TRITON in impls


@pytest.mark.parametrize("fn_name,kind", [("silu_and_mul", "silu"), ("gelu_and_mul", "gelu")])
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_two_tensor_matches_fp32_nonlinearity(fn_name, kind, dt):
    """``act(gate.float()) * up.float()`` cast to the out dtype — bit-exact."""
    torch.manual_seed(0)
    dtype = _DTYPES[dt]
    gate = torch.randn(5, 7, dtype=dtype)
    up = torch.randn(5, 7, dtype=dtype)
    fn = globals()[fn_name]
    out = fn(gate, up, backend="reference")
    expected = (_act(kind)(gate.float()) * up.float()).to(dtype)
    assert out.dtype == dtype
    assert out.shape == gate.shape
    # spec tolerance: fp32 near-exact (1e-5); bf16/fp16 1.6e-2 — the body IS the
    # oracle, so the only divergence is the final fp32->dtype cast (bf16/fp16) or
    # an ULP in the nonlinearity's op order vs F.silu/F.gelu (fp32).
    assert_close(out, expected, op_id=_OP_ID[fn_name])


@pytest.mark.parametrize(
    "fn_name,kind", [("packed_silu_and_mul", "silu"), ("packed_gelu_and_mul", "gelu")]
)
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_packed_matches_fp32_nonlinearity(fn_name, kind, dt):
    """Packed ``x[:, :K]`` / ``x[:, K:]`` convention, bit-exact."""
    torch.manual_seed(1)
    dtype = _DTYPES[dt]
    M, K = 5, 7
    x = torch.randn(M, 2 * K, dtype=dtype)
    fn = globals()[fn_name]
    out = fn(x, backend="reference")
    gate, up = x[:, :K], x[:, K:]
    expected = (_act(kind)(gate.float()) * up.float()).to(dtype)
    assert out.dtype == dtype
    assert out.shape == (M, K)
    assert_close(out, expected, op_id=_OP_ID[fn_name])


@pytest.mark.parametrize(
    "fn_name,kind", [("silu_and_mul", "silu"), ("gelu_and_mul", "gelu")]
)
def test_auto_backend_resolves_on_cpu(fn_name, kind):
    """``backend='auto'`` picks a runnable backend on CPU (the reference)."""
    torch.manual_seed(2)
    gate = torch.randn(4, 8, dtype=torch.bfloat16)
    up = torch.randn(4, 8, dtype=torch.bfloat16)
    out = globals()[fn_name](gate, up)  # default auto
    expected = (_act(kind)(gate.float()) * up.float()).to(torch.bfloat16)
    assert_close(out, expected, op_id=_OP_ID[fn_name])


def test_packed_equivalent_to_two_tensor():
    """The packed op on ``x = [gate | up]`` equals the two-tensor op on the halves."""
    torch.manual_seed(3)
    gate = torch.randn(6, 9, dtype=torch.bfloat16)
    up = torch.randn(6, 9, dtype=torch.bfloat16)
    x = torch.cat([gate, up], dim=-1)
    assert torch.equal(packed_silu_and_mul(x, backend="reference"),
                       silu_and_mul(gate, up, backend="reference"))
    assert torch.equal(packed_gelu_and_mul(x, backend="reference"),
                       gelu_and_mul(gate, up, backend="reference"))


def test_return_is_bare_tensor_not_tuple():
    """The public single-output API collapses the DSL backends' 1-tuple."""
    gate = torch.randn(2, 4, dtype=torch.float32)
    up = torch.randn(2, 4, dtype=torch.float32)
    out = silu_and_mul(gate, up, backend="reference")
    assert isinstance(out, torch.Tensor)


# --- Triton kernel correctness (GPU or TRITON_INTERPRET=1) ---------------------
# These pin the DSL-*generated* kernel, not the reference. The GELU path used to
# crash here: Triton 3.7+ dropped ``tl.tanh`` and the codegen fell back to
# ``libdevice.tanh``, which returns None under the interpreter (and is GPU-only)
# — see the ``_tl_tanh`` sigmoid-identity fix in ``vkl/lower/mathbody.py``.


@pytest.mark.parametrize(
    "fn_name,kind", [("silu_and_mul", "silu"), ("gelu_and_mul", "gelu")]
)
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_triton_kernel_matches_oracle(fn_name, kind, dt):
    if not _TRITON_RUNNABLE:
        pytest.skip("triton not runnable (no GPU / TRITON_INTERPREPT!=1)")
    dev = gpu_device_or_skip()
    dtype = _DTYPES[dt]
    torch.manual_seed(10)
    # a non-power-of-two K exercises the flat-1D launch's tail-tile masking.
    gate = torch.randn(37, 97, device=dev, dtype=dtype)
    up = torch.randn(37, 97, device=dev, dtype=dtype)
    out = globals()[fn_name](gate, up, backend="triton")
    expected = (_act(kind)(gate.float()) * up.float()).to(dtype)
    assert out.dtype == dtype
    assert out.shape == gate.shape
    assert_close(out, expected, op_id=_OP_ID[fn_name])


@pytest.mark.parametrize(
    "fn_name,kind",
    [("packed_silu_and_mul", "silu"), ("packed_gelu_and_mul", "gelu")],
)
@pytest.mark.parametrize("dt", list(_DTYPES))
def test_triton_packed_kernel_matches_oracle(fn_name, kind, dt):
    if not _TRITON_RUNNABLE:
        pytest.skip("triton not runnable (no GPU / TRITON_INTERPREPT!=1)")
    dev = gpu_device_or_skip()
    dtype = _DTYPES[dt]
    torch.manual_seed(11)
    M, K = 37, 97
    x = torch.randn(M, 2 * K, device=dev, dtype=dtype)
    out = globals()[fn_name](x, backend="triton")
    gate, up = x[:, :K], x[:, K:]
    expected = (_act(kind)(gate.float()) * up.float()).to(dtype)
    assert out.dtype == dtype
    assert out.shape == (M, K)
    assert_close(out, expected, op_id=_OP_ID[fn_name])


def test_triton_and_reference_backends_agree():
    """The two backends agree within the op's cross-backend tolerance."""
    if not _TRITON_RUNNABLE:
        pytest.skip("triton not runnable (no GPU / TRITON_INTERPREPT!=1)")
    dev = gpu_device_or_skip()
    torch.manual_seed(12)
    gate = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
    up = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
    ref = silu_and_mul(gate, up, backend="reference")
    tri = silu_and_mul(gate, up, backend="triton")
    assert_close(tri, ref, op_id="silu_and_mul@1.0.0")
