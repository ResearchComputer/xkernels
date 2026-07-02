# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 2.1 GPU half: the native CUDA override compiles + passes verify on a
real chip (ds5 / GB10 sm_121). This validates the per-target override pipeline
end-to-end on hardware: a native nvcc kernel (via ``load_inline``) registers as
the ``cuda`` backend and passes ``verify`` against the exact oracle.

These tests are GPU-gated on an NVIDIA GPU with nvcc (the NGC pytorch container
on ds5). They compile a real native CUDA kernel via ``load_inline``, register it
as the ``cuda`` backend, and run ``verify`` on it — the docs/brainstorm/04 Ex.2
loop closed on hardware.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from xkernels import verify  # noqa: E402
from xkernels.vkl import (  # noqa: E402
    check_override_math_ir,
    register_dsl_cuda,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16  # noqa: E402


def _spec_with_cuda_override():
    """The gemm_bf16 spec carries the cuda/sm_121 override (examples file)."""
    return spec_of(gemm_bf16)


@pytest.fixture(scope="module")
def _registered():
    """Compile + register the native cuda override once for the module."""
    spec = _spec_with_cuda_override()
    assert spec.override_for("cuda", "nvidia_sm121") is not None, (
        "gemm_bf16 must declare a cuda/sm_121 override (examples/gemm_bf16.py)"
    )
    register_dsl_cuda(spec, spec.override_for("cuda", "nvidia_sm121"))
    return spec


class TestOverrideMathIrInvariant:
    """The oracle property holds BEFORE any GPU compile (CPU-checkable)."""

    def test_cuda_override_same_math_ir(self):
        spec = _spec_with_cuda_override()
        ov = spec.override_for("cuda", "nvidia_sm121")
        chk = check_override_math_ir(spec, ov)
        assert chk.ok, f"oracle invariant failed: {chk.reason}"
        assert chk.portable_signature == chk.override_signature
        assert "MMA" in chk.portable_signature


class TestNativeCudaOverride:
    """The native cuda override compiles + passes correctness on the chip."""

    def test_bf16_compiles_and_passes(self, _registered):
        res = verify(
            "gemm_bf16.cuda@1.0.0",
            arch="nvidia_sm121",
            shapes=[{"dtype": "bf16", "M": 128, "N": 256, "K": 256}],
        )
        assert res["compiled"], (
            f"native cuda kernel did not compile: {res.get('artifacts', {}).get('error')}"
        )
        assert res["correctness"]["passed"], (
            f"bf16 correctness failed: {res['correctness']}"
        )

    def test_fp32_true_fp32_on_blackwell(self, _registered):
        """The native cuda override is TRUE fp32 (CUDA-core FMA, no TF32): bit-exact
        with the CPU fp32 reference. (The triton backend ALSO does true fp32 on
        this arch; the earlier "triton degrades to tf32" reports were an
        oracle-side bug, fixed in ``run_reference``. This test pins the native
        override's exactness as the mechanism-validation baseline.)"""
        res = verify(
            "gemm_bf16.cuda@1.0.0",
            arch="nvidia_sm121",
            shapes=[{"dtype": "fp32", "M": 128, "N": 256, "K": 256}],
        )
        assert res["compiled"]
        assert res["correctness"]["passed"], (
            f"fp32 native correctness failed: {res['correctness']}"
        )
        # the abs-err is at the true-fp32 floor (CUDA-core FMA vs the exact
        # oracle), well inside the fp32 tolerance.
        assert res["correctness"]["max_abs_err"] < 1e-3, (
            f"expected true-fp32 floor, got {res['correctness']['max_abs_err']}"
        )

    def test_native_matches_portable_across_sweep(self, _registered):
        """Both bf16 and fp32 sweep points pass on the native card."""
        res = verify(
            "gemm_bf16.cuda@1.0.0",
            arch="nvidia_sm121",
        )
        assert res["compiled"]
        assert res["correctness"]["passed"], res["correctness"]
