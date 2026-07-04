# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase D GPU half (issue #75 criterion #1): the native HIP override compiles +
passes verify on a real AMD chip (beverin / MI300A / gfx942). This validates the
per-target HIP override pipeline end-to-end on AMD hardware: a native hipcc kernel
(via ``load_inline``) registers as the ``hip`` backend and passes ``verify``
against the exact oracle.

Maturity bar (mirrors test_vkl_override_cuda.py exactly): the override compiles to
a REAL native HIP kernel and verifies on amd_cdna3. It is correct-but-slow
(wavefront FMA, NOT MFMA); the MFMA matrix-core ceiling is the
``map-to-matrix-cores`` follow-up (parallel to the CUTLASS/wgmma follow-up on the
CUDA side). This closes the issue-#75 criterion #1 gate at the mechanism-validation
maturity the CUDA twin ships at.

GPU-gated on an AMD GPU with hipcc (the tokenspeed-rocm uenv on beverin).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA/HIP device", allow_module_level=True)
# This test compiles a native HIP kernel — only meaningful on a ROCm build.
if not getattr(torch.version, "hip", None):
    pytest.skip("not a ROCm/HIP torch build", allow_module_level=True)

from xkernels import verify  # noqa: E402
from xkernels.vkl import (  # noqa: E402
    check_override_math_ir,
    register_dsl_hip,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16  # noqa: E402


def _spec_with_hip_override():
    """The gemm_bf16 spec carries the hip/amd_cdna3 override (examples file)."""
    return spec_of(gemm_bf16)


@pytest.fixture(scope="module")
def _registered():
    """Compile + register the native hip override once for the module."""
    spec = _spec_with_hip_override()
    assert spec.override_for("hip", "amd_cdna3") is not None, (
        "gemm_bf16 must declare a hip/amd_cdna3 override (examples/gemm_bf16.py)"
    )
    register_dsl_hip(spec, spec.override_for("hip", "amd_cdna3"))
    return spec


class TestOverrideMathIrInvariant:
    """The oracle property holds BEFORE any GPU compile (CPU-checkable)."""

    def test_hip_override_same_math_ir(self):
        spec = _spec_with_hip_override()
        ov = spec.override_for("hip", "amd_cdna3")
        chk = check_override_math_ir(spec, ov)
        assert chk.ok, f"oracle invariant failed: {chk.reason}"
        assert chk.portable_signature == chk.override_signature
        assert "MMA" in chk.portable_signature


class TestNativeHipOverride:
    """The native hip override compiles + passes correctness on the AMD chip."""

    def test_bf16_compiles_and_passes(self, _registered):
        res = verify(
            "gemm_bf16.hip@1.0.0",
            arch="amd_cdna3",
            shapes=[{"dtype": "bf16", "M": 128, "N": 256, "K": 256}],
        )
        assert res["compiled"], (
            f"native hip kernel did not compile: {res.get('artifacts', {}).get('error')}"
        )
        assert res["correctness"]["passed"], (
            f"bf16 correctness failed: {res['correctness']}"
        )

    def test_fp32_true_fp32(self, _registered):
        """The native hip override is TRUE fp32 (FMA, no downcast): bit-exact-ish
        with the CPU fp32 reference. (Same mechanism-validation baseline as the
        cuda twin's fp32 test.)"""
        res = verify(
            "gemm_bf16.hip@1.0.0",
            arch="amd_cdna3",
            shapes=[{"dtype": "fp32", "M": 128, "N": 256, "K": 256}],
        )
        assert res["compiled"]
        assert res["correctness"]["passed"], (
            f"fp32 native correctness failed: {res['correctness']}"
        )
        assert res["correctness"]["max_abs_err"] < 1e-3, (
            f"expected true-fp32 floor, got {res['correctness']['max_abs_err']}"
        )

    def test_native_matches_portable_across_sweep(self, _registered):
        """Both bf16 and fp32 sweep points pass on the native card."""
        res = verify(
            "gemm_bf16.hip@1.0.0",
            arch="amd_cdna3",
        )
        assert res["compiled"]
        assert res["correctness"]["passed"], res["correctness"]
