# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""VKL contract preflight gate.

``validate_kernel`` is the CPU-decidable half of the DSL publish path: it catches
schema drift, trace/body mistakes, unsupported launch/node pairings, and dishonest
target knob declarations before a GPU compile. ``verify`` / ``verify_parity``
remain the device correctness gate.
"""
from __future__ import annotations

from dataclasses import replace

from xkernels.vkl import spec_of, validate_kernel
from xkernels.vkl.examples import gemm_bf16


def test_validate_kernel_accepts_gemm_contract():
    spec = spec_of(gemm_bf16)
    result = validate_kernel(spec, arch="nvidia_sm90")
    assert result.passed
    assert result.to_dict()["error_count"] == 0


def test_validate_kernel_rejects_accum_dtype_mismatch():
    spec = spec_of(gemm_bf16)
    bad = replace(spec, numerics=replace(spec.numerics, reduce_dtype="bf16"))
    result = validate_kernel(bad, arch="nvidia_sm90")
    assert not result.passed
    assert any(i.code == "accum_dtype_mismatch" for i in result.issues)


def test_validate_kernel_rejects_unconsumed_triton_knob():
    spec = spec_of(gemm_bf16)
    target = spec.targets["triton"]
    bad_target = replace(
        target,
        knobs={**target.knobs, "BLOCK_Q": (16, 32)},
    )
    bad = replace(spec, targets={**spec.targets, "triton": bad_target})
    result = validate_kernel(bad, arch="nvidia_sm90")
    assert not result.passed
    assert any(i.code == "unsupported_knob" and "BLOCK_Q" in i.message for i in result.issues)
