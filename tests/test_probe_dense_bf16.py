# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Unit tests for the bf16 dense-GEMM probe helpers (issue #17).

Pure / interpreter-level (no GPU). Skipped where Triton is absent because the
probe defines a @triton.jit kernel at import.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

pytest.importorskip("triton")

# probe_ffn lives in benchmarks/, which isn't an installed package.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from benchmarks import probe_ffn as P  # noqa: E402


def test_tflops_math():
    # 2*M*K*N flops; M=K=N=1000 -> 2e9 flops; at 1.0 ms -> 2.0 TFLOP/s
    assert abs(P._tflops(1.0, 1000, 1000, 1000) - 2.0) < 1e-9
    assert P._tflops(0.0, 1, 1, 1) == 0.0  # guard divide-by-zero


@pytest.mark.parametrize("mode", ["default", "hipblaslt", "no-hipblaslt", "tunableop"])
def test_apply_blas_mode_no_raise(mode):
    state = P._apply_blas_mode(mode)
    assert isinstance(state, dict)
    assert state["mode"] == mode


def test_triton_gemm_matches_torch():
    import torch

    # fp32 inputs: the Triton CPU interpreter mis-evaluates bf16 tl.dot, but the
    # tiling/masking/accumulate path is identical, so fp32 validates correctness.
    torch.manual_seed(0)
    M, K, N = 37, 70, 50  # non-tile-aligned -> exercises masking
    a = torch.randn(M, K)
    b = torch.randn(K, N)
    got = P._triton_gemm(a, b)
    ref = a @ b
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
