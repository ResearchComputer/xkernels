# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Tests for the DeepSeek-V4 fp8 block-scale dense GEMM (issue #38)."""
import os

import pytest
import torch

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
pytestmark = pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")

from xkernels.ops.gemm.reference import (  # noqa: E402
    FP8_BLOCK,
    mm_fp8_blockscale_ref,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _quantized_inputs(M, N, K, block, dev, seed=0):
    """Random fp8 block-scale A/B plus the exact fp32 dequant oracle output."""
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    b = torch.randn(N, K, device=dev, dtype=torch.float32)
    a_fp8, a_scales = per_token_group_quant_fp8(a, block=block)
    b_fp8, b_scales = per_block_quant_fp8(b, block=block)
    # Oracle: dequant then fp32 matmul (independent of the op under test only in
    # that it reuses the same dequant; the invariant is that quant->dequant->mm
    # round-trips to the dequantized operands, which is what serving consumes).
    a_deq = a_fp8.to(torch.float32) * a_scales.repeat_interleave(block, 1)[:, :K]
    b_deq = b_fp8.to(torch.float32) * (
        b_scales.repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    )
    ref = a_deq @ b_deq.t()
    return a_fp8, a_scales, b_fp8, b_scales, ref


def test_quant_roundtrip_shapes():
    dev = _dev()
    M, N, K, block = 5, 7, 256, 128
    a = torch.randn(M, K, device=dev)
    b = torch.randn(N, K, device=dev)
    a_fp8, a_scales = per_token_group_quant_fp8(a, block=block)
    b_fp8, b_scales = per_block_quant_fp8(b, block=block)
    assert a_fp8.shape == (M, K) and a_fp8.dtype == torch.float8_e4m3fn
    assert a_scales.shape == (M, (K + block - 1) // block)
    assert b_fp8.shape == (N, K) and b_fp8.dtype == torch.float8_e4m3fn
    assert b_scales.shape == ((N + block - 1) // block, (K + block - 1) // block)


def test_reference_matches_explicit_dequant():
    dev = _dev()
    block = 128
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(6, 130, 384, block, dev)
    out = mm_fp8_blockscale_ref(
        a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=torch.float32
    )
    assert out.shape == (6, 130) and out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_reference_default_bf16_out():
    dev = _dev()
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(4, 8, 128, 128, dev)
    out = mm_fp8_blockscale_ref(a_fp8, a_scales, b_fp8, b_scales)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)


def test_reference_rejects_bad_scale_shapes():
    dev = _dev()
    a_fp8, a_scales, b_fp8, b_scales, _ = _quantized_inputs(4, 8, 128, 128, dev)
    with pytest.raises(ValueError):
        mm_fp8_blockscale_ref(a_fp8, a_scales[:, :0], b_fp8, b_scales)
    with pytest.raises(ValueError):
        mm_fp8_blockscale_ref(a_fp8, a_scales, b_fp8, b_scales[:0])


from xkernels.ops.gemm import mm_fp8_blockscale  # noqa: E402


def test_native_op_dispatches_to_reference():
    dev = _dev()
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(3, 5, 256, 128, dev)
    out = mm_fp8_blockscale(
        a_fp8, a_scales, b_fp8, b_scales, out_dtype=torch.float32, backend="reference"
    )
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


from xkernels._backends import Backend  # noqa: E402
from xkernels._dispatch import registered_backends  # noqa: E402

_HAS_TRITON = Backend.TRITON in registered_backends("mm_fp8_blockscale")


def _rel_err(got, ref):
    """max|err| normalized by the reference's max magnitude (norm-relative, robust
    to near-zero output elements where an elementwise rtol explodes)."""
    err = (got.float() - ref.float()).abs().max().item()
    return err / ref.float().abs().max().clamp_min(1e-6).item()


@pytest.mark.parametrize(
    "M,N,K",
    [
        (64, 128, 256),  # all block-aligned
        (37, 130, 384),  # M, N not block-aligned
        (7, 24, 320),  # K not a multiple of 128 (320 = 2*128 + 64)
        (1, 256, 512),  # decode (M=1)
    ],
)
def test_triton_matches_reference(M, N, K):
    """Default (exact fp32 dot) Triton path reproduces the fp32 dequant oracle."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    block = 128
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(M, N, K, block, dev)
    got = mm_fp8_blockscale(
        a_fp8, a_scales, b_fp8, b_scales,
        block=block, out_dtype=torch.float32, backend=Backend.TRITON,
    )
    assert got.shape == (M, N)
    # Exact fp32 dot: matches the fp32 dequant oracle up to accumulation order.
    if _INTERP:
        torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    else:
        assert _rel_err(got, ref) < 1e-3


@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (37, 130, 384), (8, 512, 1024)])
def test_triton_bf16_dot_opt_in(M, N, K):
    """Opt-in bf16-MFMA dot: faster, only ~bf16-accurate -> norm-relative check."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if _INTERP:
        pytest.skip("CPU interpreter mis-evaluates a bf16 tl.dot")
    dev = _dev()
    block = 128
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(M, N, K, block, dev)
    got = mm_fp8_blockscale(
        a_fp8, a_scales, b_fp8, b_scales,
        block=block, out_dtype=torch.float32, dot_bf16=True, backend=Backend.TRITON,
    )
    assert _rel_err(got, ref) < 2e-2


def test_triton_bf16_out_matches_reference():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    block = 128
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(48, 64, 256, block, dev)
    got = mm_fp8_blockscale(
        a_fp8, a_scales, b_fp8, b_scales, block=block, backend=Backend.TRITON
    )
    assert got.dtype == torch.bfloat16
    assert _rel_err(got, ref) < 2e-2


def test_triton_v4_mla_shape():
    """V4-Flash-ish MLA projection: K=7168 (DeepSeek hidden), N a kv_b head dim."""
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    if _INTERP:
        pytest.skip("K=7168 too slow under the CPU interpreter")
    dev = _dev()
    block = 128
    M, N, K = 8, 512, 7168
    a_fp8, a_scales, b_fp8, b_scales, ref = _quantized_inputs(M, N, K, block, dev, seed=4)
    for dot_bf16 in (False, True):
        got = mm_fp8_blockscale(
            a_fp8, a_scales, b_fp8, b_scales,
            block=block, out_dtype=torch.float32, dot_bf16=dot_bf16,
            backend=Backend.TRITON,
        )
        assert _rel_err(got, ref) < 2e-2, dot_bf16


def test_empty_m_returns_empty():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    block = 128
    a_fp8 = torch.zeros(0, 128, device=dev, dtype=torch.float8_e4m3fn)
    a_scales = torch.zeros(0, 1, device=dev, dtype=torch.float32)
    b_fp8 = torch.zeros(8, 128, device=dev, dtype=torch.float8_e4m3fn)
    b_scales = torch.ones(1, 1, device=dev, dtype=torch.float32)
    got = mm_fp8_blockscale(
        a_fp8, a_scales, b_fp8, b_scales, block=block, backend=Backend.TRITON
    )
    assert got.shape == (0, 8)


def test_top_level_exports():
    import xkernels
    for name in ("mm_fp8_blockscale", "per_token_group_quant_fp8", "per_block_quant_fp8"):
        assert hasattr(xkernels, name), name


def test_block_constant():
    assert FP8_BLOCK == 128


def test_quant_all_zeros_group():
    """All-zero quantization groups must dequantize back to zeros (no NaN)."""
    dev = _dev()
    block = 128
    a = torch.zeros(4, 256, device=dev)
    b = torch.zeros(8, 256, device=dev)
    a_fp8, a_scales = per_token_group_quant_fp8(a, block=block)
    b_fp8, b_scales = per_block_quant_fp8(b, block=block)
    a_deq = a_fp8.to(torch.float32) * a_scales.repeat_interleave(block, 1)[:, :256]
    b_deq = b_fp8.to(torch.float32) * (
        b_scales.repeat_interleave(block, 0)[:8].repeat_interleave(block, 1)[:, :256]
    )
    torch.testing.assert_close(a_deq, torch.zeros_like(a_deq), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b_deq, torch.zeros_like(b_deq), atol=1e-6, rtol=1e-6)
    out = mm_fp8_blockscale_ref(a_fp8, a_scales, b_fp8, b_scales, out_dtype=torch.float32)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=1e-6)
