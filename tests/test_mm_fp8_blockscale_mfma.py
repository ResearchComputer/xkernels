# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Tests for the native fp8 MFMA block-scale dense GEMM (issue #41)."""
import os

import pytest
import torch

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")
pytestmark = pytest.mark.skipif(not _HAS_FP8, reason="torch lacks float8_e4m3fn")
_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def test_config_space_is_valid():
    from xkernels.ops.gemm.triton.configs import get_autotune_configs, get_fp8_gemm_config

    cfgs = get_autotune_configs()
    assert len(cfgs) >= 6
    for c in cfgs:
        k = c.kwargs
        assert 128 % k["BLOCK_K"] == 0, "BLOCK_K must divide the 128 quant block"
        assert k["BLOCK_M"] in (16, 32, 64, 128, 256)
        assert k["BLOCK_N"] in (64, 128, 256)
    # Baked direct-launch config: decode (tiny M) vs prefill (large M) differ.
    dec = get_fp8_gemm_config(1, 512, 7168)
    pre = get_fp8_gemm_config(4096, 7168, 2048)
    assert 128 % dec["BLOCK_K"] == 0 and 128 % pre["BLOCK_K"] == 0
    assert dec["BLOCK_M"] <= pre["BLOCK_M"]


from xkernels.ops.gemm.reference import (  # noqa: E402
    mm_fp8_blockscale_ref,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
)


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(M, N, K, block, dev, seed=0, fp8_dtype=torch.float8_e4m3fn):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device=dev, dtype=torch.float32)
    b = torch.randn(N, K, device=dev, dtype=torch.float32)
    a_fp8, a_s = per_token_group_quant_fp8(a, block=block, fp8_dtype=fp8_dtype)
    b_fp8, b_s = per_block_quant_fp8(b, block=block, fp8_dtype=fp8_dtype)
    ref = mm_fp8_blockscale_ref(a_fp8, a_s, b_fp8, b_s, block=block, out_dtype=torch.float32)
    return a_fp8, a_s, b_fp8, b_s, ref


def _rel(got, ref):
    err = (got.float() - ref.float()).abs().max().item()
    return err / ref.float().abs().max().clamp_min(1e-6).item()


@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (37, 130, 384), (7, 24, 320), (1, 256, 512)])
def test_mfma_matches_reference_interpreter(M, N, K):
    """Block-promotion math vs the fp32 dequant oracle (fp8 tl.dot is exact under
    the interpreter -> tight)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import (
        mm_fp8_blockscale_mfma_triton,
    )

    a_fp8, a_s, b_fp8, b_s, ref = _inputs(M, N, K, 128, _dev())
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert got.shape == (M, N)
    assert _rel(got, ref) < (1e-3 if _INTERP else 5e-3)


def test_mfma_empty_m_interpreter():
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import (
        mm_fp8_blockscale_mfma_triton,
    )

    dev = _dev()
    a_fp8 = torch.zeros(0, 128, device=dev, dtype=torch.float8_e4m3fn)
    a_s = torch.zeros(0, 1, device=dev, dtype=torch.float32)
    b_fp8 = torch.zeros(8, 128, device=dev, dtype=torch.float8_e4m3fn)
    b_s = torch.ones(1, 1, device=dev, dtype=torch.float32)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128)
    assert got.shape == (0, 8)


from xkernels._backends import Backend  # noqa: E402
from xkernels.ops.gemm import mm_fp8_blockscale  # noqa: E402


@pytest.mark.parametrize("path", ["auto", "mfma", "portable"])
def test_entry_path_routing_interpreter(path):
    """All three Triton paths reproduce the fp32 oracle under the interpreter."""
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(32, 128, 256, 128, _dev())
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32,
        path=path, backend=Backend.TRITON,
    )
    assert _rel(got, ref) < (1e-3 if _INTERP else 5e-3)


def test_dot_bf16_forces_portable_interpreter():
    """dot_bf16=True is a portable-only knob; auto must honor it (route portable)."""
    if _INTERP:
        pytest.skip("CPU interpreter mis-evaluates a bf16 tl.dot")
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(16, 128, 256, 128, _dev())
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32,
        dot_bf16=True, backend=Backend.TRITON,
    )
    assert _rel(got, ref) < 2e-2


# --- GPU-gated (gfx942) cases: skipped locally, run on beverin in Task 7. ---

_GPU = (not _INTERP) and torch.cuda.is_available()


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
@pytest.mark.parametrize("M,N,K", [(64, 128, 256), (8, 512, 7168), (2048, 512, 7168)])
def test_mfma_tight_parity_gpu(M, N, K):
    """Native fp8 MFMA vs the fp32 dequant oracle: TIGHT (<5e-3). A loose result
    means an fp8-format mismatch reached the matrix unit (the fn-vs-fnuz detector)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import (
        mm_fp8_blockscale_mfma_triton,
    )

    a_fp8, a_s, b_fp8, b_s, ref = _inputs(M, N, K, 128, "cuda", seed=4)
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert _rel(got, ref) < 5e-3, (M, N, K, _rel(got, ref))


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
def test_mfma_cross_checks_portable_gpu():
    """The two independent Triton implementations agree within fp8 tolerance."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_kernel import (
        mm_fp8_blockscale_triton as portable,
    )
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import (
        mm_fp8_blockscale_mfma_triton as mfma,
    )

    a_fp8, a_s, b_fp8, b_s, _ = _inputs(48, 256, 512, 128, "cuda")
    p = portable(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    m = mfma(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    assert _rel(m, p) < 5e-3


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fnuz"), reason="no fnuz")
def test_mfma_fnuz_operands_gpu():
    """fnuz operands also produce a correct GEMM (the AMD-native fp8 encoding)."""
    from xkernels.ops.gemm.triton.mm_fp8_blockscale_mfma_kernel import (
        mm_fp8_blockscale_mfma_triton,
    )

    a_fp8, a_s, b_fp8, b_s, ref = _inputs(
        64, 256, 512, 128, "cuda", fp8_dtype=torch.float8_e4m3fnuz
    )
    got = mm_fp8_blockscale_mfma_triton(a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32)
    # fnuz (max 240) is coarser than fn -> a looser but still real parity bound.
    assert _rel(got, ref) < 3e-2


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fnuz"), reason="no fnuz")
def test_auto_routes_fnuz_to_mfma_gpu():
    """auto + fnuz operands -> mfma fast path (the CPU interpreter cannot lower a
    fnuz tl.dot, so this is GPU-only); still matches the fp32 oracle."""
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(
        32, 128, 256, 128, "cuda", fp8_dtype=torch.float8_e4m3fnuz
    )
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, out_dtype=torch.float32,
        path="auto", backend=Backend.TRITON,
    )
    assert _rel(got, ref) < 5e-3


@pytest.mark.skipif(not _GPU, reason="needs gfx942 GPU")
def test_mfma_bf16_out_gpu():
    a_fp8, a_s, b_fp8, b_s, ref = _inputs(48, 64, 256, 128, "cuda")
    got = mm_fp8_blockscale(
        a_fp8, a_s, b_fp8, b_s, block=128, path="mfma", backend=Backend.TRITON
    )
    assert got.dtype == torch.bfloat16 and _rel(got, ref) < 5e-3
