# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
import os

import pytest
import torch
import torch.nn.functional as F

from xkernels.ops.mhc.reference import hc_prenorm_gemm_ref

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _full(a, fn):
    """Independent oracle: full F.linear(A, fn) and per-row sum of squares (fp32)."""
    af = a.float()
    return F.linear(af, fn.float()), (af * af).sum(dim=-1)


def test_reference_split_sum_invariant():
    dev = _dev()
    torch.manual_seed(0)
    T, hc_mult, hidden = 5, 4, 32
    K = hc_mult * hidden
    N = 2 * hc_mult + hc_mult * hc_mult  # 24
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    for n_splits in (1, 4, 16):
        mul, sqr = hc_prenorm_gemm_ref(a, fn, n_splits=n_splits)
        assert mul.shape == (n_splits, T, N)
        assert sqr.shape == (n_splits, T)
        assert mul.dtype == torch.float32 and sqr.dtype == torch.float32
        fmul, fsqr = _full(a, fn)
        torch.testing.assert_close(mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(sqr.sum(0), fsqr, atol=1e-4, rtol=1e-4)


def test_reference_empty_tokens():
    dev = _dev()
    a = torch.zeros(0, 16, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(6, 16, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm_ref(a, fn, n_splits=4)
    assert mul.shape == (4, 0, 6) and sqr.shape == (4, 0)


from xkernels.ops.mhc import hc_prenorm_gemm, tf32_hc_prenorm_gemm  # noqa: E402


def test_native_op_dispatches_to_reference():
    dev = _dev()
    torch.manual_seed(1)
    T, K, N = 3, 64, 8
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=4, backend="reference")
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(sqr.sum(0), fsqr, atol=1e-4, rtol=1e-4)


def test_faithful_wrapper_writes_in_place():
    """tf32_hc_prenorm_gemm matches the deep_gemm signature: in-place, returns None."""
    dev = _dev()
    torch.manual_seed(2)
    T, K, N, n_splits = 4, 128, 24, 3
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    gemm_out_mul = torch.empty(n_splits, T, N, device=dev, dtype=torch.float32)
    gemm_out_sqrsum = torch.empty(n_splits, T, device=dev, dtype=torch.float32)
    ret = tf32_hc_prenorm_gemm(a, fn, gemm_out_mul, gemm_out_sqrsum, n_splits,
                               backend="reference")
    assert ret is None
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(gemm_out_mul.sum(0), fmul, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(gemm_out_sqrsum.sum(0), fsqr, atol=1e-4, rtol=1e-4)


from xkernels._backends import Backend  # noqa: E402
from xkernels._dispatch import registered_backends  # noqa: E402

_HAS_TRITON = Backend.TRITON in registered_backends("hc_prenorm_gemm")


@pytest.mark.parametrize("hc_mult,hidden", [(4, 64), (2, 48), (4, 70)])  # 70 -> K not /64
@pytest.mark.parametrize("n_splits", [1, 4, 16])
def test_triton_matches_reference(hc_mult, hidden, n_splits):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    torch.manual_seed(3)
    T = 7
    K = hc_mult * hidden
    N = 2 * hc_mult + hc_mult * hc_mult
    dt = torch.float32 if _INTERP else torch.bfloat16
    a = torch.randn(T, K, device=dev, dtype=dt)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    got_mul, got_sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits, backend=Backend.TRITON)
    fmul, fsqr = _full(a, fn)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    # Only the SUM over splits is the invariant (Triton distributes K genuinely).
    torch.testing.assert_close(got_mul.sum(0), fmul, atol=atol, rtol=rtol)
    torch.testing.assert_close(got_sqr.sum(0), fsqr, atol=atol, rtol=rtol)


def test_triton_v4_flash_shape():
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _dev()
    if _INTERP:
        pytest.skip("V4 K=16384 too slow under the CPU interpreter")
    torch.manual_seed(4)
    T, hc_mult, hidden = 8, 4, 4096
    K, N = hc_mult * hidden, 24
    a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)
    mul, sqr = hc_prenorm_gemm(a, fn, n_splits=16, backend=Backend.TRITON)
    fmul, fsqr = _full(a, fn)
    torch.testing.assert_close(mul.sum(0), fmul, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(sqr.sum(0), fsqr, atol=2e-2, rtol=2e-2)


def test_top_level_exports():
    import xkernels
    for name in ("hc_prenorm_gemm", "tf32_hc_prenorm_gemm"):
        assert hasattr(xkernels, name), name
