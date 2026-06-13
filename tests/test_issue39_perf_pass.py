# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #39 perf pass for the V4 sparse-MLA (#33) + MHC prenorm GEMM (#37) kernels.

The perf pass makes the launch parameters (``BLOCK_N`` / ``BLOCK_M`` / ``BLOCK_K``
and the CDNA3 lowering knobs ``waves_per_eu`` / ``matrix_instr_nonkdim`` /
``kpack``) **tunable and env-overridable**, with defaults that reproduce the
original #33 / #37 launches. These knobs are pure performance knobs — the
numerical result must be invariant to them. These tests pin that invariance (so a
future config promotion cannot silently change correctness) and the config
resolution contract. Runs on CPU via ``TRITON_INTERPRET=1`` or on a GPU.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from xkernels._backends import Backend
from xkernels._dispatch import registered_backends

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


def _dev():
    if _INTERP:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Config resolution contract                                                  #
# --------------------------------------------------------------------------- #
def test_mhc_config_default_is_measured_winner(monkeypatch):
    from xkernels.ops.mhc.triton.configs import (
        BASELINE_MHC_GEMM_CONFIG,
        DEFAULT_MHC_GEMM_CONFIG,
        resolve_mhc_gemm_config,
    )

    monkeypatch.delenv("XKERNELS_MHC_GEMM_CONFIG", raising=False)
    cfg = resolve_mhc_gemm_config()
    # Promoted from the #39 on-device sweep (beverin/MI300A, job 384616).
    assert cfg["BLOCK_M"] == 32 and cfg["BLOCK_K"] == 128
    assert cfg == DEFAULT_MHC_GEMM_CONFIG
    # The original #36 launch is retained for A/B regression.
    assert BASELINE_MHC_GEMM_CONFIG["BLOCK_M"] == 64
    assert BASELINE_MHC_GEMM_CONFIG["BLOCK_K"] == 64


def test_mhc_config_env_override_and_partial(monkeypatch):
    from xkernels.ops.mhc.triton.configs import resolve_mhc_gemm_config

    monkeypatch.setenv("XKERNELS_MHC_GEMM_CONFIG", '{"BLOCK_K": 256}')
    cfg = resolve_mhc_gemm_config()
    assert cfg["BLOCK_K"] == 256
    assert cfg["BLOCK_M"] == 32  # untouched key falls back to default (winner)
    monkeypatch.setenv("XKERNELS_MHC_GEMM_CONFIG", "not json")
    with pytest.raises(ValueError):
        resolve_mhc_gemm_config()


def test_sparse_mla_config_default_and_override(monkeypatch):
    from xkernels.ops.attention.triton.sparse_mla_config import (
        DECODE_SPARSE_MLA_CONFIG,
        resolve_sparse_mla_config,
    )

    monkeypatch.delenv("XKERNELS_SPARSE_MLA_CONFIG", raising=False)
    # Default stays #33 (BLOCK_N=64): the #39 sweep found no static winner — wider
    # BLOCK_N regresses the multi-token case, so the decode win is opt-in.
    assert resolve_sparse_mla_config()["BLOCK_N"] == 64
    # The opt-in single-token decode config is the 1.13-1.24x Tq=1 winner.
    assert DECODE_SPARSE_MLA_CONFIG["BLOCK_N"] == 128
    assert DECODE_SPARSE_MLA_CONFIG["num_warps"] == 8
    monkeypatch.setenv("XKERNELS_SPARSE_MLA_CONFIG", '{"BLOCK_N": 128}')
    assert resolve_sparse_mla_config()["BLOCK_N"] == 128
    # BLOCK_N must be a power of two (chunk-aligned).
    monkeypatch.setenv("XKERNELS_SPARSE_MLA_CONFIG", '{"BLOCK_N": 100}')
    with pytest.raises(ValueError):
        resolve_sparse_mla_config()


# --------------------------------------------------------------------------- #
# Numerical invariance under the perf knobs                                    #
# --------------------------------------------------------------------------- #
_HAS_MLA_TRITON = Backend.TRITON in registered_backends("sparse_mla_attention")
_HAS_MHC_TRITON = Backend.TRITON in registered_backends("hc_prenorm_gemm")


@pytest.mark.skipif(not _HAS_MLA_TRITON, reason="triton backend not registered")
@pytest.mark.parametrize("block_n", ["32", "64", "128", "256"])
def test_sparse_mla_result_invariant_to_block_n(block_n, monkeypatch):
    """Changing BLOCK_N must not change the sparse-MLA result — the flash
    reduction is exact for any chunk size, including topk not a multiple of it."""
    from xkernels.ops.attention.interface import sparse_mla_attention

    dev = _dev()
    torch.manual_seed(11)
    dt = torch.float32 if _INTERP else torch.bfloat16
    T, H, D, d_v, Kv, topk = 4, 6, 64, 56, 256, 96  # topk not a multiple of 64/128/256
    q = torch.randn(T, H, D, device=dev, dtype=dt)
    kv = torch.randn(Kv, D, device=dev, dtype=dt)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    idx[0, -5:] = -1
    lens = torch.randint(1, topk + 1, (T,), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)

    monkeypatch.delenv("XKERNELS_SPARSE_MLA_CONFIG", raising=False)
    ref = sparse_mla_attention(
        q, kv, idx, sm_scale=0.1, topk_length=lens, attn_sink=sink,
        d_v=d_v, backend=Backend.TRITON,
    )
    monkeypatch.setenv("XKERNELS_SPARSE_MLA_CONFIG", f'{{"BLOCK_N": {block_n}}}')
    try:
        got = sparse_mla_attention(
            q, kv, idx, sm_scale=0.1, topk_length=lens, attn_sink=sink,
            d_v=d_v, backend=Backend.TRITON,
        )
    except Exception as exc:  # OutOfResources: config infeasible on this device
        if "out of resource" in str(exc).lower() or "OutOfResources" in type(exc).__name__:
            pytest.skip(f"BLOCK_N={block_n} exceeds device LDS")
        raise
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(got[0].float(), ref[0].float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(got[1], ref[1], atol=atol, rtol=rtol)


# (BLOCK_K, num_stages): BLOCK_K=256 fp32 only fits CDNA3's 64 KB LDS at
# num_stages=1 (num_stages=2 needs 96 KB -> OutOfResources on gfx942, observed
# on beverin job 384614). The wider tiles use num_stages=1 accordingly.
@pytest.mark.skipif(not _HAS_MHC_TRITON, reason="triton backend not registered")
@pytest.mark.parametrize("block_k,num_stages", [(64, 2), (128, 2), (256, 1)])
def test_mhc_gemm_sum_invariant_to_block_k(block_k, num_stages, monkeypatch):
    """The summed GEMM/sqsum invariant must hold for any BLOCK_K (the split-K
    partition is by k-block range; the downstream only sums over splits). A
    config that exceeds the device's LDS is skipped, not failed — it is a
    hardware-capacity limit, not a correctness bug."""
    from xkernels.ops.mhc import hc_prenorm_gemm

    dev = _dev()
    torch.manual_seed(12)
    dt = torch.float32 if _INTERP else torch.bfloat16
    T, hc_mult, hidden = 7, 4, 70  # K=280 not a multiple of 64/128/256
    K, N = hc_mult * hidden, 2 * hc_mult + hc_mult * hc_mult
    a = torch.randn(T, K, device=dev, dtype=dt)
    fn = torch.randn(N, K, device=dev, dtype=torch.float32)

    monkeypatch.setenv(
        "XKERNELS_MHC_GEMM_CONFIG",
        f'{{"BLOCK_K": {block_k}, "num_stages": {num_stages}}}',
    )
    try:
        mul, sqr = hc_prenorm_gemm(a, fn, n_splits=8, backend=Backend.TRITON)
    except Exception as exc:  # OutOfResources: config infeasible on this device
        if "out of resource" in str(exc).lower() or "OutOfResources" in type(exc).__name__:
            pytest.skip(f"BLOCK_K={block_k} num_stages={num_stages} exceeds device LDS")
        raise
    fmul = F.linear(a.float(), fn.float())
    fsqr = (a.float() ** 2).sum(-1)
    atol = rtol = 1e-3 if _INTERP else 2e-2
    torch.testing.assert_close(mul.sum(0), fmul, atol=atol, rtol=rtol)
    torch.testing.assert_close(sqr.sum(0), fsqr, atol=atol, rtol=rtol)
