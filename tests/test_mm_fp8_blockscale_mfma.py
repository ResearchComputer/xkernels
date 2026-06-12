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
