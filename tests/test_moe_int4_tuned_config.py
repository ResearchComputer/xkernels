# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Unit tests for the tuned INT4 W4A16 MoE config loader + selection (issue #16).

Pure-Python (no GPU). Skipped where Triton is absent, because the config module
imports Triton for the autotune ``Config`` builders.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("triton")

from xkernels.ops.moe.triton import configs as C  # noqa: E402


def test_align_block_m():
    assert C.align_block_m(1) == 16
    assert C.align_block_m(16) == 16
    assert C.align_block_m(32) == 16
    assert C.align_block_m(33) == 64
    assert C.align_block_m(4096) == 64


def _table():
    return {
        "_provenance": {"device": "X"},
        "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
              "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2, "_ms": 0.01},
        "8": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
              "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2},
        "64": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
               "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2},
    }


def test_select_exact_bucket():
    cfg = C._select_config(_table(), 8)
    assert cfg["BLOCK_SIZE_N"] == 128 and cfg["num_warps"] == 4


def test_select_closest_below():
    assert C._select_config(_table(), 5)["BLOCK_SIZE_N"] == 64    # -> bucket 1
    assert C._select_config(_table(), 40)["num_warps"] == 4       # -> bucket 8


def test_select_clamps():
    assert C._select_config(_table(), 0)["BLOCK_SIZE_N"] == 64    # below min -> min
    assert C._select_config(_table(), 100000)["BLOCK_SIZE_M"] == 64  # above max -> max


def test_select_strips_metadata_keys():
    assert "_ms" not in C._select_config(_table(), 1)


def test_get_config_missing_returns_none():
    assert C.get_moe_int4_config(48, 1, 1, 1, arch="No_Such_Device") is None


def test_loader_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_config_dir", lambda: str(tmp_path))
    C._TUNED_CACHE.clear()
    fname = C._config_filename(48, 4096, 7168, "Test_Dev", "int4_w4a16")
    (tmp_path / fname).write_text(json.dumps(_table()))
    cfg = C.get_moe_int4_config(48, 4096, 7168, 8, arch="Test Dev")
    assert cfg is not None and cfg["num_warps"] == 4
    C._TUNED_CACHE.clear()
