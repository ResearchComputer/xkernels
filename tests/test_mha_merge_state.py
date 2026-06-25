# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: mha_merge_state backends vs the online-softmax torch oracle.

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import mha_merge_state
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.attention.reference import merge_state_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("mha_merge_state")


def _inputs(T, H, D, dtype, dev):
    torch.manual_seed(0)
    out_a = torch.randn(T, H, D, device=dev, dtype=dtype)
    out_b = torch.randn(T, H, D, device=dev, dtype=dtype)
    # LSEs span a range so the max/weight selection is exercised.
    lse_a = (torch.randn(T, H, device=dev) * 2.0).float()
    lse_b = (torch.randn(T, H, device=dev) * 2.0).float()
    return out_a, lse_a, out_b, lse_b


@pytest.mark.parametrize("T,H,D", [(4, 8, 128), (2, 16, 64), (3, 5, 96)])
def test_triton_matches_reference(T, H, D):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    out_a, lse_a, out_b, lse_b = _inputs(T, H, D, dtype, dev)
    out, lse = mha_merge_state(out_a, lse_a, out_b, lse_b, backend=Backend.TRITON)
    ref_out, ref_lse = merge_state_ref(out_a, lse_a, out_b, lse_b)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(out.float(), ref_out.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)


def test_reference_backend_matches_oracle():
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    out_a, lse_a, out_b, lse_b = _inputs(2, 4, 64, dtype, dev)
    out, lse = mha_merge_state(out_a, lse_a, out_b, lse_b, backend=Backend.REFERENCE)
    ref_out, ref_lse = merge_state_ref(out_a, lse_a, out_b, lse_b)
    torch.testing.assert_close(out.float(), ref_out.float())
    torch.testing.assert_close(lse, ref_lse)
