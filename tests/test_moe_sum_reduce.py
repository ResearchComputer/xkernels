# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: moe_sum_reduce backends vs the torch oracle.

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import moe_sum_reduce
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe.sum_reduce import moe_sum_reduce_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("moe_sum_reduce")


@pytest.mark.parametrize("M,top_k,H", [(8, 8, 256), (4, 2, 7168), (5, 4, 96)])
@pytest.mark.parametrize("use_w", [False, True])
@pytest.mark.parametrize("scaling", [1.0, 2.5])
def test_triton_matches_reference(M, top_k, H, use_w, scaling):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    torch.manual_seed(0)
    y = torch.randn(M, top_k, H, device=dev, dtype=dtype)
    w = torch.rand(M, top_k, device=dev, dtype=torch.float32) if use_w else None
    got = moe_sum_reduce(y, w, routed_scaling_factor=scaling, backend=Backend.TRITON)
    ref = moe_sum_reduce_ref(y, w, scaling)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)


def test_reference_backend_matches_oracle():
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    torch.manual_seed(0)
    y = torch.randn(4, 4, 128, device=dev, dtype=dtype)
    w = torch.rand(4, 4, device=dev, dtype=torch.float32)
    got = moe_sum_reduce(y, w, routed_scaling_factor=1.5, backend=Backend.REFERENCE)
    ref = moe_sum_reduce_ref(y, w, 1.5)
    torch.testing.assert_close(got.float(), ref.float())
