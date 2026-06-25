# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: fused dual RMSNorm backends vs the two-launch torch oracle.

Runs on GPU (bf16) or CPU via ``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import dual_rmsnorm
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.norm.reference import dual_rmsnorm_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("dual_rmsnorm")


def _inputs(T, d1, d2, dtype, dev):
    torch.manual_seed(0)
    x1 = (torch.randn(T, d1, device=dev) * 0.5).to(dtype)
    x2 = (torch.randn(T, d2, device=dev) * 0.5).to(dtype)
    w1 = (torch.randn(d1, device=dev) * 0.1 + 1).to(dtype)
    w2 = (torch.randn(d2, device=dev) * 0.1 + 1).to(dtype)
    return x1, w1, x2, w2


# MLA-ish latent dims plus a non-power-of-2 case to exercise masking.
@pytest.mark.parametrize("T,d1,d2", [(4, 1536, 512), (16, 512, 512), (3, 320, 192)])
def test_triton_matches_reference(T, d1, d2):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    x1, w1, x2, w2 = _inputs(T, d1, d2, dtype, dev)
    o1, o2 = dual_rmsnorm(x1, w1, x2, w2, backend=Backend.TRITON)
    r1, r2 = dual_rmsnorm_ref(x1, w1, x2, w2)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(o1.float(), r1.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(o2.float(), r2.float(), atol=atol, rtol=rtol)


def test_reference_backend_matches_oracle():
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    x1, w1, x2, w2 = _inputs(8, 256, 128, dtype, dev)
    o1, o2 = dual_rmsnorm(x1, w1, x2, w2, backend=Backend.REFERENCE)
    r1, r2 = dual_rmsnorm_ref(x1, w1, x2, w2)
    torch.testing.assert_close(o1.float(), r1.float())
    torch.testing.assert_close(o2.float(), r2.float())
