# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Correctness: parametric xIELU (issue #80) backends vs the pure-torch oracle.

The reference oracle is bit-identical to ``transformers.XIELUActivation``
(verified on the mini-sglang CPU probe). Runs on GPU (bf16) or CPU via
``TRITON_INTERPRET=1`` (fp32).
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels import xielu
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.activation.reference import xielu as xielu_ref
from xkernels.utils.testing import gpu_device_or_skip as _device

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("xielu")


def _inputs(M, K, dtype, dev, alpha_p_val=0.3, alpha_n_val=0.4):
    torch.manual_seed(0)
    # span both xIELU branches: randn hits the positive side ~half the time and
    # is unbounded negative the other half, exercising the expm1(min(x,eps)) arm.
    x = torch.randn(M, K, device=dev).to(dtype)
    # raw log-space params. Default near init; callers pass large values to test
    # the softplus overflow regime (see test_triton_handles_large_checkpoint_params).
    alpha_p = torch.tensor([alpha_p_val], device=dev, dtype=dtype)
    alpha_n = torch.tensor([alpha_n_val], device=dev, dtype=dtype)
    return x, alpha_p, alpha_n


@pytest.mark.parametrize("M,K", [(4, 1536), (16, 4096), (3, 21504), (1, 97)])
def test_triton_matches_reference(M, K):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered")
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    x, alpha_p, alpha_n = _inputs(M, K, dtype, dev)
    out = xielu(x, alpha_p, alpha_n, backend=Backend.TRITON)
    ref = xielu_ref(x, alpha_p, alpha_n)
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


def test_reference_backend_matches_oracle():
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    x, alpha_p, alpha_n = _inputs(8, 256, dtype, dev)
    out = xielu(x, alpha_p, alpha_n, backend=Backend.REFERENCE)
    ref = xielu_ref(x, alpha_p, alpha_n)
    torch.testing.assert_close(out.float(), ref.float())


@pytest.mark.skipif(not _HAS_TRITON, reason="triton backend not registered")
def test_triton_handles_large_checkpoint_params():
    """Regression: the naive log(1+exp(z)) softplus overflows fp32 for z > ~88.

    swiss-ai/Apertus-8B-Instruct-2509 stores alpha_p = 166.0 in the checkpoint;
    the naive softplus yields inf and poisons every positive activation. The
    kernel must use the numerically stable formulation. With alpha_p=166 the
    reference (F.softplus) gives ~166.0; a broken naive kernel gives inf/NaN.
    """
    dev = _device()
    dtype = torch.float32 if _INTERP else torch.bfloat16
    x, alpha_p, alpha_n = _inputs(8, 4096, dtype, dev, alpha_p_val=166.0, alpha_n_val=40.75)
    out = xielu(x, alpha_p, alpha_n, backend=Backend.TRITON)
    ref = xielu_ref(x, alpha_p, alpha_n)
    assert torch.isfinite(out.float()).all(), "triton xielu non-finite (softplus overflow)"
    atol = rtol = 1e-4 if _INTERP else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)
