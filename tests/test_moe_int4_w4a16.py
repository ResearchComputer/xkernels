# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Numerical correctness: INT4 W4A16 fused-MoE GEMM backends vs PyTorch oracle.

Acceptance (issue #1): match dequant-then-matmul within ``atol/rtol ~ 2e-2``
(bf16). Runs on:

* GPU (NVIDIA or AMD gfx942) with a real Triton install -> bf16 activations,
  the production dtype, ``atol/rtol = 2e-2``.
* CPU via ``TRITON_INTERPRET=1`` (no GPU) -> **fp32** activations, ``atol/rtol =
  3e-3``. NOTE: the Triton CPU interpreter (>=3.4) mis-evaluates ``tl.dot`` with
  bf16 operands (returns garbage); fp32 exercises the identical kernel path
  (unpack, group-scale broadcast, dot, accumulate, masking, dispatch) since
  ``b_deq`` is always cast to ``a.dtype`` before the dot.

Usage::

    pytest tests/test_moe_int4_w4a16.py                       # GPU, bf16
    TRITON_INTERPRET=1 pytest tests/test_moe_int4_w4a16.py     # CPU, fp32
"""

from __future__ import annotations

import os

import pytest
import torch

from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.moe import dequant_w4a16, fused_moe_int4_w4a16, make_w4a16_weights

_INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
_HAS_TRITON = Backend.TRITON in registered_backends("moe_int4_w4a16")


def _device():
    if _INTERP:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    pytest.skip("no GPU and TRITON_INTERPRET!=1")


def _pin_single_config():
    """Pin the autotuner to one config (autotune is a no-op under the interpreter)."""
    from xkernels.ops.moe.triton.moe_int4_kernel import fused_moe_int4_kernel

    node = fused_moe_int4_kernel
    while node is not None and not hasattr(node, "configs"):
        node = getattr(node, "fn", None)
    if node is not None:
        node.configs = node.configs[:1]


def _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed):
    """fp32/bf16 grouped-MoE oracle reduced to ``[M, N]`` (kept in fp32)."""
    W = dequant_w4a16(packed, scale, group_size).to(A.dtype)
    M, topk = topk_ids.shape
    out = torch.zeros(M, W.shape[1], dtype=torch.float32, device=A.device)
    for m in range(M):
        for j in range(topk):
            e = int(topk_ids[m, j])
            contrib = A[m].float() @ W[e].float().T
            if mul_routed:
                contrib = topk_w[m, j].float() * contrib
            out[m] += contrib
    return out


def _inputs(M, E, N, K, top_k, dev, group_size=32):
    torch.manual_seed(0)
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    dtype = torch.float32 if _INTERP else torch.bfloat16
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return packed, scale, A, topk_ids, topk_w


def _params():
    if _INTERP:  # keep the slow interpreter tractable
        return [(1, 8, 64, 128, 2), (4, 8, 128, 256, 4), (2, 4, 96, 64, 2)]
    return [
        (1, 48, 256, 512, 8),  # decode-like, Kimi-ish E/top_k
        (4, 8, 512, 1024, 4),
        (16, 16, 1024, 2048, 4),
    ]


@pytest.mark.parametrize("M,E,N,K,top_k", _params())
@pytest.mark.parametrize("mul_routed", [False, True])
def test_triton_backend_matches_reference(M, E, N, K, top_k, mul_routed):
    if not _HAS_TRITON:
        pytest.skip("triton backend not registered (triton not installed)")
    dev = _device()
    group_size = 32
    _pin_single_config()
    packed, scale, A, topk_ids, topk_w = _inputs(M, E, N, K, top_k, dev, group_size)
    got = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=mul_routed, backend=Backend.TRITON,
    )
    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed)
    # Interpreter path runs fp32 but the group scales are still bf16, so the
    # different K-accumulation order vs the reference loop leaves a small bf16-
    # scale rounding gap; 3e-3 covers it. Hardware bf16 uses the issue-#1 2e-2.
    atol = rtol = 3e-3 if _INTERP else 2e-2
    torch.testing.assert_close(got.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("mul_routed", [False, True])
def test_reference_backend_matches_oracle(mul_routed):
    dev = _device()
    group_size = 32
    packed, scale, A, topk_ids, topk_w = _inputs(2, 4, 96, 64, 2, dev, group_size)
    got = fused_moe_int4_w4a16(
        A, packed, scale, topk_ids, topk_w,
        group_size=group_size, mul_routed_weight=mul_routed, backend=Backend.REFERENCE,
    )
    ref = _ref_grouped(A, packed, scale, topk_ids, topk_w, group_size, mul_routed)
    torch.testing.assert_close(got.float(), ref.float(), atol=3e-3, rtol=3e-3)


def test_dequant_roundtrip():
    """Packed weights from ``make_w4a16_weights`` dequant exactly (no kernel)."""
    dev = _device()
    packed, scale, w_ref = make_w4a16_weights(2, 64, 128, 32, device=dev, seed=3)
    torch.testing.assert_close(dequant_w4a16(packed, scale, 32), w_ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
