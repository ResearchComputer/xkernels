# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Pure-torch reference for the DeepSeek-V4 fp8 block-scale dense GEMM (issue #38)
— numerical oracle and default (CPU / no-Triton) backend on gfx942.

V4-Flash stores the MLA (``q_a``/``kv_a``/``q_b``/``kv_b``), ``gate`` and
shared-expert projections as **fp8 block-scale** weights. The forward needs

    out[M, N] = (A_deq) @ (B_deq).T

where ``A`` is the activation ``[M, K]`` fp8 e4m3 quantized in **1×block** groups
along K (per-token-group scale ``A_scales [M, ceil(K/block)]``) and ``B`` is the
weight ``[N, K]`` fp8 e4m3 quantized in **block×block** tiles (per-block scale
``B_scales [ceil(N/block), ceil(K/block)]``). This is the standard DeepSeek
block-scale layout (``block=128``).

Upstream's portable path is ``torch_mm_fp8_blockscale``:
``(A.float()*A_scales) @ (B.float()*B_scales).T`` (``numerics/reference/gemm.py``).
It is numerically correct but materializes both operands in fp32 and runs a dense
fp32 matmul (no MFMA), so it dominates serve latency on the MLA hot path. This
module is the faithful oracle; the Triton backend
(``triton/mm_fp8_blockscale_kernel.py``) is the performant gfx942 replacement.

All math is fp32 (the parity target is this dequant-then-matmul reference, not
NVIDIA bit-equality).
"""

from __future__ import annotations

import warnings

import torch

from ..._backends import Backend
from ..._dispatch import register

__all__ = [
    "mm_fp8_blockscale_ref",
    "per_token_group_quant_fp8",
    "per_block_quant_fp8",
    "FP8_BLOCK",
]

#: Default block (group) size along every quantized axis (DeepSeek convention).
FP8_BLOCK = 128

#: e4m3 finite max (``torch.float8_e4m3fn``), used to scale into representable range.
_FP8_MAX = 448.0


def _dequant_a(a_fp8: torch.Tensor, a_scales: torch.Tensor, block: int) -> torch.Tensor:
    """Dequantize per-token-group fp8 ``A [M, K]`` to fp32 using ``A_scales [M, kt]``."""
    _M, K = a_fp8.shape
    out = a_fp8.to(torch.float32)
    # Expand each column-group scale across its (up to ``block``) columns.
    scales = a_scales.to(torch.float32).repeat_interleave(block, dim=1)[:, :K]
    return out * scales


def _dequant_b(b_fp8: torch.Tensor, b_scales: torch.Tensor, block: int) -> torch.Tensor:
    """Dequantize per-block fp8 ``B [N, K]`` to fp32 using ``B_scales [nt, kt]``."""
    N, K = b_fp8.shape
    out = b_fp8.to(torch.float32)
    scales = (
        b_scales.to(torch.float32)
        .repeat_interleave(block, dim=0)[:N]
        .repeat_interleave(block, dim=1)[:, :K]
    )
    return out * scales


def mm_fp8_blockscale_ref(
    a_fp8: torch.Tensor,
    a_scales: torch.Tensor,
    b_fp8: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    block: int = FP8_BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
    dot_bf16: bool = False,  # noqa: ARG001 - accepted for backend-signature parity
) -> torch.Tensor:
    """fp8 block-scale dense GEMM reference. See module docstring.

    Args:
        a_fp8: ``[M, K]`` activation, ``torch.float8_e4m3fn``.
        a_scales: ``[M, ceil(K/block)]`` fp32 per-token-group scales.
        b_fp8: ``[N, K]`` weight, ``torch.float8_e4m3fn`` (Linear orientation).
        b_scales: ``[ceil(N/block), ceil(K/block)]`` fp32 per-block scales.
        block: group/tile size along each quantized axis (default 128).
        out_dtype: output dtype (default bf16; fp32 also supported).
        dot_bf16: ignored here (the reference is always exact fp32); accepted so
            the reference and Triton backends share one signature.

    Returns:
        ``out [M, N]`` of ``out_dtype`` with ``out == (A_deq @ B_deq.T)``
        computed in fp32.
    """
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    if b_fp8.shape[1] != K:
        raise ValueError(f"b_fp8 must be [N, K] with K={K}, got {tuple(b_fp8.shape)}")
    kt = (K + block - 1) // block
    if tuple(a_scales.shape) != (M, kt):
        raise ValueError(
            f"a_scales must be [M, kt] = [{M}, {kt}], got {tuple(a_scales.shape)}"
        )
    nt = (N + block - 1) // block
    if tuple(b_scales.shape) != (nt, kt):
        raise ValueError(
            f"b_scales must be [{nt}, {kt}], got {tuple(b_scales.shape)}"
        )
    if dot_bf16:
        warnings.warn(
            "dot_bf16=True is ignored by the REFERENCE backend (always exact fp32).",
            stacklevel=2,
        )
    a_deq = _dequant_a(a_fp8, a_scales, block)
    b_deq = _dequant_b(b_fp8, b_scales, block)
    out = a_deq @ b_deq.t()
    return out.to(out_dtype)


def per_token_group_quant_fp8(
    x: torch.Tensor, *, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x [M, K]`` (fp32/bf16) to per-token-group fp8 e4m3.

    Each contiguous ``block``-length group along K shares one scale. Returns
    ``(x_fp8 [M, K] float8_e4m3fn, x_scales [M, ceil(K/block)] fp32)`` such that
    ``mm_fp8_blockscale_ref`` consumes them directly. The scale is the OCP-style
    ``amax/FP8_MAX`` per group — keeping the reference an exact dequant oracle.
    """
    M, K = x.shape
    kt = (K + block - 1) // block
    xf = x.to(torch.float32)
    x_fp8 = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    x_scales = torch.empty(M, kt, device=x.device, dtype=torch.float32)
    for j in range(kt):
        c0, c1 = j * block, min((j + 1) * block, K)
        g = xf[:, c0:c1]
        amax = g.abs().amax(dim=1).clamp_min(1e-12)
        scale = amax / _FP8_MAX
        q = (g / scale.unsqueeze(1)).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
        x_fp8[:, c0:c1] = q
        x_scales[:, j] = scale
    return x_fp8, x_scales


def per_block_quant_fp8(
    w: torch.Tensor, *, block: int = FP8_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight ``w [N, K]`` (fp32/bf16) to per-``block``×``block`` fp8 e4m3.

    Returns ``(w_fp8 [N, K] float8_e4m3fn,
    w_scales [ceil(N/block), ceil(K/block)] fp32)``.
    """
    N, K = w.shape
    nt = (N + block - 1) // block
    kt = (K + block - 1) // block
    wf = w.to(torch.float32)
    w_fp8 = torch.empty(N, K, device=w.device, dtype=torch.float8_e4m3fn)
    w_scales = torch.empty(nt, kt, device=w.device, dtype=torch.float32)
    for i in range(nt):
        r0, r1 = i * block, min((i + 1) * block, N)
        for j in range(kt):
            c0, c1 = j * block, min((j + 1) * block, K)
            g = wf[r0:r1, c0:c1]
            amax = g.abs().amax().clamp_min(1e-12)
            scale = amax / _FP8_MAX
            q = (g / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            w_fp8[r0:r1, c0:c1] = q
            w_scales[i, j] = scale
    return w_fp8, w_scales


register("mm_fp8_blockscale", Backend.REFERENCE)(mm_fp8_blockscale_ref)
