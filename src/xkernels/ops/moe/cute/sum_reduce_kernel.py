# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) fp32 kernel for ``moe_sum_reduce`` on sm_121.

Weighted reduction of top-k expert outputs:
``out[m, h] = routed_scaling_factor * sum_k w[m, k] * y[m, k, h]``.

Host-side dtype plumbing matches the reference + the passing triton card: ``y``
is upcast to fp32 on the host (bit-identical to the reference's ``y.float()``),
the CUTE device kernel is a PURE fp32 weighted reduction, and the host casts the
result back to ``y.dtype`` (round-to-nearest — identical to a bf16 store). The
ordering (fp32-multiply-then-cast) matches the triton card, well inside the op's
calibrated bf16 rtol 0.016.

Design — the reduction axis (``top_k``, =8 in the sweep) is TINY, so this is a
per-thread sum over k, NOT a block-wide reduction. One CTA per token row ``m``;
128 threads tile the hidden dim ``H`` (thread-stride -> coalesced). Each thread,
for every ``h`` it owns, sums ``top_k`` weighted ``y[m, k, h]`` values in fp32
(Kahan, for faithfulness vs torch's reduction order), multiplies by the scalar
``routed_scaling_factor``, and stores. No SMEM, no sync — the simplest correct
reduction. (If ``top_k`` grew large this would become a block-wide reduce like
dual_rmsnorm; at top_k=8 the per-thread sum is optimal.)
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

from ..._cute_backend.launch import _cached_handle, _require_cuda

_BLOCK_THREADS = 128

# Compile-once / launch-many handle cache, keyed by (M, H, dtype). The
# load-bearing rationale (why not @cute.jit __call__; why tensors-only launch)
# lives ONCE in ``ops/_cute_backend/launch.py`` — every CUTE card shares it.
_COMPILED_HANDLE_CACHE: dict[tuple[int, int, str], object] = {}


@cute.kernel
def _moe_sum_reduce_kernel(
    gY: Tensor,        # [M, top_k, H] row-major bf16 (READ NATIVE bf16 -> halve traffic)
    gW: Tensor,        # [M, top_k]    row-major fp32 (tiny; host-upcast)
    gOut: Tensor,      # [M, H]        row-major fp32
    M: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    H: cutlass.Constexpr,
    routed_scaling_factor: cutlass.Constexpr,
) -> None:
    """One CTA per token row m; threads tile H (thread-stride, coalesced).

    Perf: y is read NATIVELY as bf16 and promoted to fp32 on load (lossless —
    bf16->fp32 is exact), so the kernel moves 14.7MB not 29MB. This is the real
    lever for this memory-bound op (AI=2.3): the host no longer upcasts y to
    fp32 (which was both a separate launch AND doubled the read traffic).
    Kahan accumulation stays fp32; the on-load promotion is bit-identical to the
    reference's y.float()."""
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    m = bidx

    h = tidx
    while h < H:
        # Kahan sum over the (small) top_k axis: sum_k w[m,k] * y[m,k,h].
        acc = cutlass.Float32(0.0)
        c = cutlass.Float32(0.0)
        k = cutlass.Int32(0)
        while k < top_k:
            wv = gW[(m, k)]
            yv = gY[(m, k, h)]   # bf16 load -> promotes in the fp32 multiply below
            term = wv * yv
            y_ = term - c
            t_ = acc + y_
            c = (t_ - acc) - y_
            acc = t_
            k = k + 1
        gOut[(m, h)] = acc * cutlass.Float32(routed_scaling_factor)
        h = h + _BLOCK_THREADS


@cute.jit
def _moe_sum_reduce(
    Y: Tensor,
    W: Tensor,
    Out: Tensor,
    M: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    H: cutlass.Constexpr,
    routed_scaling_factor: cutlass.Constexpr,
) -> None:
    """Host JIT: one CTA per token row (M blocks of 128 threads)."""
    _moe_sum_reduce_kernel(
        Y, W, Out, M, top_k, H, routed_scaling_factor,
    ).launch(
        grid=[M, 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
)


def moe_sum_reduce_cute(
    y: torch.Tensor, w: torch.Tensor, routed_scaling_factor: float = 1.0
) -> torch.Tensor:
    """fp32 weighted top-k reduction via a JIT CUTE DSL kernel.

    ``y`` is upcast to fp32 on the host (bit-identical to the reference); the
    kernel runs pure fp32; the caller casts the result back to ``y.dtype``.
    Uses the compile-once / launch-many path keyed by ``(M, H)``.
    """
    _require_cuda(y)

    # Perf: read y NATIVELY as bf16 in the kernel (no host upcast) — halves the
    # memory traffic for this memory-bound op (14.7MB vs 29MB) and kills the
    # separate upcast launch. bf16->fp32 on load is lossless, so bit-identical
    # to the reference's y.float(). w is tiny ([M,top_k]); upcast stays on host.
    yc = y.contiguous()
    wf = w.to(torch.float32).contiguous()
    M, top_k, H = yc.shape
    out = torch.empty((M, H), device=yc.device, dtype=torch.float32)

    gY = from_dlpack(yc)   # bf16 — kernel promotes on load
    gW = from_dlpack(wf)   # fp32
    gOut = from_dlpack(out)

    key = (M, H, str(yc.dtype))
    handle = _cached_handle(
        _COMPILED_HANDLE_CACHE, key, _moe_sum_reduce,
        (gY, gW, gOut), (M, top_k, H, routed_scaling_factor),
    )
    handle(gY, gW, gOut)  # fast launch — tensors only (constexpr baked in)
    return out
