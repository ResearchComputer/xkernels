# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CUTE DSL (`cutlass.cute`) fp32 kernel for ``mha_merge_state`` on sm_121.

Online-softmax merge of two attention partials by their log-sum-exp (natural-log
basis), matching the reference ``merge_state_ref``:

    m    = max(lse_a, lse_b)
    wa   = exp(lse_a - m);  wb = exp(lse_b - m);  denom = wa + wb
    out  = (wa * out_a + wb * out_b) / denom        # along D
    lse  = m + log(denom)

Host-side dtype plumbing matches the reference + the passing triton card:
``out_a``/``out_b`` are upcast to fp32 on the host (bit-identical to the
reference's ``out_a.float()``), the CUTE device kernel is PURE fp32, and the host
casts ``out`` back to ``out_a.dtype`` (``lse`` stays fp32). The reference uses
natural exp/log, so the kernel uses ``math.exp`` / ``math.log`` directly (the
triton card uses log2 + /LOG2E as an equivalent optimization; the result is the
same natural log).

Design — one CTA per (t,h) row (``n_rows = T*H``); 128 threads tile the head dim
``D`` (thread-stride -> coalesced, since out_a[t,h,:] is contiguous in D). The
per-row weights (wa, wb, denom, m) are SCALAR, so each thread computes them
locally (3 transcendentals, hidden behind the memory-bound D reads — no SMEM
broadcast needed). Thread 0 writes the scalar ``lse[row]``; every thread writes
its own ``out[row, d]`` (distinct d -> no race).
"""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math, nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

_BLOCK_THREADS = 128

# Compile-once / launch-many handle cache, keyed by the constexpr (n_rows, D).
# See mm_fp8_blockscale_kernel._COMPILED_HANDLE_CACHE for the full rationale.
_COMPILED_HANDLE_CACHE: "dict[tuple[int, int], object]" = {}


@cute.kernel
def _merge_state_kernel(
    gOutA: Tensor,   # [n_rows, D] fp32 (host-upcast from out_a.dtype)
    gLseA: Tensor,   # [n_rows]    fp32
    gOutB: Tensor,   # [n_rows, D] fp32
    gLseB: Tensor,   # [n_rows]    fp32
    gOut: Tensor,    # [n_rows, D] fp32
    gLse: Tensor,    # [n_rows]    fp32
    n_rows: cutlass.Constexpr,
    D: cutlass.Constexpr,
) -> None:
    """One CTA per row; threads tile D (thread-stride, coalesced)."""
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())
    row = bidx

    la = gLseA[(row,)]
    lb = gLseB[(row,)]
    # m = max(la, lb) via a scalar conditional (no intrinsic guess).
    m = la
    if lb > m:
        m = lb
    wa = math.exp(la - m)
    wb = math.exp(lb - m)
    denom = wa + wb
    inv = cutlass.Float32(1.0) / denom

    d = tidx
    while d < D:
        oa = gOutA[(row, d)]
        ob = gOutB[(row, d)]
        gOut[(row, d)] = (wa * oa + wb * ob) * inv
        d = d + _BLOCK_THREADS

    # Only thread 0 writes the scalar lse (avoid a write race).
    if tidx == 0:
        gLse[(row,)] = m + math.log(denom)


@cute.jit
def _merge_state(
    OutA: Tensor,
    LseA: Tensor,
    OutB: Tensor,
    LseB: Tensor,
    Out: Tensor,
    Lse: Tensor,
    n_rows: cutlass.Constexpr,
    D: cutlass.Constexpr,
) -> None:
    """Host JIT: one CTA per (t,h) row (n_rows blocks of 128 threads)."""
    _merge_state_kernel(
        OutA, LseA, OutB, LseB, Out, Lse, n_rows, D,
    ).launch(
        grid=[n_rows, 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def merge_state_cute(
    out_a: "torch.Tensor", lse_a: "torch.Tensor", out_b: "torch.Tensor", lse_b: "torch.Tensor"  # type: ignore[name-defined]
) -> "tuple[torch.Tensor, torch.Tensor]":  # type: ignore[name-defined]
    """Online-softmax merge via a JIT CUTE DSL kernel (pure fp32).

    ``out_a``/``out_b`` are upcast to fp32 on the host (bit-identical to the
    reference); the kernel runs pure fp32; ``out`` is cast back to ``out_a.dtype``
    by the caller (``lse`` stays fp32). Uses the compile-once / launch-many path
    keyed by ``(n_rows, D)``.
    """
    import torch

    if not getattr(out_a, "is_cuda", False):
        # GPU-only: verify_parity() hardcodes device='cpu'; raising here lets the
        # harness record CUDA as a caught backend error instead of segfaulting.
        raise RuntimeError(
            "CUTE DSL kernel requires CUDA tensors; got device='cpu'. "
            "verify_parity() hardcodes device='cpu' and cannot exercise a GPU-only card."
        )
    # Perf: read out_a/out_b NATIVELY as bf16 (no host upcast) — halves the
    # memory traffic for this memory-bound op (AI=1.3). The kernel's arithmetic
    # (fp32 weights wa/wb/inv * bf16 inputs oa/ob) promotes to fp32 on load, so
    # it's bit-identical to the reference's out_a.float(). lse stays fp32.
    oa_c = out_a.contiguous()
    ob_c = out_b.contiguous()
    laf = lse_a.to(torch.float32).contiguous()
    lbf = lse_b.to(torch.float32).contiguous()
    # Flatten [T, H, D] -> [n_rows, D] and [T, H] -> [n_rows].
    oaf2 = oa_c.view(-1, oa_c.shape[-1])
    obf2 = ob_c.view(-1, ob_c.shape[-1])
    laf1 = laf.view(-1)
    lbf1 = lbf.view(-1)
    n_rows = laf1.shape[0]
    D = oaf2.shape[1]
    out = torch.empty((n_rows, D), device=oa_c.device, dtype=torch.float32)
    lse = torch.empty_like(laf1)

    gOutA = from_dlpack(oaf2)   # bf16 — kernel promotes on load
    gLseA = from_dlpack(laf1)   # fp32
    gOutB = from_dlpack(obf2)   # bf16
    gLseB = from_dlpack(lbf1)   # fp32
    gOut = from_dlpack(out)
    gLse = from_dlpack(lse)

    key = (n_rows, D, str(oa_c.dtype))
    handle = _COMPILED_HANDLE_CACHE.get(key)
    if handle is None:
        _merge_state(gOutA, gLseA, gOutB, gLseB, gOut, gLse, n_rows, D)
        torch.cuda.synchronize()
        handle = cute.compile(_merge_state, gOutA, gLseA, gOutB, gLseB, gOut, gLse, n_rows, D)
        _COMPILED_HANDLE_CACHE[key] = handle

    # Fast launch — tensors only (constexpr baked in at compile; see cache note).
    handle(gOutA, gLseA, gOutB, gLseB, gOut, gLse)
    # Restore the original [T, H, D] / [T, H] shapes.
    return out.view_as(out_a), lse.view_as(lse_a)
