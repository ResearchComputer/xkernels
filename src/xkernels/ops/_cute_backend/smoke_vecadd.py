# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Minimal CUTE DSL kernel — vector add ``y = a*x + y`` — as the sm_121 reachability probe.

This is NOT an xkernels op (no Op Spec, no card). It exists to prove the one
load-bearing fact every CUTE card on ds5 depends on: that ``cutlass.cute`` can
**JIT-compile and run** a kernel on the Grace+Blackwell GB10 (sm_121) end-to-end,
through the torch interop path (``from_dlpack``).

Modeled on the canonical minimal pattern in
``cutlass.cute.testing._convert`` / ``_convert_kernel`` (read directly from the
installed package, not from memory): a ``@cute.jit`` host function that
partitions a 1-D tensor into (CTA, thread) tiles and calls a ``@cute.kernel``
device function, ending in ``.launch(grid=, block=)``.

Run on ds5::

    rcc --profile ds5 run -s 'cd /local/home/xiayao/xkernels && \
        export CUDA_HOME=/usr/local/cuda-13.0 && . .venv/bin/activate && \
        python -m xkernels.ops._cute_backend.smoke_vecadd'
"""
from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute import nvgpu
from cutlass.cute.algorithm import copy
from cutlass.cute.atom import make_copy_atom
from cutlass.cute.core import composition, get, rank, size, zipped_divide
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.tensor import make_identity_tensor, make_rmem_tensor
from cutlass.cute.typing import Float32, Tensor
from cutlass.cutlass_dsl import T
from cutlass._mlir.dialects import nvvm

# Elements each thread processes per kernel launch (a small, vectorizable tile).
_ELEM_PER_THREAD = 4
_BLOCK_THREADS = 128


@cute.kernel
def _vecadd_kernel(
    gX: Tensor,
    gY: Tensor,
    gOut: Tensor,
    cCoord: Tensor,
    tv_layout: cutlass.Layout,
    n: cutlass.Constexpr,
) -> None:
    """Device kernel: each thread loads ELEM_PER_THREAD x/y, adds, stores out.

    Tile layout mirrors ``_convert_kernel``: CTA tiles the leading mode, then a
    (thread, vec) TV layout maps lanes -> element addresses via ``composition``.
    """
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    bidx = nvvm.read_ptx_sreg_ctaid_x(T.i32())

    cta_coord = (None, bidx)
    ctaX = gX[cta_coord]
    ctaY = gY[cta_coord]
    ctaOut = gOut[cta_coord]
    ctaCCoord = cCoord[cta_coord]

    # (Thread, Vec) layout: 128 threads x ELEM_PER_THREAD vec.
    tidfrgX = composition(ctaX, tv_layout)
    tidfrgY = composition(ctaY, tv_layout)
    tidfrgOut = composition(ctaOut, tv_layout)
    tidfrgCCoord = composition(ctaCCoord, tv_layout)

    thr_coord = (tidx, None)
    thrX = tidfrgX[thr_coord]
    thrY = tidfrgY[thr_coord]
    thrOut = tidfrgOut[thr_coord]
    thrCCoord = tidfrgCCoord[thr_coord]

    # Predicate: stay in-bounds for the tail CTA.
    if get(thrCCoord[0], mode=[0]) < n:
        frgX = make_rmem_tensor(get(tv_layout, mode=[1]), gX.element_type)
        frgY = make_rmem_tensor(get(tv_layout, mode=[1]), gY.element_type)
        frgOut = make_rmem_tensor(get(tv_layout, mode=[1]), gOut.element_type)

        copy(make_copy_atom(nvgpu.CopyUniversalOp(), gX.element_type), thrX, frgX)
        copy(make_copy_atom(nvgpu.CopyUniversalOp(), gY.element_type), thrY, frgY)

        vx = frgX.load()
        vy = frgY.load()
        frgOut.store(vx + vy)

        copy(make_copy_atom(nvgpu.CopyUniversalOp(), gOut.element_type), frgOut, thrOut)


@cute.jit
def _vecadd(
    x: Tensor,
    y: Tensor,
    out: Tensor,
    n: cutlass.Constexpr,
) -> None:
    """Host JIT function: build the CTA tiler + TV layout and launch the kernel."""
    tv_layout = cute.make_layout((_BLOCK_THREADS, _ELEM_PER_THREAD),
                                 stride=(_ELEM_PER_THREAD, 1))

    cta_tile = [1] * rank(x.layout)
    cta_tile[0] = size(tv_layout)  # 128 * 4 = 512 elems per CTA on the leading mode

    idA = make_identity_tensor(x.shape)
    gX = zipped_divide(x, tuple(cta_tile))
    gY = zipped_divide(y, tuple(cta_tile))
    gOut = zipped_divide(out, tuple(cta_tile))
    cCoord = zipped_divide(idA, tuple(cta_tile))

    _vecadd_kernel(
        gX, gY, gOut, cCoord, tv_layout, n,
    ).launch(
        grid=[size(gX, mode=[1]), 1, 1],
        block=[_BLOCK_THREADS, 1, 1],
    )


def vecadd_cute(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute ``x + y`` on the GPU via a JIT-compiled CUTE DSL kernel.

    Inputs are contiguous 1-D float32 tensors on cuda. Returns a new tensor.
    Raises if the CUTE DSL cannot compile for the current device.
    """
    if x.dim() != 1 or y.dim() != 1:
        raise ValueError(f"vecadd_cute expects 1-D tensors, got {x.dim()}-D / {y.dim()}-D")
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("vecadd_cute expects contiguous tensors")
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    gx = from_dlpack(x)
    gy = from_dlpack(y)
    gout = from_dlpack(out)
    n = x.numel()

    _vecadd(gx, gy, gout, n)
    return out


def _self_check() -> None:
    """Tiny correctness check vs torch on the current GPU device."""
    assert torch.cuda.is_available(), "CUTE DSL smoke kernel needs a CUDA device"
    dev = torch.cuda.current_device()
    torch.manual_seed(0)
    # Exercise aligned + tail-CTA (non-multiple-of-tile) sizes to validate the
    # in-bounds predicate, not just the happy aligned path.
    sizes = [4096, 512, 513, 1000, 1, 2, 128 * 4 - 1]
    worst = 0.0
    for n in sizes:
        x = torch.randn(n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, device="cuda", dtype=torch.float32)
        got = vecadd_cute(x, y)
        ref = x + y
        err = (got - ref).abs().max().item()
        worst = max(worst, err)
        ok = torch.allclose(got, ref, atol=1e-5)
        print(f"  n={n:6d}: max_abs_err={err:.3e} pass={ok}")
        if not ok:
            raise SystemExit(f"SMOKE FAIL at n={n}: err={err}")
    print(f"device: {torch.cuda.get_device_name(dev)} "
          f"(cc {''.join(map(str, torch.cuda.get_device_capability(dev)))})")
    print(f"vecadd_cute vs torch over {len(sizes)} sizes: worst max_abs_err = {worst:.3e} — PASS")


if __name__ == "__main__":
    _self_check()
