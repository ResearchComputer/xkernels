"""Timing helpers. Uses Triton's do_bench when available, else CUDA events."""
from __future__ import annotations

from collections.abc import Callable

import torch


def benchmark(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> float:
    """Return median wall-clock milliseconds per call of `fn`.

    Prefers Triton's ``do_bench`` (more robust warmup/rep + median over repeats)
    when available, falling back to CUDA events, then to wall clock.

    Two failure modes are distinguished:
      * ``do_bench`` infra failure (e.g. triton's driver-extension build breaks
        on a box missing ``python3-dev``): swallowed — it's environmental, not a
        kernel bug, and the CUDA-events fallback times the same ``fn`` honestly.
      * ``fn()`` itself failing: NOT swallowed. The fallback paths run ``fn()``
        directly and propagate its errors, so a real OOM/crash surfaces instead
        of being masked by a silent fall-through. (The previous ``except Exception``
        around the whole block swallowed both, which masked ``fn()`` failures.)
    """
    try:
        from triton.testing import do_bench
    except ImportError:
        do_bench = None

    if do_bench is not None:
        try:
            return float(do_bench(fn, warmup=warmup, rep=iters))
        except Exception:
            # do_bench INFRA failed (triton driver build, etc.) — environmental,
            # not a kernel bug. Fall through to the CUDA-events path, which runs
            # fn() directly and will propagate any REAL fn() error.
            pass

    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    import time

    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters
