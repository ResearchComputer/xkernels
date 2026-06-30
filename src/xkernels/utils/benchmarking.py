"""Timing helpers. Uses Triton's do_bench when available, else CUDA events."""
from __future__ import annotations

from collections.abc import Callable

import torch


def benchmark(fn: Callable[[], object], warmup: int = 10, iters: int = 50) -> float:
    """Return median wall-clock milliseconds per call of `fn`.

    Prefers Triton's ``do_bench`` (more robust warmup/rep + median over repeats)
    when available, falling back to CUDA events, then to wall clock. An import
    failure of ``do_bench`` is swallowed (it's an optional dep); a *measurement-
    time* error from ``do_bench`` is NOT — a swallowed OOM/crash inside the timing
    loop would silently mask a real failure behind a less-accurate fallback, which
    is worse than no measurement in a perf harness.
    """
    try:
        from triton.testing import do_bench
    except ImportError:
        do_bench = None

    if do_bench is not None:
        return float(do_bench(fn, warmup=warmup, rep=iters))

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
