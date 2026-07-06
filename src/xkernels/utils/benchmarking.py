# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Timing helpers. Uses Triton's do_bench when available, else CUDA events.

Two surfaces:
  * :func:`benchmark`         -- the legacy single-float API (median ms per call).
  * :func:`benchmark_repeat`  -- runs do_bench N times and returns median + spread
                                 (p10/p90), so run-to-run variance is VISIBLE.

The repeat surface exists because single-shot ``do_bench`` for sub-100 us kernels
on ROCm is high-variance across processes (dynamic clock scaling / first-dispatch
residency): the same kernel lands anywhere in a ~5x window (issue #89). A single
median hides that; ``benchmark_repeat`` makes it a number callers can flag.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

import torch

__all__ = ["benchmark", "benchmark_repeat"]


def _do_bench_median(fn: Callable[[], object], warmup: int, iters: int) -> float | None:
    """One ``do_bench`` invocation -> median ms, or ``None`` if do_bench is
    unavailable / its infra failed (environmental, not a kernel bug)."""
    try:
        from triton.testing import do_bench
    except ImportError:
        return None
    try:
        return float(do_bench(fn, warmup=warmup, rep=iters))
    except Exception:
        # do_bench INFRA failed (triton driver build, etc.) -- environmental.
        return None


def _cuda_events_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Fallback: CUDA events over ``iters`` calls. Runs ``fn`` directly so a real
    kernel error propagates (not masked by the timing path)."""
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


def _wall_ms(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """Last-resort CPU wall clock."""
    import time

    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def _one_shot(fn: Callable[[], object], warmup: int, iters: int) -> float:
    """One timing sample via the best available path (do_bench > CUDA events > wall).

    ``fn`` errors propagate on the fallback paths (do_bench swallows them as infra
    noise, so a ``None`` return routes to the honest fallback that re-runs ``fn``).
    """
    ms = _do_bench_median(fn, warmup, iters)
    if ms is not None:
        return ms
    if torch.cuda.is_available():
        return _cuda_events_ms(fn, warmup, iters)
    return _wall_ms(fn, warmup, iters)


def benchmark(fn: Callable[[], object], warmup: int = 25, iters: int = 100) -> float:
    """Return median wall-clock milliseconds per call of ``fn``.

    Prefers Triton's ``do_bench`` (robust warmup + median over repeats) when
    available, falling back to CUDA events, then to wall clock. Defaults raised to
    ``warmup=25, iters=100`` (was 10/50) to better stabilize sub-100 us kernels on
    ROCm where short warmup left the clock cold (issue #89). For the spread-aware
    surface (median + p10/p90 across N invocations) use :func:`benchmark_repeat`.
    """
    return _one_shot(fn, warmup, iters)


def benchmark_repeat(
    fn: Callable[[], object],
    *,
    warmup: int = 25,
    iters: int = 100,
    n_repeat: int = 5,
) -> dict:
    """Time ``fn`` ``n_repeat`` times (each a full ``do_bench``) and return the
    steady-state floor plus the run-to-run spread.

    Single-shot ``do_bench`` is high-variance for sub-100 us kernels on ROCm
    (issue #89: ~5.7x across processes from clock-state / first-dispatch effects).
    Outliers are always UPWARD (clock throttling, OS interference), so the
    reproducible number is the floor (min) -- the warmest, least-interfered run --
    which is the GPU-benchmarking standard for kernel capability. ``ms = min``;
    ``median`` / ``p90`` / ``max`` are exposed so a noisy point (wide ``p90/min``)
    is machine-detectable instead of trusted as a single number.

    Returns a dict::

        {ms, median, p10, p90, min, max, spread_ratio, n_repeat}

    where ``ms`` is the min (steady-state floor) and ``spread_ratio =
    max / min`` (1.0 = perfectly stable). ``fn`` errors propagate (the fallback
    paths run ``fn`` directly).
    """
    samples = [_one_shot(fn, warmup, iters) for _ in range(n_repeat)]
    samples_sorted = sorted(samples)
    floor = samples_sorted[0]
    median = statistics.median(samples_sorted)
    # percentile indices over the sorted samples (nearest-rank, in-bounds).
    def _pct(q: float) -> float:
        if len(samples_sorted) == 1:
            return samples_sorted[0]
        idx = max(0, min(len(samples_sorted) - 1, round(q * (len(samples_sorted) - 1))))
        return samples_sorted[idx]

    p10, p90 = _pct(0.10), _pct(0.90)
    return {
        "ms": round(floor, 6),
        "median": round(median, 6),
        "p10": round(p10, 6),
        "p90": round(p90, 6),
        "min": round(samples_sorted[0], 6),
        "max": round(samples_sorted[-1], 6),
        "spread_ratio": round(p90 / floor, 4) if floor > 0 else float("inf"),
        "n_repeat": n_repeat,
    }
