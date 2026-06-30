# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Shared launch helpers for every CUTE DSL (``cutlass.cute``) card.

Two facts are load-bearing for *all* CUTE kernels in this repo and were
re-discovered (re-debugged) on every card until they were factored out here:

1. **Compile-once / launch-many.** The ``@cute.jit`` ``__call__`` path rebuilds
   the MLIR execution engine on *every* call (~9 ms, ~99% idle GPU), while a
   ``cute.compile`` handle replays in ~40 us. The handle SPECIALIZES on the
   ``cutlass.Constexpr`` args at compile time, so it MUST be launched with ONLY
   the tensor args — re-passing the constexpr corrupts the TVM-FFI execution-args
   ABI and SEGFAULTS in the native launch. So every card caches one handle per
   constexpr key and always calls it with the tensor args alone.

2. **GPU-only.** The native launch segfaults on a host pointer. Calling these
   on CPU (e.g. ``verify_parity`` on a CPU box) must raise a clean
   ``RuntimeError`` rather than SIGSEGV so the harness records it as a caught
   backend error.

Every CUTE card's host function therefore looks like::

    _require_cuda(a)                                  # (2)
    ...
    key = (M, N, K, str(a.dtype))                      # the constexpr specialization
    handle = _cached_handle(_CACHE, key, _jit_fn,     # (1)
                            (gA, gB, gOut), (M, N, K))
    handle(gA, gB, gOut)                               # tensors ONLY
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _require_cuda(tensor: Any) -> None:
    """Raise ``RuntimeError`` if ``tensor`` is not on CUDA.

    A clean error (rather than a SIGSEGV in the native launch) lets the
    verification harness record the CUDA backend as a *caught* error instead of
    crashing the process. The message is intentionally free of harness internals
    so it reads correctly for a direct user too.
    """
    if not getattr(tensor, "is_cuda", False):
        dev = getattr(tensor, "device", "?")
        raise RuntimeError(
            f"CUTE DSL kernel requires CUDA tensors; got device={dev}. "
            "This backend is GPU-only (NVIDIA CUTE DSL)."
        )


def _cached_handle(
    cache: dict[Any, Any],
    key: Any,
    jit_fn: Callable[..., Any],
    tensor_args: tuple,
    constexpr_args: tuple,
) -> Any:
    """Return a reusable ``cute.compile`` handle for ``jit_fn``, keyed by ``key``.

    First call for ``key``: warm up via the ``@cute.jit`` path (preprocesses the
    AST + builds the MLIR module), ``synchronize``, then ``cute.compile`` into a
    reusable handle. Subsequent calls replay the cached handle (~40 us vs ~9 ms).

    The returned handle SPECIALIZES on ``constexpr_args`` at COMPILE time, so the
    caller MUST launch it with ONLY the tensor args (``handle(*tensor_args)``);
    re-passing the constexpr corrupts the TVM-FFI ABI and SEGFAULTS. The caller
    owns the cache dict (one per kernel) and is responsible for putting every
    constexpr dimension — and, where dtype changes read traffic (bf16-native
    read), the dtype string — into ``key``.

    Args:
        cache: per-kernel handle cache (``{key: handle}``).
        key: hashable specialization key (the constexpr dims [+ dtype]).
        jit_fn: the ``@cute.jit``-decorated host function.
        tensor_args: the runtime tensor arguments, in declaration order.
        constexpr_args: the ``cutlass.Constexpr`` arguments, in declaration order.
    """
    handle = cache.get(key)
    if handle is None:
        cute, torch = _import_cute()
        all_args = (*tensor_args, *constexpr_args)
        jit_fn(*all_args)          # warmup: preprocess AST + build MLIR module
        torch.cuda.synchronize()
        handle = cute.compile(jit_fn, *all_args)
        cache[key] = handle
    return handle


# Imported lazily so this module imports cleanly on a box without the CUTE DSL
# installed — the same graceful-degradation every CUTE card uses.
# ``cutlass.cute.compile`` and ``torch.cuda.synchronize`` are the only entry
# points the helper needs.
def _import_cute():
    import cutlass.cute as cute  # type: ignore[import-not-found]
    import torch

    return cute, torch
