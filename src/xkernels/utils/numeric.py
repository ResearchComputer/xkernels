# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Numerics helpers shared across references and tests.

The load-bearing one is :func:`no_tf32`: a context manager that forces a matmul /
reduction to run in true fp32 (or fp16/bf16) on CUDA, defeating ``torch``'s silent
TF32 / reduced-precision defaults. References are the high-precision source of
truth (library.md §10), so every reference that calls ``@`` / ``torch.matmul`` /
``torch.sum`` on CUDA fp32 operands MUST wrap the compute in ``with no_tf32():`` --
otherwise the oracle is TF32 on NVIDIA (10-bit mantissa) and true fp32 on AMD
(which has no TF32), making "correctness" arch-dependent (issue #86).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

__all__ = ["no_tf32"]


@contextmanager
def no_tf32():
    """Force CUDA matmul / reductions to true fp32 (no TF32) for the block.

    Saves and restores ``torch.backends.cuda.matmul.allow_tf32`` and
    ``torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`` so the
    reference's ``@`` / ``torch.matmul`` / ``.sum()`` computes in the operand's
    real dtype on every CUDA device. No-op on CPU and on AMD (the flags are
    NVIDIA TF32 levers; ROCm ignores them). Use around any reference compute
    that must be the high-precision oracle::

        with no_tf32():
            out = a_deq @ b_deq.t()
    """
    matmul = torch.backends.cuda.matmul
    prev_tf32 = matmul.allow_tf32
    prev_fp16 = matmul.allow_fp16_reduced_precision_reduction
    matmul.allow_tf32 = False
    matmul.allow_fp16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_tf32 = prev_tf32
        matmul.allow_fp16_reduced_precision_reduction = prev_fp16
