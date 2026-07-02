# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-generated Triton backends for the fp8 quantization helpers (issue #57).

Two ``@kernel`` sources (``xkernels.vkl.examples.quant_fp8``:
``per_token_group_quant_fp8`` / ``per_block_quant_fp8``) lower to generated
row-wise Triton kernels via ``register_dsl``. The contract (Op Spec + reference +
Impl Cards) is emitted from those SAME sources, so this module adds NO new math
-- it only binds the generated kernels to the dispatch registry under the
``TRITON`` backend, making them reachable as
``dispatch("per_token_group_quant_fp8", backend="triton")`` etc. and verifying
on the ``*.triton@1.0.0`` cards.

This closes the §13 gap (backends register by import side-effect; the DSL cards
landed but ``import xkernels`` registered only the hand ``mm_fp8_blockscale``
triton backend, so ``verify`` on the quant cards raised ``backend 'TRITON' not
registered``). Same pattern as ``ops.norm.triton.rmsnorm_kernel`` (#66).

NOTE on the public ``[M,K]`` helpers: ``ops/gemm/reference.py``'s
``per_token_group_quant_fp8`` / ``per_block_quant_fp8`` take the NATURAL
``[M,K]`` / ``[N,K]`` shape (with ``block=`` + ``fp8_dtype="auto"`` + non-multiple-
of-block tail handling) and remain the high-level convenience API. The DSL cards
operate on the GROUPED full-tile view ``[G, B]`` (``G = M*K//block``, ``B =
block``; FP8_MAX=448 = float8_e4m3fn only -- the fnuz/240 AMD encoding is an
arch override). Routing the ``[M,K]`` helpers through this triton backend
therefore needs reshape glue + a dtype/path guard (full-tile, e4m3fn only) and
is tracked as the ergonomic follow-up; the kernels themselves are real,
verified, and reachable at the grouped-view level via ``dispatch(...)``.

The generated kernels are compiled lazily on first call, so importing this
module is safe without a GPU and without ``triton`` installed.
"""
from __future__ import annotations

from ....vkl import register_dsl, spec_of
from ....vkl.examples import quant_fp8

# Bind each DSL-authored quant body to its generated Triton launcher.
# ``register_dsl`` also (re)asserts the seeded input generator + graph-node
# wiring; both are idempotent / first-writer-wins with the ``@kernel``
# decorator's own auto-wire.
for _body in (
    quant_fp8.per_token_group_quant_fp8,
    quant_fp8.per_block_quant_fp8,
):
    register_dsl(spec_of(_body), backend="triton")

del _body
