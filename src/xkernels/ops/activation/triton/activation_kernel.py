# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-generated Triton backends for the gated activations (issue #67).

One ``@kernel`` source (``xkernels.vkl.examples.activation``) lowers to a
generated flat-1D elementwise Triton kernel per op via ``register_dsl``. The
contract (Op Spec + reference + Impl Cards) is emitted from that SAME source, so
this module adds NO new math — it only binds the generated kernels to the
dispatch registry under the ``TRITON`` backend, making them reachable as
``dispatch("silu_and_mul", backend="triton")`` and verifying on the
``*.triton@1.0.0`` cards.

The generated kernel is compiled lazily on first call (Triton recompiles per
dtype anyway), so importing this module is safe without a GPU and without
``triton`` installed: registration builds only the host launcher, and the card
honestly reports ``compiled=False`` on a CPU box (the launch fails for want of a
driver — see ``test_triton_card_honestly_uncompiled_without_gpu``).
"""
from __future__ import annotations

from ....vkl import register_dsl, spec_of
from ....vkl.examples import activation as _act

# Bind each DSL-authored gated activation to its generated Triton launcher.
# ``register_dsl`` also (re)asserts the seeded input generator + graph-node
# wiring; both are idempotent / first-writer-wins with the ``@kernel`` decorator's
# own auto-wire.
for _body in (
    _act.silu_and_mul,
    _act.gelu_and_mul,
    _act.packed_silu_and_mul,
    _act.packed_gelu_and_mul,
):
    register_dsl(spec_of(_body), backend="triton")

del _body
