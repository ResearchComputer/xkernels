# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-generated Triton backend for plain single-tensor ``rmsnorm`` (issue #66).

One ``@kernel`` source (``xkernels.vkl.examples.rmsnorm``) lowers to a generated
row-wise Triton RMSNorm kernel via ``register_dsl``. The contract (Op Spec +
reference + Impl Cards) is emitted from that SAME source, so this module adds NO
new math -- it only binds the generated kernel to the dispatch registry under the
``TRITON`` backend, making it reachable as ``dispatch("rmsnorm",
backend="triton")`` and verifying on the ``rmsnorm.triton@1.0.0`` card.

This is the wiring that was missing when #66 first landed: the card existed and
verified via the standalone ``scripts/ds5_rmsnorm_gpu_gate.py`` (which calls
``register_dsl`` itself), but ``import xkernels`` did NOT register the triton
backend, so ``verify("rmsnorm.triton@1.0.0")`` raised
``KeyError: backend 'TRITON' not registered``. Importing this module via
``ops.norm.__init__`` (the package import side effect) closes that gap -- the
same pattern ``ops.activation`` uses for the #67 gated activations.

The generated kernel is compiled lazily on first call (Triton recompiles per
dtype anyway), so importing this module is safe without a GPU and without
``triton`` installed: registration builds only the host launcher, and the card
honestly reports ``compiled=False`` on a CPU box.
"""
from __future__ import annotations

from ....vkl import register_dsl, spec_of
from ....vkl.examples import rmsnorm

# Bind the DSL-authored rmsnorm body to its generated Triton launcher.
# ``register_dsl`` also (re)asserts the seeded input generator + graph-node
# wiring; both are idempotent / first-writer-wins with the ``@kernel``
# decorator's own auto-wire.
register_dsl(spec_of(rmsnorm), backend="triton")
