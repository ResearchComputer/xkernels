# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-generated Triton backend for ``apply_rope`` (issue #68).

One ``@kernel`` source (``xkernels.vkl.examples.rope``) lowers to a generated
multi-dim addressing Triton kernel via ``register_dsl``. The contract (Op Spec +
reference + Impl Cards) is emitted from that SAME source, so this module adds NO
new math -- it only binds the generated kernel to the dispatch registry under the
``TRITON`` backend, making it reachable as ``dispatch("apply_rope",
backend="triton")`` and verifying on the ``apply_rope.triton@1.0.0`` card.

This is the wiring that was missing: the card existed, but ``import xkernels``
did NOT register the triton backend, so ``verify("apply_rope.triton@1.0.0")``
raised ``KeyError: backend 'TRITON' not registered`` (the §13 wiring gap, the
same one #66 / #57 hit). Importing this module via ``ops.attention.__init__``
(the package import side effect) closes that gap.

The generated multi-dim kernel (``_TritonGenMultiDim``) is verified bit-exact-
within-bf16-tol on GB10 (sm_121): ``verify`` 5/5, ``verify_parity agree=True``
vs the torch reference. The earlier "device kernel OOBs" diagnosis (wiki §14)
was a *modulo-sign* bug in the per-axis offset (CUDA ``%`` follows C sign, so a
``Concat`` b-branch's negative coord ``c2 - len_a`` read before the buffer);
fixed by flooring the broadcast modulo. The generated kernel is compiled lazily
on first call, so importing this module is safe without a GPU and without
``triton`` installed (registration builds only the host launcher).
"""
from __future__ import annotations

from ....vkl import register_dsl, spec_of
from ....vkl.examples import apply_rope
from ....vkl.examples import apply_rope_gqa

# Bind the DSL-authored apply_rope body to its generated Triton launcher.
# ``register_dsl`` also (re)asserts the seeded input generator + graph-node
# wiring; both are idempotent / first-writer-wins with the ``@kernel``
# decorator's own auto-wire. (NB: import the ``apply_rope`` FUNCTION, not the
# ``rope`` module -- ``vkl.examples`` re-exports the function, not the module
# name, so ``spec_of`` needs the decorated callable.)
register_dsl(spec_of(apply_rope), backend="triton")

# Bind the GQA-native variant (issue #104). Same rotate-half body, distinct
# head symbols (Hq query / Hk key, ``Hq % Hk == 0``), so mini-sglang's adapter
# rotates query + key in ONE launch instead of two. The multi-dim codegen
# (``_TritonGenMultiDim`` / ``_launch_multidim``) was extended (#104) so each
# ``Store`` gets its OWN coord decomposition (from its own output shape) and
# its OWN ``offs < numel_<out>`` mask, over a grid sized by ``max(numel)`` --
# that is what makes different-sized outputs (``Hq != Hk``) safe (no OOB on the
# smaller output). VERIFIED on GB10 (sm_121):
# ``verify("apply_rope_gqa.triton@1.0.0", arch="nvidia_sm121")`` 5/5
# (max_abs=1.5e-05), ``verify_parity`` agree (max_rel=0.0074 < 0.01). The
# launcher registers safely without a GPU (registration builds only the host
# launcher; the kernel is JIT-compiled on first call).
register_dsl(spec_of(apply_rope_gqa), backend="triton")
