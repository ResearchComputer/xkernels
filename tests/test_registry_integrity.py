"""Registry integrity guards -- machine-independent invariants the contract
depends on. Runs in the CPU CI step (no GPU, no ``.rcc`` local config).

Guards the VKL-layer registration invariant (issue #88): the ``.rcc/rccignore``
silent-amputation class -- every VKL op vanishing from ``managed_short_names()``
and therefore from the registration wiring -- is caught here regardless of
whether the generated specs live in ``registry/ops`` or ``dist/ops``, because the
assertion is on the *post-import registered state*, not on file presence. A fresh
CI clone has no ``.rcc/`` (it is gitignored laptop-only config), so the historical
amputation would not reproduce in CI by itself; this test pins the invariant so a
code change that breaks the registration path (or a future layout move that drops
the specs) fails CI instead of silently shipping an op set of zero.
"""
from __future__ import annotations

import xkernels  # noqa: F401  importing triggers the VKL registration wiring
from xkernels._dispatch import backend_diagnostics
from xkernels.vkl.artifacts import managed_short_names

# VKL ops whose generated Triton kernel is CPU-runnable (no MFMA / GPU-feature
# dependency), so they register TRITON on a CPU box. Asserting this subset -- not
# every managed op -- keeps the guard green in CPU CI while still proving the
# registration *path* is wired (gemm_bf16 / paged_kv_gather / rowwise_softmax /
# temperature_softmax legitimately omit TRITON until a GPU is present).
_CPU_RUNNABLE_VKL = ["rmsnorm", "silu_and_mul", "apply_rope", "gelu_and_mul"]


def test_managed_short_names_nonempty():
    """The VKL-managed layer is present (>=10 ops).

    The ``.rcc/rccignore`` amputation class (#88) makes ``managed_short_names()``
    return ``[]`` on the affected machine, silently unregistering every VKL op.
    This asserts that never regresses: at least 10 managed ops are discovered.
    """
    ops = managed_short_names()
    assert len(ops) >= 10, (
        f"managed_short_names() returned only {len(ops)} ops {ops!r}; "
        f"the VKL layer is missing/ignored (the .rccignore-amputation class, #88)"
    )


def test_cpu_runnable_vkl_ops_register_triton():
    """A stable CPU-runnable subset of VKL ops must register TRITON after import.

    Catches a registration-*path* break (not just missing specs): if the wiring
    that turns a managed spec into a registered TRITON backend stops firing, these
    ops drop to REFERENCE-only and the assertion fails.
    """
    diag = backend_diagnostics()
    missing = [
        op
        for op in _CPU_RUNNABLE_VKL
        if "triton" not in [b.lower() for b in diag.get(op, {}).get("registered", [])]
    ]
    assert not missing, (
        f"VKL ops missing TRITON backend after import (registration-path break, "
        f"the amputation class #88): {missing}; "
        f"diagnostics={{{', '.join(f'{o}: {diag.get(o)}' for o in missing)}}}"
    )
