"""Normalization kernels.

Ships the fused parallel dual RMSNorm (MLA ``q_a`` / ``kv_a`` latents, issue #2):
two independent RMSNorms over differently-sized feature dims in a single launch.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .interface import dual_rmsnorm, rmsnorm

# Import the Triton backend for its registration side effect. Optional.
# The import is routed through the optional ``_triton_compat`` redirect so the kernel
# binds ``tokenspeed_triton`` (not stock ``triton``) when running inside
# tokenspeed; see ``xkernels/_triton_compat.py``.
with backend_registration_guard(
    "dual_rmsnorm",
    Backend.TRITON,
    source="xkernels.ops.norm.triton.dual_rmsnorm_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import dual_rmsnorm_kernel  # noqa: F401

# Register the DSL-generated Triton backend for the standalone ``rmsnorm``
# (issue #66) for its dispatch side effect. Unlike the hand-written
# ``dual_rmsnorm_kernel`` above (which imports ``triton`` at top level, so it
# routes through ``triton_import_ctx``), the DSL path builds only a lazy host
# launcher at import time -- no triton import happens until the kernel is
# actually called -- so no import-context redirect is needed here (same pattern
# as ``ops.activation`` for the #67 gated activations).
with backend_registration_guard(
    "rmsnorm",
    Backend.TRITON,
    source="xkernels.ops.norm.triton.rmsnorm_kernel",
):  # pragma: no cover - hardware dependent
    from .triton import rmsnorm_kernel  # noqa: F401

# Import the CUTE DSL (native CUDA) backend for its registration side effect
# (optional). NVIDIA-only; gated on `nvidia-cutlass-dsl` (the `cute` extra). On a
# box without the DSL (CI, AMD) the guard records the failure and the op is
# still served by the triton/reference card.
with backend_registration_guard(
    "dual_rmsnorm", Backend.CUDA, source="xkernels.ops.norm.cute.entry"
):  # pragma: no cover - requires nvidia-cutlass-dsl + NVIDIA GPU
    from .cute import entry  # noqa: F401  (registers CUDA: CUTE fp32 path)

__all__ = ["dual_rmsnorm", "rmsnorm"]
