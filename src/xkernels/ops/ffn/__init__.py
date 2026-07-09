"""Fused FFN kernels."""
from ..._backends import Backend, detect_vendor
from ..._dispatch import backend_registration_guard
from .interface import fused_ffn, fused_xielu_ffn

# Import backend modules for their registration side effects. Triton/CUDA are
# optional — guard so the package imports on any machine. The Triton import is
# routed through the optional ``_triton_compat`` redirect so the kernel binds
# ``tokenspeed_triton`` (not stock ``triton``) inside tokenspeed; see
# ``xkernels/_triton_compat.py``.
with backend_registration_guard(
    "ffn", Backend.TRITON, source="xkernels.ops.ffn.triton.ffn_kernel"
):  # pragma: no cover - hardware dependent
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import ffn_kernel  # noqa: F401

# Triton backend for the Apertus non-gated xIELU FFN (issue #80). Reuses the
# existing hand-written xIELU kernel (xkernels.ops.activation.triton) so it too
# routes through triton_import_ctx; the kernel module imports ``xielu_triton``
# which transitively imports triton at top level.
with backend_registration_guard(
    "xielu_ffn", Backend.TRITON, source="xkernels.ops.ffn.triton.xielu_ffn_kernel"
):  # pragma: no cover - hardware dependent
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import xielu_ffn_kernel  # noqa: F401

_cuda_backend = Backend.HIP if detect_vendor() == "amd" else Backend.CUDA
with backend_registration_guard(
    "ffn", _cuda_backend, source="xkernels.ops.ffn.cuda"
):  # pragma: no cover - requires compiled extension
    from . import cuda  # noqa: F401

__all__ = ["fused_ffn", "fused_xielu_ffn"]
