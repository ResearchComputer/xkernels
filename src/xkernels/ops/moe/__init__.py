"""Mixture-of-Experts kernels.

Ships the INT4 W4A16 grouped fused-MoE GEMM (issue #1), the MXFP4 grouped
fused-MoE GEMM for DeepSeek-V4 routed experts (issue #43), the weighted top-k
reduction (issue #5), and the block-align dispatch builder (issue #4). Each
public op dispatches across a pure-torch reference (default on CPU / no Triton)
and, where available, an autotuned Triton backend.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .align import moe_align_block_size
from .interface import fused_moe_int4_w4a16
from .mxfp4 import dequant_mxfp4_weight, make_mxfp4_moe_weights
from .mxfp4_interface import fused_moe_mxfp4
from .sum_reduce import moe_sum_reduce
from .topk_softmax import topk_softmax
from .w4a16 import (
    dequant_w4a16,
    make_w4a16_weights,
    moe_align_block_size_ep,
    moe_align_block_size_ref,
)
from .workspace import (
    MoeAlignWorkspace,
    MoeInt4Workspace,
    MoeMxfp4Workspace,
)

# Import Triton backends for their registration side effects. Optional — guard
# each so the package imports without Triton installed. Routed through the
# optional ``_triton_compat`` redirect so the kernels bind ``tokenspeed_triton`` (not
# stock ``triton``) inside tokenspeed; see ``xkernels/_triton_compat.py``.
# This matters for ``moe_int4_kernel`` in particular: its ``tl.dot`` asserts that
# both operands share the same dtype *object*, which fails across packages.
with backend_registration_guard(
    "moe_int4_w4a16", Backend.TRITON, source="xkernels.ops.moe.triton.moe_int4_kernel"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import moe_int4_kernel  # noqa: F401

with backend_registration_guard(
    "moe_mxfp4", Backend.TRITON, source="xkernels.ops.moe.triton.moe_mxfp4_kernel"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import moe_mxfp4_kernel  # noqa: F401

with backend_registration_guard(
    "moe_sum_reduce", Backend.TRITON, source="xkernels.ops.moe.triton.sum_reduce_kernel"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import sum_reduce_kernel  # noqa: F401

# Fused MoE gating (softmax + top-k + optional renorm) Triton backend (issue #70).
with backend_registration_guard(
    "topk_softmax", Backend.TRITON, source="xkernels.ops.moe.triton.topk_softmax_kernel"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import topk_softmax_kernel  # noqa: F401

# Import the CUTE DSL (native CUDA) backend for moe_sum_reduce (optional).
# NVIDIA-only; gated on `nvidia-cutlass-dsl` (the `cute` extra).
with backend_registration_guard(
    "moe_sum_reduce", Backend.CUDA, source="xkernels.ops.moe.cute.entry"
):  # pragma: no cover - requires nvidia-cutlass-dsl + NVIDIA GPU
    from .cute import entry  # noqa: F401  (registers CUDA: CUTE fp32 path)

with backend_registration_guard(
    "moe_align_block_size", Backend.TRITON, source="xkernels.ops.moe.triton.align_kernel"
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import align_kernel  # noqa: F401

__all__ = [
    "fused_moe_int4_w4a16",
    "fused_moe_mxfp4",
    "moe_align_block_size",
    "moe_sum_reduce",
    "topk_softmax",
    "dequant_w4a16",
    "make_w4a16_weights",
    "dequant_mxfp4_weight",
    "make_mxfp4_moe_weights",
    "moe_align_block_size_ref",
    "moe_align_block_size_ep",
    "MoeAlignWorkspace",
    "MoeInt4Workspace",
    "MoeMxfp4Workspace",
]
