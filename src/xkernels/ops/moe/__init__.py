"""Mixture-of-Experts kernels.

Currently ships the INT4 W4A16 grouped fused-MoE GEMM (issue #1). The public op
dispatches across a pure-torch reference (default on CPU / no Triton) and an
autotuned Triton backend (auto-selected on GPU).
"""
from .interface import fused_moe_int4_w4a16
from .w4a16 import dequant_w4a16, make_w4a16_weights, moe_align_block_size_ref

# Import the Triton backend for its registration side effect. Optional — guard
# so the package imports without Triton installed.
try:  # pragma: no cover - requires triton
    from .triton import moe_int4_kernel  # noqa: F401
except Exception:
    pass

__all__ = [
    "fused_moe_int4_w4a16",
    "dequant_w4a16",
    "make_w4a16_weights",
    "moe_align_block_size_ref",
]
