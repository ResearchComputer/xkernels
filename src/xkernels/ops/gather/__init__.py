"""Gather kernels.

Ships ``mxfp4_paged_gather`` (issue #27 / DeepSeek-V4 DSA indexer): the gfx942
Triton replacement for the CUDA-only ``indexer_mxfp4_paged_gather`` — gather +
dequantize DSA-selected mxfp4 KV positions out of a paged (block-table) cache.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .interface import mxfp4_paged_gather
from .mxfp4 import dequant_mxfp4, make_mxfp4_kv

# Import the Triton backend for its registration side effect. Optional.
with backend_registration_guard(
    "mxfp4_paged_gather",
    Backend.TRITON,
    source="xkernels.ops.gather.triton.paged_gather_kernel",
):  # pragma: no cover - requires triton
    from .triton import paged_gather_kernel  # noqa: F401

__all__ = ["mxfp4_paged_gather", "dequant_mxfp4", "make_mxfp4_kv"]
