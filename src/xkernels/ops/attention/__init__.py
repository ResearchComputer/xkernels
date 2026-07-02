"""Attention kernels.

Ships ``mha_merge_state`` (issue #3): the numerically-stable online-softmax
merge of two attention partials by their log-sum-exp, used by chunked-prefill /
split-KV MLA on AMD MI300A.

Also ships ``dsa_indexer_logits`` (issue #27): the DeepSeek-V4 DSA indexer
weighted-ReLU MQA logits — the gfx942 forward path for the top-k KV selection
that drives V4 sparse attention (portable replacement for the NVIDIA-only
``deep_gemm.fp8_fp4_mqa_logits``). Pair with ``dsa_indexer_topk``.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .interface import (
    apply_rope,
    dsa_indexer_logits,
    dsa_indexer_topk,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
    mha_merge_state,
    sparse_mla_attention,
)

# Import the Triton backends for their registration side effect. Optional. Routed
# through the optional ``_triton_compat`` redirect so the kernel binds ``tokenspeed_triton``
# (not stock ``triton``) inside tokenspeed; see
# ``xkernels/_triton_compat.py``.
with backend_registration_guard(
    "dsa_indexer_logits",
    Backend.TRITON,
    source="xkernels.ops.attention.triton.dsa_indexer_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import dsa_indexer_kernel  # noqa: F401

with backend_registration_guard(
    "mha_merge_state",
    Backend.TRITON,
    source="xkernels.ops.attention.triton.merge_state_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import merge_state_kernel  # noqa: F401

# Import the CUTE DSL (native CUDA) backend for mha_merge_state (optional).
# NVIDIA-only; gated on `nvidia-cutlass-dsl` (the `cute` extra).
with backend_registration_guard(
    "mha_merge_state", Backend.CUDA, source="xkernels.ops.attention.cute.entry"
):  # pragma: no cover - requires nvidia-cutlass-dsl + NVIDIA GPU
    from .cute import entry  # noqa: F401  (registers CUDA: CUTE fp32 path)

with backend_registration_guard(
    "sparse_mla_attention",
    Backend.TRITON,
    source="xkernels.ops.attention.triton.sparse_mla_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import sparse_mla_kernel  # noqa: F401

# NOTE: ``apply_rope`` (issue #68) has a DSL-emitted triton card, but its device
# kernel currently CRASHES with an illegal-memory-access (a true OOB in the
# multi-dim gather/slice lowering, ``_TritonGenMultiDim`` -- confirmed by
# compute-sanitizer; see meta/docs/wiki/04-gotchas.md §14). The triton backend
# is therefore deliberately NOT registered here: a runtime illegal-memory-access
# is not caught by dispatch's registration-failure fallback and would poison
# the CUDA context, making ``xkernels.apply_rope(...)`` crash by default. The
# public ``apply_rope`` dispatches to REFERENCE (the DSL auto-reference, which
# verifies bit-exact on GPU) until the codegen bug is fixed -- then the one-line
# ``register_dsl`` module + this import is the only addition needed (wiki §13).

__all__ = [
    "mha_merge_state",
    "dsa_indexer_logits",
    "dsa_indexer_topk",
    "sparse_mla_attention",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
    "apply_rope",
]
