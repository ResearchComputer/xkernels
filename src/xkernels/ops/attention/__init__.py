"""Attention kernels.

Ships ``mha_merge_state`` (issue #3): the numerically-stable online-softmax
merge of two attention partials by their log-sum-exp, used by chunked-prefill /
split-KV MLA on AMD MI300A.

Also ships ``dsa_indexer_logits`` (issue #27): the DeepSeek-V4 DSA indexer
weighted-ReLU MQA logits — the gfx942 forward path for the top-k KV selection
that drives V4 sparse attention (portable replacement for the NVIDIA-only
``deep_gemm.fp8_fp4_mqa_logits``). Pair with ``dsa_indexer_topk``.
"""
from .interface import (
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
try:  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import (  # noqa: F401
            dsa_indexer_kernel,
            merge_state_kernel,
            sparse_mla_kernel,
        )
except Exception:
    pass

__all__ = [
    "mha_merge_state",
    "dsa_indexer_logits",
    "dsa_indexer_topk",
    "sparse_mla_attention",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
]
