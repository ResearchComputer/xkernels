"""xkernels — customized compute kernels across vendors and kernel types."""

from .ops.attention import (
    dsa_indexer_logits,
    dsa_indexer_topk,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
    mha_merge_state,
    sparse_mla_attention,
)
from .ops.comm import (
    build_topology_groups,
    flat_all_reduce,
    hierarchical_all_reduce,
    residual_rmsnorm,
)
from .ops.ffn import fused_ffn
from .ops.gather import mxfp4_paged_gather
from .ops.moe import fused_moe_int4_w4a16, moe_align_block_size, moe_sum_reduce
from .ops.norm import dual_rmsnorm

__version__ = "0.0.1"
__all__ = [
    "fused_ffn",
    "fused_moe_int4_w4a16",
    "moe_align_block_size",
    "moe_sum_reduce",
    "mxfp4_paged_gather",
    "mha_merge_state",
    "dsa_indexer_logits",
    "dsa_indexer_topk",
    "sparse_mla_attention",
    "flash_mla_sparse_fwd",
    "flash_mla_with_kvcache",
    "get_mla_metadata",
    "dual_rmsnorm",
    "build_topology_groups",
    "flat_all_reduce",
    "hierarchical_all_reduce",
    "residual_rmsnorm",
    "__version__",
]
