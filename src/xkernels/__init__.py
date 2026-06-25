"""xkernels — customized compute kernels across vendors and kernel types."""

from . import registry
from ._dispatch import backend_diagnostics
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
from .ops.gemm import (
    mm_fp8_blockscale,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    preferred_fp8_dtype,
)
from .ops.mhc import hc_prenorm_gemm, mhc_post, mhc_pre, tf32_hc_prenorm_gemm
from .ops.moe import (
    fused_moe_int4_w4a16,
    fused_moe_mxfp4,
    moe_align_block_size,
    moe_sum_reduce,
)
from .ops.norm import dual_rmsnorm

# --- agent-native surfaces (docs/library.md) ----------------------------------
# These are lazily-evaluating; importing the package does not parse the registry.
from .retrieval import find_impl
from .verify import verify, verify_parity

__version__ = "0.0.1"
__all__ = [
    "fused_ffn",
    "fused_moe_int4_w4a16",
    "fused_moe_mxfp4",
    "moe_align_block_size",
    "moe_sum_reduce",
    "mxfp4_paged_gather",
    "mm_fp8_blockscale",
    "per_token_group_quant_fp8",
    "per_block_quant_fp8",
    "preferred_fp8_dtype",
    "hc_prenorm_gemm",
    "tf32_hc_prenorm_gemm",
    "mhc_pre",
    "mhc_post",
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
    "backend_diagnostics",
    "__version__",
    # agent-native surfaces
    "find_impl",
    "verify",
    "verify_parity",
    "registry",
]
