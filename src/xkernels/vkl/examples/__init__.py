# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL-authored examples — the artifacts the Phase 1 pipeline proves with.

Importing this subpackage registers each example's ``@kernel`` body as an
auto-reference (``xkernels.vkl.auto``) so the emitted ``numerics.reference``
path resolves. These are the inputs to ``tests/test_vkl_*.py``.

Access a kernel's spec via ``examples.<name>_spec`` (the function and its module
share a name, so we re-export the spec to dodge the ambiguity).
"""
from __future__ import annotations

# The KernelSpec for each example, re-exported for tests.
from ..graph import graph_of
from ..surface import spec_of
from . import activation as _activation_mod  # noqa: F401  (side-effect: register auto-ref)
from . import dual_rmsnorm as _dual_rmsnorm_mod  # noqa: F401  (side-effect: register auto-ref)
from . import gemm_bf16 as _gemm_bf16_mod  # noqa: F401  (side-effect: register auto-ref)
from . import gemm_chain as _gemm_chain_mod  # noqa: F401  (side-effect: attach @graph)
from . import (
    paged_kv_gather as _paged_kv_gather_mod,  # noqa: F401  (side-effect: register auto-ref)
)
from . import quant_fp8 as _quant_fp8_mod  # noqa: F401  (side-effect: register auto-ref)
from . import rmsnorm as _rmsnorm_mod  # noqa: F401  (side-effect: register auto-ref)
from . import rope as _rope_mod  # noqa: F401  (side-effect: register auto-ref)
from . import softmax as _softmax_mod  # noqa: F401  (side-effect: register auto-ref)
from .activation import gelu_and_mul, packed_gelu_and_mul, packed_silu_and_mul, silu_and_mul
from .dual_rmsnorm import dual_rmsnorm
from .gemm_bf16 import gemm_bf16
from .gemm_chain import gemm_chain
from .paged_kv_gather import paged_kv_gather
from .quant_fp8 import per_block_quant_fp8, per_token_group_quant_fp8
from .rmsnorm import rmsnorm
from .rope import apply_rope
from .softmax import rowwise_softmax, temperature_softmax

dual_rmsnorm_spec = spec_of(dual_rmsnorm)
gemm_bf16_spec = spec_of(gemm_bf16)
gemm_chain_spec = graph_of(gemm_chain)
rmsnorm_spec = spec_of(rmsnorm)
silu_and_mul_spec = spec_of(silu_and_mul)
gelu_and_mul_spec = spec_of(gelu_and_mul)
packed_silu_and_mul_spec = spec_of(packed_silu_and_mul)
packed_gelu_and_mul_spec = spec_of(packed_gelu_and_mul)
per_token_group_quant_fp8_spec = spec_of(per_token_group_quant_fp8)
per_block_quant_fp8_spec = spec_of(per_block_quant_fp8)
apply_rope_spec = spec_of(apply_rope)
paged_kv_gather_spec = spec_of(paged_kv_gather)
temperature_softmax_spec = spec_of(temperature_softmax)
rowwise_softmax_spec = spec_of(rowwise_softmax)

__all__ = [
    "dual_rmsnorm",
    "dual_rmsnorm_spec",
    "gemm_bf16",
    "gemm_bf16_spec",
    "gemm_chain",
    "gemm_chain_spec",
    "rmsnorm",
    "rmsnorm_spec",
    "silu_and_mul",
    "silu_and_mul_spec",
    "gelu_and_mul",
    "gelu_and_mul_spec",
    "packed_silu_and_mul",
    "packed_silu_and_mul_spec",
    "packed_gelu_and_mul",
    "packed_gelu_and_mul_spec",
    "per_token_group_quant_fp8",
    "per_token_group_quant_fp8_spec",
    "per_block_quant_fp8",
    "per_block_quant_fp8_spec",
    "apply_rope",
    "apply_rope_spec",
    "paged_kv_gather",
    "paged_kv_gather_spec",
    "temperature_softmax",
    "temperature_softmax_spec",
    "rowwise_softmax",
    "rowwise_softmax_spec",
    "spec_of",
]
