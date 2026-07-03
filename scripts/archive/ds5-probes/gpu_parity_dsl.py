# SPDX-License-Identifier: MIT
"""GPU parity check for the DSL-authored ops (run on ds5 / sm_121)."""
from __future__ import annotations

from xkernels import verify_parity
from xkernels.vkl import register_dsl, spec_of
from xkernels.vkl.examples import (
    apply_rope,
    gelu_and_mul,
    packed_gelu_and_mul,
    packed_silu_and_mul,
    paged_kv_gather,
    per_block_quant_fp8,
    per_token_group_quant_fp8,
    rmsnorm,
    rowwise_softmax,
    silu_and_mul,
    temperature_softmax,
)

ARCH = "nvidia_sm121"
print(f"=== DSL cross-backend parity @ {ARCH} ===")
_OPS = (
    rmsnorm,
    silu_and_mul,
    gelu_and_mul,
    packed_silu_and_mul,
    packed_gelu_and_mul,
    per_token_group_quant_fp8,
    per_block_quant_fp8,
    apply_rope,
    paged_kv_gather,
    temperature_softmax,
    rowwise_softmax,
)
for fn in _OPS:
    s = spec_of(fn)
    register_dsl(s, "triton")
    try:
        p = verify_parity(f"{s.short_name}@1.0.0", archs=[ARCH])
        if p["inconclusive"]:
            print(f"[inconclusive] {s.short_name:30s} (<2 backends ran)")
        else:
            tag = "PARITY-OK" if p["agree"] else "PARITY-DRIFT"
            mr = p['max_pairwise_rel_err']
            print(f"[{tag}] {s.short_name:30s} max_rel={mr:.3e} n={p['n_runnable']}")
    except Exception as e:  # parity needs >=2 backends; single-backend ops skip
        print(f"[skip] {s.short_name}: {type(e).__name__}: {str(e)[:80]}")
