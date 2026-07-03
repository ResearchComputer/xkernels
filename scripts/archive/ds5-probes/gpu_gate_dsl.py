# SPDX-License-Identifier: MIT
"""GPU gate driver for the DSL-authored ops (run on ds5 / sm_121).

Registers each DSL op's generated Triton kernel with the dispatch registry, then
runs ``verify`` on its triton card against the op's auto-reference on the GPU.

This closes the docs/brainstorm/04 Ex.1 loop on real hardware: one @kernel
source -> a registered, verified Triton kernel, with zero JSON hand-editing.
"""
from __future__ import annotations

import torch

from xkernels import verify
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


def _gate(name: str, fn) -> None:
    spec = spec_of(fn)
    register_dsl(spec, "triton")
    card = f"{spec.short_name}.triton@1.0.0"
    v = verify(card, arch=ARCH)
    ok = v["compiled"] and v["correctness"]["passed"]
    print(
        f"[{'PASS' if ok else 'FAIL'}] {card:42s} "
        f"n={v['correctness']['n_points']} "
        f"max_abs={v['correctness']['max_abs_err']:.3e} "
        f"det={v['determinism_check']}"
        + (f"  err={v['artifacts'].get('error', '')!r:.120s}" if not ok else "")
    )


print(f"=== DSL GPU gate @ {ARCH} (torch {torch.__version__}) ===")
_gate("rmsnorm", rmsnorm)
_gate("silu_and_mul", silu_and_mul)
_gate("gelu_and_mul", gelu_and_mul)
_gate("packed_silu_and_mul", packed_silu_and_mul)
_gate("packed_gelu_and_mul", packed_gelu_and_mul)
_gate("per_token_group_quant_fp8", per_token_group_quant_fp8)
_gate("per_block_quant_fp8", per_block_quant_fp8)
_gate("apply_rope", apply_rope)
_gate("paged_kv_gather", paged_kv_gather)
_gate("temperature_softmax", temperature_softmax)
_gate("rowwise_softmax", rowwise_softmax)
