# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""PR #61 benchmark: auto-selected single-token (Tq=1) sparse-MLA decode config.

Measures, on one gfx942 (MI300A) GPU, the sparse-MLA attention compute at the
single-token decode geometry, comparing:

  * "main default"  -> DEFAULT_SPARSE_MLA_CONFIG (BLOCK_N=64), i.e. what main
    resolves for Tq=1 (it ignores num_query_tokens).
  * "PR default"    -> the auto path on the PR branch (Tq=1 -> DECODE config,
    BLOCK_N=128, num_warps=8, waves_per_eu=1).

Both configs are force-selected through XKERNELS_SPARSE_MLA_CONFIG so the two
runs are directly comparable on the same node. The script also asserts the PR's
auto path actually picks DECODE for Tq=1 (and DEFAULT for Tq>1).

    python benchmarks/pr/pr61_sparse_mla_decode.py
"""
from __future__ import annotations

import json
import os

import torch

from xkernels import sparse_mla_attention
from xkernels._backends import Backend
from xkernels.ops.attention.triton.sparse_mla_config import (
    DECODE_SPARSE_MLA_CONFIG,
    DEFAULT_SPARSE_MLA_CONFIG,
    resolve_sparse_mla_config,
)


def _set_env(cfg: dict | None) -> None:
    if cfg is None:
        os.environ.pop("XKERNELS_SPARSE_MLA_CONFIG", None)
    else:
        os.environ["XKERNELS_SPARSE_MLA_CONFIG"] = json.dumps(cfg)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (MI300A) GPU.")
        return
    import triton

    dev = "cuda"
    # V4 single-token decode geometry.
    T, H, D, D_V, Kv, topk = 1, 128, 512, 448, 8192, 512
    sm_scale = 1.0 / (D**0.5)

    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, Kv, (T, topk), device=dev, dtype=torch.int32)
    sink = torch.randn(H, device=dev)

    def run():
        sparse_mla_attention(
            q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
            backend=Backend.TRITON,
        )

    # Sanity: confirm the PR's auto-resolution on this branch.
    _set_env(None)
    assert resolve_sparse_mla_config(num_query_tokens=1) == DECODE_SPARSE_MLA_CONFIG, \
        "auto path should pick DECODE for Tq=1"
    assert resolve_sparse_mla_config(num_query_tokens=8) == DEFAULT_SPARSE_MLA_CONFIG, \
        "auto path should keep DEFAULT for Tq>1"

    _set_env(DEFAULT_SPARSE_MLA_CONFIG)
    t_default = triton.testing.do_bench(run, warmup=25, rep=200)
    _set_env(None)  # PR auto path -> DECODE
    t_decode = triton.testing.do_bench(run, warmup=25, rep=200)
    _set_env(DECODE_SPARSE_MLA_CONFIG)
    t_decode_forced = triton.testing.do_bench(run, warmup=25, rep=200)
    _set_env(None)

    print(f"geometry: T={T}, H={H}, D={D}, d_v={D_V}, topk={topk}, Kv={Kv} (bf16)")
    print(f"DEFAULT config (BLOCK_N=64, main @Tq=1): {t_default:.4f} ms")
    print(f"DECODE  config (BLOCK_N=128, PR auto):  {t_decode:.4f} ms")
    print(f"DECODE  config (forced, parity check):  {t_decode_forced:.4f} ms")
    print(f"speedup (DEFAULT -> DECODE): {t_default / t_decode:.3f}x")


if __name__ == "__main__":
    main()
