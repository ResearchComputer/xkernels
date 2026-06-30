# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""PR #65 benchmark: auto-enabled INT4 fused top-k combine for decode buckets.

Two views are reported on one gfx942 (MI300A) GPU, over the Kimi-K2.6 per-rank
EP=8 geometry (E=48, top_k=8):

  (A) Raw GEMM + combine step in isolation (what the PR actually changes; matches
      the author's pinned-config microbench). The per-expert dispatch tensors are
      built once, outside timing. "unfused" = int4_w4a16_moe_gemm(combine=False)
      into [M*top_k,N] bf16 + view(M,top_k,N).sum(1); "fused" =
      int4_w4a16_moe_gemm(combine=True) atomic-accumulate into [M,N] fp32 + .to(bf16).

  (B) The full public fused_moe_int4_w4a16 op (incl. the shared moe_align build),
      fused_combine=False (main default) vs None (PR auto default).

The script asserts the PR's auto policy selects fused for these decode buckets.

    python meta/benchmarks/pr/pr65_int4_fused_combine.py
"""
from __future__ import annotations

import torch
import triton.language as tl

from xkernels._backends import Backend
from xkernels.ops.moe import make_w4a16_weights, moe_align_block_size_ref
from xkernels.ops.moe.interface import fused_moe_int4_w4a16
from xkernels.ops.moe.triton.configs import get_moe_int4_config
from xkernels.ops.moe.triton.moe_int4_kernel import int4_w4a16_moe_gemm

KIMI = dict(E=48, HIDDEN=7168, INTER=2048, TOP_K=8)


def _do_bench(fn):
    import triton

    return triton.testing.do_bench(fn, warmup=25, rep=100)


def _setup(M, N, K, *, dev="cuda"):
    top_k = KIMI["TOP_K"]
    group_size = 32
    packed, scale, _ = make_w4a16_weights(KIMI["E"], N, K, group_size, device=dev, seed=1)
    a = (torch.randn(M, K, device=dev) * 0.1).to(torch.bfloat16)
    topk_ids = torch.stack(
        [torch.randperm(KIMI["E"], device=dev)[:top_k] for _ in range(M)]
    ).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    config = get_moe_int4_config(KIMI["E"], N, K, M)
    block_m = config["BLOCK_SIZE_M"]
    sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, KIMI["E"])
    return a, packed, scale, topk_w, sorted_ids, expert_ids, num_post, top_k, group_size, config


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (MI300A) GPU.")
        return
    dev = "cuda"

    from xkernels.ops.moe.triton.moe_int4_kernel import _auto_fused_combine

    gate_up = dict(N=2 * KIMI["INTER"], K=KIMI["HIDDEN"], tag="gate_up")
    down = dict(N=KIMI["HIDDEN"], K=KIMI["INTER"], tag="down")

    print("== (A) raw GEMM + combine step in isolation ==")
    print(f"{'gemm':8} {'M':>4} | {'unfused_ms':>11} {'fused_ms':>9} | {'unfused/fused':>13}")
    for gemm in (gate_up, down):
        for M in (4, 8, 16):
            assert _auto_fused_combine(M, KIMI["TOP_K"], None), \
                f"auto policy must enable fused for M={M}"
            (a, packed, scale, topk_w, sid, eid, npp, top_k, gs, config) = _setup(
                M, gemm["N"], gemm["K"], dev=dev
            )

            def run_unfused(a=a, packed=packed, scale=scale, topk_w=topk_w, sid=sid,
                            eid=eid, npp=npp, top_k=top_k, gs=gs, config=config, M=M,
                            N=gemm["N"]):
                c = torch.empty((M * top_k, N), dtype=torch.bfloat16, device=a.device)
                int4_w4a16_moe_gemm(
                    a, packed, scale, c, topk_w.reshape(-1).float(), sid, eid, npp,
                    top_k=top_k, group_size=gs, mul_routed_weight=True,
                    compute_type=tl.bfloat16, filter_expert=False, config=config,
                )
                return c.view(M, top_k, N).sum(dim=1)

            def run_fused(a=a, packed=packed, scale=scale, topk_w=topk_w, sid=sid,
                          eid=eid, npp=npp, top_k=top_k, gs=gs, config=config, M=M,
                          N=gemm["N"]):
                out = torch.zeros((M, N), dtype=torch.float32, device=a.device)
                int4_w4a16_moe_gemm(
                    a, packed, scale, out, topk_w.reshape(-1).float(), sid, eid, npp,
                    top_k=top_k, group_size=gs, mul_routed_weight=True,
                    compute_type=tl.float32, filter_expert=False, config=config,
                    combine=True,
                )
                return out.to(a.dtype)

            t_unfused = _do_bench(run_unfused)
            t_fused = _do_bench(run_fused)
            print(
                f"{gemm['tag']:8} {M:4d} | {t_unfused:>11.4f} {t_fused:>9.4f} | "
                f"{t_unfused / t_fused:>12.3f}x"
            )

    print("\n== (B) full public fused_moe_int4_w4a16 op (incl. shared moe_align) ==")
    print(f"{'gemm':8} {'M':>4} | {'unfused_ms':>11} {'auto_ms':>9} | {'unfused/auto':>12}")
    for gemm in (gate_up, down):
        for M in (4, 8, 16):
            packed, scale, _ = make_w4a16_weights(
                KIMI["E"], gemm["N"], gemm["K"], 32, device=dev, seed=1
            )
            a = (torch.randn(M, gemm["K"], device=dev) * 0.1).to(torch.bfloat16)
            topk_ids = torch.stack(
                [torch.randperm(KIMI["E"], device=dev)[:KIMI["TOP_K"]] for _ in range(M)]
            ).to(torch.int32)
            topk_w = torch.rand(M, KIMI["TOP_K"], device=dev, dtype=torch.float32)

            def run_op(a=a, packed=packed, scale=scale, topk_ids=topk_ids, topk_w=topk_w,
                       fc=None):
                fused_moe_int4_w4a16(
                    a, packed, scale, topk_ids, topk_w, group_size=32,
                    mul_routed_weight=True, fused_combine=fc, backend=Backend.TRITON,
                )

            t_unfused = _do_bench(lambda: run_op(fc=False))
            t_auto = _do_bench(lambda: run_op(fc=None))
            print(
                f"{gemm['tag']:8} {M:4d} | {t_unfused:>11.4f} {t_auto:>9.4f} | "
                f"{t_unfused / t_auto:>11.3f}x"
            )


if __name__ == "__main__":
    main()
