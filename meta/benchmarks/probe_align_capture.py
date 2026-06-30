# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Prove moe_align_block_size_triton(truncate=False) is HIP/CUDA-graph capturable.

Captures the sync-free align into a graph, replays it, and checks the replayed
output matches eager. Contrasts truncate=True (which keeps the .item() sync and
is not capturable). GPU-only; run on gfx942
(scripts/archive/issues/probe_align_capture_beverin.sbatch).
"""
from __future__ import annotations

import torch

from xkernels.ops.moe.triton.align_kernel import moe_align_block_size_triton


def main():
    if not torch.cuda.is_available():
        print("No GPU; graph capture proof requires gfx942 (or any CUDA/ROCm GPU).")
        return
    dev = "cuda"
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    M, top_k, E, block = 16, 8, 48, 16  # Kimi-K2.6 decode-ish
    g = torch.Generator(device=dev).manual_seed(0)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, dtype=torch.int32, device=dev)

    s_ref, e_ref, n_ref = moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    torch.cuda.synchronize()

    # Warmup on a side stream (JIT compile is not capturable).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s_cap, e_cap, n_cap = moe_align_block_size_triton(topk_ids, block, E, truncate=False)
    graph.replay()
    torch.cuda.synchronize()

    ok = (
        torch.equal(s_cap, s_ref)
        and torch.equal(e_cap, e_ref)
        and torch.equal(n_cap, n_ref)
    )
    print(f"truncate=False capture+replay matches eager: {ok}")
    print(f"  expert_ids fixed length = {e_cap.numel()} (max_blocks), "
          f"num_post = {int(n_ref.item())}, used_blocks = {int(n_ref.item()) // block}")

    # Contrast: truncate=True keeps the .item() sync; capturing it should error.
    try:
        warm = torch.cuda.CUDAGraph()
        with torch.cuda.graph(warm):
            moe_align_block_size_triton(topk_ids, block, E, truncate=True)
        print("  truncate=True captured WITHOUT error (unexpected — .item() sync)")
    except Exception as exc:  # noqa: BLE001
        print(f"  truncate=True not capturable (expected, host sync): {str(exc)[:90]}")


if __name__ == "__main__":
    main()
