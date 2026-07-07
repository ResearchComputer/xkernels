"""Benchmark: moe_align eager vs graph-replay (issue #52).

moe_align_block_size runs once per layer per token in decode serving. Its 5
scratch buffers are counters (must re-init each call), so the workspace's value
is address stability for graph capture, not skipping the init.
"""
import sys
sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.moe import MoeAlignWorkspace

DEV = "cuda"
torch.manual_seed(0)

def bench(fn, n_warmup=50, n_iter=500):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n_iter):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n_iter

print(f"{'M':>5} {'top_k':>6} {'E':>4} {'BS':>4} {'eager_ms':>10} {'graph_ms':>10} {'speedup':>9}")
for M, top_k, E, BS in [(8, 8, 256, 64), (64, 8, 256, 64), (128, 4, 64, 128), (8, 4, 8, 64)]:
    tids = torch.randint(0, E, (M, top_k), device=DEV, dtype=torch.int32)
    ws = MoeAlignWorkspace.allocate(M, top_k, E, BS, device=DEV)
    eager_ms = bench(lambda: xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False))
    for _ in range(3):
        xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
    graph_ms = bench(lambda: g.replay())
    sp = eager_ms / graph_ms if graph_ms > 0 else float("inf")
    print(f"{M:>5} {top_k:>6} {E:>4} {BS:>4} {eager_ms:>10.4f} {graph_ms:>10.4f} {sp:>8.2f}x")
