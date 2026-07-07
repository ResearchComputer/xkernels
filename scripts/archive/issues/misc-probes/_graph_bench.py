"""Benchmark: eager vs CUDA-graph-replay for decode (issue #52's real payoff).

The workspace's eager-mode allocation savings are marginal (torch's caching
allocator makes torch.empty cheap). The load-bearing win is enabling CUDA graph
CAPTURE: with stable buffer addresses, the whole decode-step launch chain
captures once and replays as a single graph launch -- eliminating the per-op
Python dispatch + kernel-launch overhead. This is impossible with per-call
allocation (each capture would record a different address).
"""
import sys
sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.attention import PagedAttentionWorkspace
from xkernels.registry.input_gen import generate_inputs

DEV = "cuda"
torch.manual_seed(0)

def bench(fn, n_warmup=20, n_iter=200):
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

print(f"{'B':>4} {'seq':>5} {'eager_ms':>10} {'graph_ms':>10} {'saved_us':>10} {'speedup':>9}")
for B in [1, 4, 16]:
    for msl in [128, 512, 2048]:
        pt = {"dtype": "bf16", "B": B, "H_q": 32, "H_kv": 8, "D": 128,
              "block_size": 1, "max_seq_len": msl}
        inp = generate_inputs("paged_attention@1.0.0", pt, seed=0, device=DEV)
        ws = PagedAttentionWorkspace.allocate(B, 32, 128, device=DEV, dtype=torch.bfloat16)
        eager_ms = bench(lambda: xkernels.paged_attention(backend="triton", **inp))
        # capture the graph (workspace keeps addresses stable)
        for _ in range(3):
            xkernels.paged_attention(backend="triton", workspace=ws, **inp)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o = xkernels.paged_attention(backend="triton", workspace=ws, **inp)
        graph_ms = bench(lambda: g.replay())
        saved_us = (eager_ms - graph_ms) * 1000
        speedup = eager_ms / graph_ms if graph_ms > 0 else float("inf")
        print(f"{B:>4} {msl:>5} {eager_ms:>10.4f} {graph_ms:>10.4f} {saved_us:>10.2f} {speedup:>8.2f}x")
