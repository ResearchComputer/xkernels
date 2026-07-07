"""Benchmark: allocation-owning eager vs preallocated workspace eager (issue #52).

Measures the per-call overhead the workspace removes: torch.empty + the CUDA
allocator round-trip on every decode step. Small-batch decode (where this
matters most) is dominated by launch + allocation overhead, not compute.
"""
import sys
sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.attention import PagedAttentionWorkspace
from xkernels.registry.input_gen import generate_inputs

DEV = "cuda"
torch.manual_seed(0)

def bench(fn, n_warmup=20, n_iter=100):
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

print(f"{'B':>4} {'seq':>5} {'eager_ms':>10} {'ws_ms':>10} {'saved_us':>10} {'pct':>7}")
for B in [1, 4, 16, 64]:
    for msl in [128, 512]:
        pt = {"dtype": "bf16", "B": B, "H_q": 32, "H_kv": 8, "D": 128,
              "block_size": 1, "max_seq_len": msl}
        inp = generate_inputs("paged_attention@1.0.0", pt, seed=0, device=DEV)
        ws = PagedAttentionWorkspace.allocate(B, 32, 128, device=DEV, dtype=torch.bfloat16)
        eager_ms = bench(lambda: xkernels.paged_attention(backend="triton", **inp))
        ws_ms = bench(lambda: xkernels.paged_attention(backend="triton", workspace=ws, **inp))
        saved_us = (eager_ms - ws_ms) * 1000
        pct = 100 * (eager_ms - ws_ms) / eager_ms
        print(f"{B:>4} {msl:>5} {eager_ms:>10.4f} {ws_ms:>10.4f} {saved_us:>10.2f} {pct:>6.1f}%")
