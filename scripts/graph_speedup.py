#!/usr/bin/env python
"""Report the Phase 3 §8 graph-capture speedup honestly, on the current GPU."""
import torch

from xkernels.vkl import capture, graph_of, measure, register_dsl, run_graph, spec_of
from xkernels.vkl.examples import gemm_bf16, gemm_chain

register_dsl(spec_of(gemm_bf16), backend="triton")
spec = graph_of(gemm_chain)
dev = torch.cuda.get_device_name(0)

g = torch.Generator(device="cuda").manual_seed(0)


def mk(*s):
    return (torch.rand(s, generator=g, device="cuda") * 2 - 1).to(torch.bfloat16)


print(f"=== {dev} | 3-node GEMM chain (gemm_bf16 x3) ===")
for M, K, N, label in [
    (128, 64, 128, "small (launch-bound)"),
    (512, 512, 512, "medium"),
    (2048, 2048, 2048, "large (compute-bound)"),
]:
    ins = {"a": mk(M, K), "w1": mk(K, K), "w2": mk(K, K), "w3": mk(K, N)}
    p = measure(spec, ins, backend="triton", n_iters=200)
    flag = "WIN" if p.beats_sequential else "LOSS"
    print(
        f"  [{label}] {M}x{K}x{N}: seq={p.sequential_ms:.3f}ms "
        f"cap={p.captured_ms:.3f}ms speedup={p.speedup:.2f}x -> {flag}"
    )

# correctness cross-check: captured == sequential
ins = {"a": mk(128, 64), "w1": mk(64, 64), "w2": mk(64, 64), "w3": mk(64, 128)}
seq = run_graph(spec, ins, backend="triton")
cap = capture(spec, ins, backend="triton").replay()
print(f"\ncorrectness: max|cap-seq| = {(cap['y'].float() - seq['y'].float()).abs().max():.2e}")
