# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 3 example: a captured graph of DSL kernels (docs/brainstorm/07 §3,
strawman Ex.3).

A **3-GEMM chain** (``a -> gemm -> gemm -> gemm -> y``) is the cleanest
launch-overhead-bound composition: three small kernels where per-launch overhead
dominates compute, so capturing them into ONE ``torch.cuda.CUDAGraph`` replay
beats three sequential launches (the §8 table row "chain of many small kernels:
yes, a lot").

Each node is a ``register_dsl``-ed ``gemm_bf16`` launcher (the same kernel
Phase 2.0a/2.2a tuned to 97% of cuBLAS at 4096^3, here run at SMALL shapes where
launch overhead is the whole story). The body declares *what* composes;
``vkl.graph.capture`` records the node calls into one replayable graph.

Run on hardware:
    from xkernels.vkl import register_dsl
    from xkernels.vkl.examples import gemm_bf16, gemm_chain
    register_dsl(vkl.spec_of(gemm_bf16))
    perf = vkl.measure(vkl.graph_of(gemm_chain), inputs, backend="triton")
    print(perf)  # captured_ms < sequential_ms  -> the §8 gate
"""
from __future__ import annotations

from xkernels.vkl import TensorDecl, graph


# Boundary: a[M,K], w1[K,K], w2[K,K], w3[K,N] -> y[M,N]. Each GEMM's output
# feeds the next; the chain is a launch-overhead-bound composition at small K.
@graph(
    id="gemm_chain@1.0.0",
    inputs={
        "a": TensorDecl(rank=2, dtype=("bf16",), symbols=("M", "K")),
        "w1": TensorDecl(rank=2, dtype=("bf16",), symbols=("K", "K")),
        "w2": TensorDecl(rank=2, dtype=("bf16",), symbols=("K", "K")),
        "w3": TensorDecl(rank=2, dtype=("bf16",), symbols=("K", "N")),
    },
    outputs={
        "y": TensorDecl(rank=2, dtype=("bf16",), symbols=("M", "N")),
    },
    params=("a", "w1", "w2", "w3"),  # all vary at runtime -> static buffers + copy-in
    notes=(
        "3-node GEMM chain (a @ w1 @ w2 @ w3). Launch-overhead-bound at small K; "
        "the §8 'chain of small kernels' case where graph capture wins."
    ),
)
def gemm_chain(ctx, a, w1, w2, w3):
    """Three GEMMs in sequence; each output is the next node's input (a graph edge)."""
    h1, = ctx.call("gemm_bf16", a=a, b=w1)   # node 1
    h2, = ctx.call("gemm_bf16", a=h1, b=w2)  # node 2 (depends on node 1)
    h3, = ctx.call("gemm_bf16", a=h2, b=w3)  # node 3 (depends on node 2)
    return {"y": h3}
