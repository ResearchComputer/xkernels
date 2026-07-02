# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 3: capture a composition of DSL kernels into a CUDA/HIP graph
(docs/brainstorm/07). The 3-GEMM chain (``examples/gemm_chain.py``) is the
§8 "chain of small kernels" case where graph capture wins.

Two gates (mirroring the substrate's verify contract):
  1. **Correctness**: captured-replay output == sequential output (the graph
     replays the SAME kernels, so it must not change numerics), AND the whole
     chain matches the composed CPU auto-reference (reference-mode ``ctx.call``).
  2. **Perf (§8)**: the captured graph BEATS sequential launch on a
     launch-overhead-bound chain. A graph that is correct-but-slower is a bug.

GPU-gated (torch.cuda.CUDAGraph needs a device). Runs on both sgs-gpu07 (H100)
and ds5 (GB10) — no native extension required (graph nodes ARE DSL launchers).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from xkernels.vkl import (  # noqa: E402
    capture,
    graph_of,
    measure,
    register_dsl,
    run_graph,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16, gemm_chain  # noqa: E402

_DT = torch.bfloat16
_DEV = "cuda"


@pytest.fixture(scope="module")
def _registered():
    """Register the gemm_bf16 node once (the chain calls it 3x)."""
    register_dsl(spec_of(gemm_bf16), backend="triton")
    return graph_of(gemm_chain)


def _inputs(M=128, K=64, N=128, *, dtype=_DT):
    g = torch.Generator(device=_DEV).manual_seed(0)

    def mk(*s):
        return (torch.rand(s, generator=g, device=_DEV) * 2 - 1).to(dtype)

    return {"a": mk(M, K), "w1": mk(K, K), "w2": mk(K, K), "w3": mk(K, N)}


class TestGraphCorrectness:
    """Gate 1: capture preserves numerics (vs sequential + vs composed reference)."""

    def test_captured_equals_sequential(self, _registered):
        spec = _registered
        ins = _inputs()
        seq = run_graph(spec, ins, backend="triton")
        cap = capture(spec, ins, backend="triton")
        try:
            out_cap = cap.replay()  # clone=True default
            torch.testing.assert_close(out_cap["y"], seq["y"], atol=1e-2, rtol=1e-2)
        finally:
            cap.close()  # release capture-pool state before later tests

    def test_captured_matches_composed_reference(self, _registered):
        """The whole chain vs the composed CPU auto-reference (reference-mode ctx)."""
        spec = _registered
        ins = _inputs(M=64, K=32, N=64)
        # reference mode runs each node's exact CPU auto-reference
        ref = run_graph(spec, ins, mode="reference")
        # device mode (sequential) should agree within bf16 tolerance
        seq = run_graph(spec, ins, backend="triton")
        torch.testing.assert_close(seq["y"], ref["y"].to(_DEV), atol=2e-2, rtol=2e-2)

    def test_replay_serves_new_args_via_static_buffers(self, _registered):
        """§4.2: one captured graph serves many args — replay(new_inputs) copies
        into the static buffers and replays. The output tracks the new inputs."""
        spec = _registered
        ins1 = _inputs()
        cap = capture(spec, ins1, backend="triton")
        try:
            y1 = cap.replay()["y"]
            # new, different inputs through the SAME captured graph
            g = torch.Generator(device=_DEV).manual_seed(99)

            def mk(*s):
                return (torch.rand(s, generator=g, device=_DEV) * 2 - 1).to(_DT)

            ins2 = {"a": mk(128, 64), "w1": mk(64, 64), "w2": mk(64, 64), "w3": mk(64, 128)}
            y2 = cap.replay(ins2)["y"]
            seq2 = run_graph(spec, ins2, backend="triton")
            torch.testing.assert_close(y2, seq2["y"], atol=1e-2, rtol=1e-2)
            # and it actually changed (not a stale replay)
            assert not torch.equal(y1, y2)
        finally:
            cap.close()


class TestGraphPerfGate:
    """Gate 2 (§8): the captured graph must beat sequential launch."""

    def test_captured_beats_sequential_on_small_chain(self, _registered):
        """The §8 win case: a chain of small kernels where launch overhead
        dominates. At 128x64x128 bf16 each GEMM is ~sub-µs compute but ~µs
        launch overhead, so 3 nodes × 3 launches vs 1 graph replay should win."""
        spec = _registered
        ins = _inputs(M=128, K=64, N=128)
        perf = measure(spec, ins, backend="triton", n_iters=200)
        assert perf.beats_sequential, (
            f"§8 gate FAILED: captured ({perf.captured_ms:.3f}ms) did not beat "
            f"sequential ({perf.sequential_ms:.3f}ms) over {perf.n_iters} iters; "
            f"speedup={perf.speedup:.2f}x. A graph that doesn't beat sequential "
            f"on a launch-bound chain is a bug (07 §8)."
        )

    def test_reports_honest_speedup(self, _registered):
        """The perf report is honest (not a hand-wave): speedup > 1 on the
        launch-bound regime, with real measured ms for both paths."""
        spec = _registered
        ins = _inputs(M=128, K=64, N=128)
        perf = measure(spec, ins, backend="triton", n_iters=200)
        assert perf.speedup > 1.0
        assert perf.sequential_ms > 0
        assert perf.captured_ms > 0
        # sanity: sequential is ~3 node-launches of work, captured is ~1 replay
        assert perf.captured_ms < perf.sequential_ms
