# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 3 §4.3 probe: conditional nodes (data-dependent control flow — the
MoE/sparse case). This is the genuinely open part of the graph story
(docs/brainstorm/07 §4.3, open question B3).

A captured graph is a STATIC DAG: every node runs on every replay. Host-side
``if`` on DEVICE data is a sync (it must read a GPU value back to the CPU to
branch), which is **illegal inside graph capture** — the sync invalidates the
capture and torch raises ``cudaErrorStreamCaptureInvalidated``.

The destructive wrinkle: an invalidated capture POISONS the CUDA context for the
rest of the process (no ``capture_error_mode`` recovers it — verified across
global/thread_local/relaxed). So this test runs the provoked capture in a
**subprocess**: the child asserts capture raises (the §4.3 rejection), and the
poisoned CUDA state dies with the child. The parent pytest process stays clean
(the failure mode is isolated, never leaking into sibling tests).

The honest v1 boundary: **dense chains only; data-dependent control flow is
rejected at capture time, never silently degraded** (07 §4.3 — re-launching per
branch throws away the graph benefit exactly on the workloads that need it most).
When/if a future torch/CUDA version supports conditional graph nodes
(``cudaGraphAddCond`` / ``hipGraph*``), the subprocess flips to "captures" and the
boundary moves — this test is the canary.

GPU-gated.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from xkernels.vkl import (  # noqa: E402
    GraphCtx,
    TensorDecl,
    graph,
    graph_of,
    register_dsl,
    run_graph,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16  # noqa: E402


@graph(
    id="cond_probe@0.0.0",
    inputs={
        "a": TensorDecl(rank=2, dtype=("bf16",), symbols=("M", "K")),
        "b": TensorDecl(rank=2, dtype=("bf16",), symbols=("K", "N")),
        "gate": TensorDecl(rank=1, dtype=("fp32",), symbols=("G",)),  # device data
    },
    outputs={"y": TensorDecl(rank=2, dtype=("bf16",), symbols=("M", "N"))},
)
def cond_probe(ctx, a, b, gate):
    """A body with data-dependent control flow on DEVICE data (the MoE/sparse
    shape). ``gate.sum().item()`` is a GPU->CPU sync — legal at runtime, ILLEGAL
    under graph capture. This is exactly the branch a conditional node would
    need to express."""
    if gate.sum().item() > 0:  # GPU->CPU sync — legal sequential, illegal in capture
        y, = ctx.call("gemm_bf16", a=a, b=b)
    else:
        y = a[:, : b.shape[1]].to(torch.bfloat16) * 0  # degenerate branch
    return {"y": y}


@pytest.fixture(scope="module")
def _registered():
    register_dsl(spec_of(gemm_bf16), backend="triton")
    return graph_of(cond_probe)


def test_conditional_runs_sequentially(_registered):
    """The body is correct when run sequentially (the sync is legal here)."""
    spec = _registered
    g = torch.Generator(device="cuda").manual_seed(0)

    def mk(*s):
        return (torch.rand(s, generator=g, device="cuda") * 2 - 1).to(torch.bfloat16)

    ins = {"a": mk(64, 64), "b": mk(64, 64), "gate": torch.ones(4, device="cuda")}
    out = run_graph(spec, ins, backend="triton")
    assert out["y"].shape == (64, 64)


# The provoked-capture runs in a SUBPROCESS: an invalidated capture poisons the
# CUDA context for the whole process (no capture_error_mode recovers it), so the
# rejection must be isolated from sibling tests. The child asserts capture raises
# and exits 0; its poisoned state dies with it.
_PROBE = textwrap.dedent(
    r"""
    import sys
    import torch
    from xkernels.vkl import TensorDecl, capture, graph, graph_of, register_dsl, spec_of
    from xkernels.vkl.examples import gemm_bf16

    register_dsl(spec_of(gemm_bf16), backend="triton")

    @graph(id="c@0", inputs={"a": TensorDecl(2,("bf16",),("M","K")),
                             "b": TensorDecl(2,("bf16",),("K","N")),
                             "gate": TensorDecl(1,("fp32",),("G",))},
           outputs={"y": TensorDecl(2,("bf16",),("M","N"))})
    def body(ctx, a, b, gate):
        if gate.sum().item() > 0:   # GPU->CPU sync — invalidates capture
            y, = ctx.call("gemm_bf16", a=a, b=b)
        else:
            y = a[:, :b.shape[1]] * 0
        return {"y": y}

    spec = graph_of(body)
    g = torch.Generator(device="cuda").manual_seed(0)
    mk = lambda *s: (torch.rand(s, generator=g, device="cuda")*2-1).to(torch.bfloat16)
    ins = {"a": mk(64,64), "b": mk(64,64), "gate": torch.ones(4, device="cuda")}
    try:
        capture(spec, ins, backend="triton")
    except Exception:
        sys.exit(0)   # capture was REJECTED — the §4.3 honesty rule holds
    sys.exit(1)       # capture SUCCEEDED — conditional nodes are now supported
    """
)


def test_conditional_capture_is_rejected_not_degraded():
    """§4.3 honesty rule: a data-dependent body must FAIL capture, never silently
    fall back to slow per-branch re-launch. Run in a subprocess (an invalidated
    capture poisons the CUDA context process-wide; the child isolates it).

    If this test STARTS FAILING (child exits 1), torch grew conditional-node
    support: flip the assertion and the §4.3 boundary moves."""
    r = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"expected capture to be REJECTED (§4.3), but it succeeded (child exit 1).\n"
        f"If torch now supports conditional graph nodes, the §4.3 boundary moved.\n"
        f"child stderr:\n{r.stderr}"
    )


def test_ctx_call_rejects_unregistered_node():
    """Belt-and-suspenders: reference-mode ctx.call names an unknown node → loud
    KeyError, not a silent wrong dispatch."""
    ctx = GraphCtx(mode="reference")
    with pytest.raises(KeyError):
        ctx.call("nonexistent_kernel_xyz", a=torch.zeros(1))
