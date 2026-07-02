# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 3: capture a composition of ``register_dsl``-ed kernels into a
CUDA/HIP graph (docs/brainstorm/07). The graph body declares *what* composes
(a chain of ``ctx.call(name, ...)`` node invocations); this module captures
*how* — one ``torch.cuda.CUDAGraph`` replay instead of N kernel launches.

Design (the honest, day-1-realizable slice):

  * **Explicit construction via ``torch.cuda.CUDAGraph``.** The body's node
    calls are recorded on a stream during ``with torch.cuda.graph(g):``; torch
    turns the launch sequence into a single replayable graph. This works on
    **both** NVIDIA (CUDA) and AMD (HIP, via torch's ROCm build) with **no
    native extension** — the graph nodes ARE the DSL launchers (each a
    ``register_dsl``-ed callable). The raw ``cudaGraphNode_t``/``hipGraphNode_t``
    emitter the doc (§3) describes as the per-target output is a follow-up; the
    torch path is the v1 that is live on hardware today (sgs-gpu07 + ds5).
  * **Static buffers (the §4.2 parameter-node discipline).** Graph capture
    pins memory addresses: the inputs captured are the inputs replayed. So
    ``capture()`` copies the caller's inputs into STATIC buffers and the body
    runs against those; ``replay(new_inputs)`` copies fresh data INTO the same
    buffers before replaying. One captured graph serves a whole family of
    shapes/args (the instantiate cost is paid once) — exactly the §4.2 economy.
  * **Correctness gate: captured == sequential.** A graph replay runs the SAME
    kernels as the sequential path, so the two must agree to the kernel's own
    tolerance (capture must not change numerics). A secondary composed-reference
    check (the whole chain vs each node's CPU auto-reference) is wired via
    reference-mode ``ctx.call``.
  * **Perf gate (§8): the captured graph must BEAT sequential launch** on a
    chain where launch overhead dominates (small kernels). ``measure()`` runs N
    iters of each and reports the speedup. A graph that doesn't beat sequential
    is a bug, not a feature (§8) — and this module says so honestly.

What is NOT here yet (the open Phase 3 boundary, §4.3 / §9 B3): **conditional
nodes** (data-dependent control flow — MoE/sparse). ``torch.cuda.CUDAGraph``
cannot capture host-side ``if`` on device data; the honest v1 is dense-chains-
only, and the conditional probe is a separate test that documents the boundary.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from .._dispatch import dispatch
from .surface import KernelSpec, TensorDecl

# kernel_name -> KernelSpec, populated by ``register_dsl`` so reference-mode
# ``ctx.call`` can resolve a node's auto-reference without a reverse registry
# lookup. (Additive: the substrate's loader keys by op_id; this vkl-internal
# map keys by the dispatch name ``spec.kernel``.)
_NODE_SPECS: dict[str, KernelSpec] = {}


def register_graph_node(spec: KernelSpec) -> None:
    """Record a DSL kernel as a capturable graph node (called by ``register_dsl``)."""
    _NODE_SPECS[spec.kernel] = spec


# ═══════════════════════════════════════════════════════════════════════════════
# §1  The @graph surface
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class GraphSpec:
    """The frozen record of a captured composition (the graph twin of
    ``KernelSpec``).

    ``body`` is a pure function ``body(ctx, **inputs) -> dict[out_name, tensor]``
    whose node calls (``ctx.call(name, **kw)``) ARE the graph DAG. ``params``
    names the boundary inputs that vary at runtime (copied into static buffers
    before each replay); the rest are captured once and held.
    """

    id: str
    inputs: dict[str, TensorDecl]
    outputs: dict[str, TensorDecl]
    body: Callable[..., dict[str, torch.Tensor]]
    params: tuple[str, ...]
    notes: str = ""


def graph(
    *,
    id: str,
    inputs: dict[str, TensorDecl],
    outputs: dict[str, TensorDecl],
    params: tuple[str, ...] | list[str] | None = None,
    notes: str = "",
) -> Callable[[Callable], Callable]:
    """Decorator: declare a multi-kernel composition that captures to a graph.

    The decorated body takes ``(ctx, **inputs)`` and returns a dict of boundary
    outputs; each ``ctx.call(kernel_name, **node_inputs)`` is one graph node that
    dispatches to a ``register_dsl``-ed launcher (device mode) or the node's
    auto-reference (reference mode).
    """

    def deco(body: Callable[..., dict[str, torch.Tensor]]) -> Callable:
        spec = GraphSpec(
            id=id,
            inputs=dict(inputs),
            outputs=dict(outputs),
            body=body,
            params=tuple(params) if params is not None else tuple(inputs),
            notes=notes,
        )
        body._vkl_graph = spec  # type: ignore[attr-defined]
        return body

    return deco


def graph_of(fn: Callable) -> GraphSpec:
    """Recover the ``GraphSpec`` attached by ``@graph``."""
    s = getattr(fn, "_vkl_graph", None)
    if s is None:
        raise AttributeError(f"{fn!r} is not an @graph-decorated function")
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# §2  The graph execution context (device vs reference mode)
# ═══════════════════════════════════════════════════════════════════════════════


class GraphCtx:
    """Passed to the body. ``ctx.call(name, **kw)`` is one graph node.

    In ``device`` mode (default) it dispatches to the registered GPU launcher —
    the nodes that get captured. In ``reference`` mode it runs each node's
    auto-reference (CPU, exact), so the SAME body yields the composed reference.
    Returns the node's outputs as a tuple in spec output order (matching the
    launcher contract — the body unpacks: ``out, = ctx.call("gemm_bf16", ...)``).
    """

    __slots__ = ("mode", "backend")

    def __init__(self, mode: str = "device", backend: str = "triton") -> None:
        if mode not in ("device", "reference"):
            raise ValueError(f"ctx mode must be 'device' or 'reference', got {mode!r}")
        self.mode = mode
        self.backend = backend

    def call(self, kernel_name: str, **kw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.mode == "reference":
            spec = _NODE_SPECS.get(kernel_name)
            if spec is None:
                raise KeyError(
                    f"reference-mode ctx.call needs {kernel_name!r} registered as a graph "
                    f"node (call register_dsl on it first); known: {sorted(_NODE_SPECS)}"
                )
            from .reference import run_reference

            return run_reference(spec, kw)
        return dispatch(kernel_name, backend=self.backend, **kw)


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Execution: sequential (baseline) vs captured
# ═══════════════════════════════════════════════════════════════════════════════


def run_graph(
    spec: GraphSpec,
    inputs: dict[str, torch.Tensor],
    *,
    mode: str = "device",
    backend: str = "triton",
) -> dict[str, torch.Tensor]:
    """Run the composition as N sequential kernel launches (the §8 baseline)."""
    ctx = GraphCtx(mode=mode, backend=backend)
    out = spec.body(ctx, **inputs)
    if not isinstance(out, dict):
        raise TypeError(
            f"@graph body must return a dict[out_name, tensor]; got {type(out).__name__}"
        )
    return out


@dataclass
class CapturedGraph:
    """A captured ``torch.cuda.CUDAGraph`` + its static input/output buffers.

    ``replay(new_inputs)`` copies the params into the static input buffers (same
    addresses the graph captured) and replays — one launch for the whole chain.
    """

    spec: GraphSpec
    graph: torch.cuda.CUDAGraph
    static_inputs: dict[str, torch.Tensor]  # captured addresses; copy-in before replay
    static_outputs: dict[str, torch.Tensor]  # captured output buffers; read after replay
    backend: str = "triton"
    _closed: bool = field(default=False, repr=False)

    def replay(
        self, inputs: dict[str, torch.Tensor] | None = None, *, clone: bool = True
    ) -> dict[str, torch.Tensor]:
        """Replay the captured graph. If ``inputs`` is given, copy the params
        into the static buffers first (one graph serves many args — §4.2).

        ``clone=True`` (default) returns owned copies; the perf loop passes
        ``clone=False`` to time pure replay (no read-back).
        """
        if self._closed:
            raise RuntimeError("replay() on a closed CapturedGraph")
        if inputs is not None:
            for name in self.spec.params:
                if name in inputs:
                    self.static_inputs[name].copy_(inputs[name])
        self.graph.replay()
        if clone:
            return {n: t.clone() for n, t in self.static_outputs.items()}
        return dict(self.static_outputs)

    def replay_raw(self) -> None:
        """Time-only replay: no copy-in, no read-back (the §8 perf inner loop)."""
        if self._closed:
            raise RuntimeError("replay_raw() on a closed CapturedGraph")
        self.graph.replay()

    def close(self) -> None:
        """Release the graph + static buffers (frees the capture-pool state).

        torch.cuda.CUDAGraph holds a reference to the capture memory pool; if the
        object outlives the test that created it (Python GC is lazy), the pool's
        state can leak into later CUDA ops ("Offset increment outside graph
        capture"). Closing explicitly + dropping the buffer refs + emptying the
        caching allocator lets CUDA reclaim the pool before the next op.
        Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        self.static_inputs.clear()
        self.static_outputs.clear()
        # dropping the last ref to the CUDAGraph lets torch release the capture
        # pool; empty_cache + sync makes that reclamation visible to the next op
        # (without empty_cache the caching allocator can keep stale pool blocks
        # and later ops hit "Offset increment outside graph capture").
        self.graph = None  # type: ignore[assignment]
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def capture(
    spec: GraphSpec,
    inputs: dict[str, torch.Tensor],
    *,
    backend: str = "triton",
    warmup: int = 3,
) -> CapturedGraph:
    """Capture the composition into a ``torch.cuda.CUDAGraph`` (§4.1 explicit
    construction).

    Protocol: copy inputs into static GPU buffers, warm up the launch sequence
    on a side stream (so lazy init / autotune / cubin load happens BEFORE
    capture), then record the body's node calls into the graph. The captured
    output buffers are pinned at their capture-time addresses.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("graph capture needs a CUDA/HIP device")

    # §4.2: static buffers — the addresses the graph captures and replays.
    static_inputs = {n: t.detach().clone() for n, t in inputs.items()}

    # warmup on a side stream (torch.cuda.CUDAGraph contract): flushes lazy init,
    # autotune, JIT compile so NONE of it leaks into the captured graph.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(max(1, warmup)):
            static_outputs = run_graph(spec, static_inputs, backend=backend)
    torch.cuda.current_stream().wait_stream(side)

    # capture: the body's ctx.call invocations become graph nodes.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_outputs = run_graph(spec, static_inputs, backend=backend)

    return CapturedGraph(
        spec=spec,
        graph=g,
        static_inputs=static_inputs,
        static_outputs=static_outputs,
        backend=backend,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §4  The §8 perf gate: captured must beat sequential
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GraphPerf:
    """The §8 verdict: does the captured graph beat sequential launch?

    A graph that is correct but slower than sequential is a BUG (§8) — capture
    added instantiate cost for ~zero gain. This reports the honest ratio so the
    gate can fail loudly on a graph captured over too-few / too-big kernels.
    """

    sequential_ms: float
    captured_ms: float
    speedup: float
    n_iters: int
    beats_sequential: bool
    nodes: tuple[str, ...] = ()


def _node_names(spec: GraphSpec) -> tuple[str, ...]:
    """Best-effort: record the node call order by tracing the body in reference
    mode with sentinel inputs is fragile; instead the author can set
    ``spec.notes``. For the perf report we just label it '<captured chain>'."""
    return ()


def measure(
    spec: GraphSpec,
    inputs: dict[str, torch.Tensor],
    *,
    backend: str = "triton",
    n_iters: int = 100,
    warmup: int = 3,
) -> GraphPerf:
    """Time captured-vs-sequential on ``n_iters`` steady-state iterations.

    The sequential path re-issues every node launch per iter (Python + per-kernel
    launch overhead × N nodes); the captured path issues ONE graph launch per
    iter. On a launch-overhead-bound chain (small kernels) the captured path
    wins; on a single big kernel it loses (§8 table). This function reports the
    honest ratio so the caller's gate asserts the win where graphs are supposed
    to help.
    """
    cap = capture(spec, inputs, backend=backend, warmup=warmup)

    # warm both paths
    for _ in range(5):
        run_graph(spec, inputs, backend=backend)
    for _ in range(5):
        cap.replay_raw()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run_graph(spec, inputs, backend=backend)
    torch.cuda.synchronize()
    seq_ms = (time.perf_counter() - t0) * 1e3

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        cap.replay_raw()
    torch.cuda.synchronize()
    cap_ms = (time.perf_counter() - t0) * 1e3

    # release the captured graph + its capture-pool state BEFORE returning, so it
    # never leaks into the caller's next CUDA op (the "Offset increment outside
    # graph capture" hazard). ``measure`` only needs the timings, not the graph.
    cap.close()

    speedup = seq_ms / cap_ms if cap_ms > 0 else float("inf")
    return GraphPerf(
        sequential_ms=seq_ms,
        captured_ms=cap_ms,
        speedup=speedup,
        n_iters=n_iters,
        beats_sequential=cap_ms < seq_ms,
        nodes=_node_names(spec),
    )
