# 07 — CUDA/HIP graph capture as a first-class capability

This doc covers the third day-1 requirement: the DSL must lower multi-kernel
**compositions** to **CUDA/HIP graphs**, not just to independent kernel launches.
Graphs are not a kernel-layer feature; they are an **orchestration-layer**
feature, and they need their own treatment because they change what a "card" can
be and how launch overhead is paid.

## 1. Why graphs belong in *this* library

Two levers kill the cost of a chain of small kernels, and the library already
uses one of them:

- **Fusion** (`add-epilogue-fusion`, `fuse-elementwise-chain` skills): collapse N
  kernels into 1. Kills both the intermediate-DRAM round-trips *and* the per-launch
  CPU overhead. Best case — but only legal when the ops can share a kernel body
  and a single launch grid.
- **Graph capture**: keep N kernels, keep intermediates, but pay **one** launch
  instead of N. CUDA/HIP graphs instantiate a DAG once and re-launch the whole
  thing in ~1–3 µs vs ~5–10 µs *per kernel* for ad-hoc launches.

The library today only has the fusion lever. That leaves a large class of real
workloads under-served: chains where fusion is illegal or uneconomical (a
reduction between two GEMMs, a norm whose output feeds a gather, an MoE routing
sequence) but where launch overhead still dominates because each kernel is small
relative to its launch cost. **Graph capture is the missing second lever**, and
unlike fusion it composes *with* fusion — a captured graph can contain fused
kernels. "Fuse what you can; graph-capture the rest" is the realistic strategy,
and the DSL should express both.

This is also the honest answer to a §10 anti-goal in disguise: §10 forbids "one
lowest-common-denominator source" because it leaves both backends slow — but
*launch overhead* is an independent source of slowness that no amount of
per-kernel tuning removes. A library that ships fast kernels in a slow launch
sequence is still slow. Graphs are how the orchestration layer keeps the
per-kernel wins.

## 2. The mental model: a graph is a captured composition

The substrate already has the seed of this: every Op Spec has a `composes_with`
field (`library.md` §2.1), and composition is reasoned "at the op level." Today
that field is aspirational — nothing *captures* a composition. The DSL makes it
real: **the DSL's dataflow between kernel calls *is* the graph DAG**, and the
emitter instantiates it.

```
   @graph def f(x):  xn = rmsnorm(x);  y = gemm(xn, w);  return gelu(y)
                          │                │                   │
                          ▼                ▼                   ▼
              graph node: rmsnorm  →  graph node: gemm  →  graph node: gelu
                     (edge = "xn feeds gemm", "y feeds gelu")
                          │
                          ▼  instantiate once, launch many
                   cudaGraphExec_t / hipGraphExec_t
```

The author writes ordinary-looking kernel calls; the DSL treats each call as a
graph node and each returned tensor as an edge. There is no manual
`cudaGraphAddKernelNode` / `cudaGraphAddDependencies` in user code — that's the
emitter's job. The author declares *what* composes; the DSL captures *how*.

## 3. The `@graph` surface (strawman)

```python
from vkl import graph, kernel, Tensor

@kernel(id="rmsnorm@1.0.0", ...)
@targets(triton=..., cuda=..., hip=...)
def rmsnorm(ctx, x, w): ...

@kernel(id="gemm@1.0.0", ...)
@targets(triton=..., cuda=..., hip=...)
def gemm(ctx, a, b): ...

@kernel(id="gelu@1.0.0", ...)
@targets(triton=..., cuda=..., hip=...)
def gelu(ctx, x): ...

@graph(
    id="rmsnorm_gemm_gelu@1.0.0",
    captures=True,            # lower to an instantiated graph, not N launches
    params=["x", "w", "w2", "bias"],   # these vary at runtime → parameter nodes
)
def rmsnorm_gemm_gelu(x, w, w2, bias):
    xn = rmsnorm(x, w)        # node 1; xn is a graph edge, not a sync
    y  = gemm(xn, w2)         # node 2; depends on node 1
    return gelu(y + bias)     # node 3 (+ elementwise fused or its own node)
```

`vkl build` emits, per target, the graph construction code:

```c
// pseudo-emission for the CUDA target
cudaGraph_t g; cudaGraphCreate(&g, 0);
cudaGraphNode_t n1 = add_kernel_node(g, rmsnorm_kernel, {x, w, params...});
cudaGraphNode_t n2 = add_kernel_node(g, gemm_kernel,    {xn, w2, params...}, deps={n1});
cudaGraphNode_t n3 = add_kernel_node(g, gelu_kernel,    {y,  bias},          deps={n2});
cudaGraphExec_t exec; cudaGraphInstantiate(&exec, g, ...);
// runtime: cudaGraphLaunch(exec, stream) — one launch, three kernels
// shape/arg changes: cudaGraphExecKernelNodeSetParams(exec, n_i, new_params)
```

HIP emission is the same shape (`hipGraph_t`, `hipGraphInstantiate`,
`hipGraphLaunch`, `hipGraphExecKernelNodeSetParams`). **One DSL source → two
graph backends**, each validated against the composed reference.

## 4. The three graph mechanisms the DSL must map to

### 4.1 Explicit construction (the default)
The emitter builds `cudaGraphAddKernelNode` + `cudaGraphAddDependencies`
explicitly from the declared dataflow. Cleanest, most optimizable, and gives the
runtime the full DAG so it can schedule across nodes. **This is the v1 mode.**

### 4.2 Parameter nodes (required for reuse)
A graph instantiated with fixed args is useless across shapes/inputs. CUDA/HIP
expose parameter-update APIs (`cudaGraphExecKernelNodeSetParams`,
`hipGraphExecKernelNodeSetParams`) that change a node's kernel args *without*
re-instantiating the graph. The DSL's `params=[...]` declaration marks which
inputs are runtime-varying → the emitter wires them as updatable parameter
nodes, so **one captured graph serves a whole family of shapes/args**. A
shape-driven grid (`@launch(grid=lambda...)`) becomes a parameter on the node.

This is what makes graphs economical: the instantiate cost is paid *once*
(first call), and subsequent calls with new `x`/`w`/shapes pay only the cheap
`setParams` + `launch`.

### 4.3 Conditional nodes (the MoE/sparse unlock — and the hardest part)
Data-dependent control flow is where naive graphs break: a graph is a static
DAG, but MoE routing, sparse attention indexing, and masked dispatch *branch on
data*. CUDA exposes `cudaGraphConditionalHandle` + `cudaGraphAddConditionalNode`
(CUDA 12.2+); HIP support is newer and more limited. The DSL should map a
guarded block to a conditional node:

```python
@graph(id="moe_dispatch@1.0.0", captures=True)
def moe_dispatch(x, routing, expert_weights):
    idx = route(x, routing)                       # node: produces idx
    out = zeros_like(x)
    with ctx.conditional(idx.numel > 0):          # → conditional node
        out = gather_dispatch(x, idx, expert_weights)
    return out
```

**This is the genuinely open part of the graph story** (`06` open question B3):
conditional-node support is uneven across CUDA versions and HIP, and the DSL
cannot silently fall back to re-launching the graph per iteration (that throws
away the graph benefit *exactly* on the workloads that need it most). The honest
design: the DSL emits conditional nodes where the target supports them, and
*rejects at build time* (not silently degrades) where it doesn't — routing the
author to stream-capture or a fused kernel instead.

### 4.4 Stream capture (the escape hatch, not the default)
`cudaStreamBeginCapture` / `cudaStreamEndCapture` (and HIP equivalents) capture
an existing launch sequence into a graph by recording the stream. The DSL should
expose this as an opt-in `@graph(captures="stream")` mode for compositions that
don't fit the explicit-construction model (host-side logic between kernels,
third-party calls, etc.). **Not the default** — stream capture is opaque (the
runtime can't see node-level structure for optimization) and harder to wire to
parameter/conditional nodes. But it's the safety net that keeps the DSL from
dead-ending on a composition it can't express explicitly.

## 5. Graph × autotune

A captured graph references a *specific compiled kernel* per node. Autotuning
changes a knob → recompiles the kernel → the graph node must be re-bound. Two
honest options:

- **Re-instantiate on knob change.** Autotuning is an offline/batched activity;
  each knob point produces a fresh `cudaGraphExec_t`. The winning knob → the
  published graph. Simple; pays instantiate per knob point (fine — autotune is
  not on the hot path).
- **Parameter nodes for shape/arg; re-instantiate only for knob.** The published
  graph uses parameter nodes for everything *except* the compiled kernel, so
  runtime shape/arg variation is cheap, and only a knob change (rare, deliberate)
  re-instantiates.

**Lean: the second.** The autotune sweep (`autotune-knob-sweep` skill) produces
the winning knob and writes it to the card's `perf.measured`; the graph is
instantiated once with that kernel and then reused via parameter nodes. This
keeps the compounding loop (`05` Loop A) intact: the measured tuning is the graph
that ships.

## 6. Graph × fusion (they compose, they don't compete)

A node in a graph can itself be a *fused* kernel. The realistic pipeline for a
chain like `rmsnorm → gemm → bias+gelu`:

1. Fuse `bias + gelu` into one kernel (legal, pointwise → `fuse-elementwise-chain`).
2. Leave `rmsnorm` and `gemm` as separate nodes (fusion illegal: different grids,
   a reduction boundary).
3. Graph-capture all three nodes → one launch.

The DSL should make this *expressible in one source*: fusion is a property of a
`@kernel` body (or an `@epilogue` hook, `03` Axis E), and graph capture is a
property of the `@graph` wrapper. They are orthogonal axes the author composes.
A design that forced "fuse *or* graph" would be leaving performance on the table.

## 7. Substrate fit: is a graph a new card kind?

This is a real schema question. Proposal:

- A **single-kernel op** stays exactly as today: Op Spec + Impl Card. No change.
- A **multi-kernel composition** is authored as a `@graph`, which emits:
  - a **composed Op Spec** whose `inputs`/`outputs` are the graph's boundary
    tensors and whose `numerics.reference` is the *composed* auto-reference (the
    CPU evaluation of the same kernel chain);
  - one **Impl Card per target** whose `backend`/`arch`/`specialization_knobs`
    are inherited from its nodes, plus a new namespaced `launch: { graph: true,
    nodes: [...], params: [...] }` field.

The namespaced `launch.graph` field is exactly the §8 "put our content in
extension fields" discipline: an old consumer that doesn't know graphs sees a
normal card and just runs the kernels sequentially (functional equivalence);
a graph-aware runtime captures and re-launches. **Functional portability
(§5.3) holds either way; performance portability is the graph-aware path.**

`verify(card, arch)` then runs the *whole graph* against the composed reference
— the harness treats a graph card as one launchable unit, exactly as it treats a
fused-kernel card today. `verify_parity` checks the two graph backends (CUDA
graph vs HIP graph) agree.

## 8. When graphs win vs when they don't (perf honesty)

| Workload | Graph helps? | Why |
|---|---|---|
| Chain of many small kernels (norm→gemm→act, MoE routing) | **Yes, a lot** | Launch overhead ≳ kernel time; 1 launch beats N. |
| Single big GEMM / attention | No | Launch overhead ≪ compute; capture adds instantiate cost for ~zero gain. |
| Data-dependent control (MoE/sparse) | **Yes, if conditional nodes** | Otherwise re-launch per branch throws away the graph. |
| One-shot, shape-changing every call | Marginal | Re-instantiate cost may exceed launch savings → parameter nodes or skip capture. |

A graph card that is correct but *slower* than the sequential launches (e.g.
captured on a single big kernel, or re-instantiated every call) is a **bug**,
not a feature. The `perf.roofline` field on a graph card should carry a
`launch_overhead_bound` regime, and `verify`'s perf block should flag a graph
that doesn't beat its sequential baseline. This is the same "slow everywhere is
a failure" discipline (`06` §B) applied at the orchestration layer.

## 9. Open questions specific to graphs

- **B3 (conditional coverage):** how much of the MoE/sparse workload can
  conditional nodes actually cover, given HIP's uneven support? Fallback policy
  (reject vs stream-capture vs fused-kernel) must be decided per-op.
- **Graph-card schema:** is `launch.graph` the right home, or does composition
  deserve its own artifact type ("Composition Card")? Lean: namespaced field on
  the Impl Card until a second use case forces a split.
- **Capture determinism (§5.4):** graph instantiate/launch must be on the
  deterministic feedback path — no host-side randomness in node setup. Needs an
  explicit rule.
- **Graph × epilogue fusion ordering:** when a graph node *is* a fused kernel,
  does the fusion skill operate before or after graph capture? Lean: fusion is
  resolved at kernel-emission time; graphs compose already-fused kernels.
