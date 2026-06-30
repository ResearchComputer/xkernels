# Using the library

How to install, call, and contribute to xkernels. This is the user-facing layer;
the design contract that everything rests on is [`../library.md`](../library.md).

## Install

```bash
pip install -e ".[dev]"                       # pure-Python (reference + triton if present)
XKERNELS_FORCE_BUILD=1 pip install -e .       # also build CUDA/HIP extensions
```

Triton/CUDA backends are optional; the package runs on the pure-torch reference
path anywhere. The CUTE DSL `cuda` backend needs `nvidia-cutlass-dsl` — install
with the `cute` extra (see [`ds5-testbed.md`](ds5-testbed.md)).

## Calling a kernel

Every op has one uniform PyTorch API and selects a backend automatically
(`backend="auto"`), with an override knob:

```python
import torch
from xkernels import fused_ffn, fused_moe_int4_w4a16

y = fused_ffn(x, w_gate, w_up, w_down)                          # auto-dispatch
y = fused_ffn(x, w_gate, w_up, w_down, backend="triton")        # force a backend

out = fused_moe_int4_w4a16(A, packed, scale, topk_ids, topk_w, group_size=32)
```

Override globally with `XKERNELS_BACKEND=reference|triton|cuda|hip`.

## The agent-native surfaces

xkernels is **agent-native**: the primary consumer can be an LLM agent. The
contract (full design: [`../library.md`](../library.md)) lives in machine-readable
artifacts, not prose.

```python
from xkernels import find_impl, verify, verify_parity

# Structured retrieval over the contract (not text search). Ranked + reject_reasons.
find_impl("norm", {"x1": {"dtype": "bf16", "shape": [64, 1536]}}, target_arch="amd_cdna3")

# Correctness vs the op's single backend-neutral reference + tolerances.
verify("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3")

# Cross-backend parity gate (do the backends agree with each other?).
verify_parity("dual_rmsnorm@1.0.0")
```

- **Op Specs** (`registry/ops/*.spec.json`) — backend-agnostic contract:
  constraints, numerics/tolerances, reference, shape sweep. One per op.
- **Implementation Cards** (`registry/impls/*.card.json`) — backend-specific: arch,
  specialization knobs, `perf.measured`, provenance. Many per op, each validated
  against the same reference.
- **Skills** (`.agents/skills/*/SKILL.md`) — authoring/porting/tuning playbooks
  (open SKILL.md format at the cross-harness `.agents/` standard).
- **MCP** (`python -m xkernels.mcp_server`, optional `[mcp]` extra) — exposes
  `find_impl`/`verify`/`verify_parity`/`record_measurement` to any MCP client.
- **Introspection** — `registry` (the loaded Op Specs / Impl Cards / skills /
  outcomes) and `backend_diagnostics()` (registered backends + suppressed
  backend-import failures per kernel) for debugging dispatch.
- **`AGENTS.md`** — the minimal pointer + the hard rule for any coding agent.

## Contributing a kernel

Follow [`../adding-a-kernel.md`](../adding-a-kernel.md) (card-driven): write the
Op Spec + a reference + a shape sweep, then an Impl Card per backend, then run
`verify` (+ `verify_parity` for multi-backend ops). **The hard rule: any new or
edited kernel must pass `verify` + `verify_parity` before it is considered done.**

## Running on the clusters

The on-cluster benchmark/profiling runbooks:
- [`clusters.md`](clusters.md) — beverin (MI300A / gfx942) and bristen (A100 /
  sm_80): benchmarking + profiling (rocprof-compute / ncu / nsys).
- [`ds5-testbed.md`](ds5-testbed.md) — the GB10 (sm_121) single-node CUTE-DSL
  testbed.

## The shared knowledge base

The cross-cutting benchmark/profile campaign, roofline diagnoses, host-side
gotchas, and the CUTE-DSL authoring reference live in **`meta/wiki/`** — the
shared source of "facts that cost real debugging time". Read the matching page
before profiling or porting.
