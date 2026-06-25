# AGENTS.md

This repo is an **agent-native kernel library**. The design contract is
`docs/library.md` — read it if you're extending the substrate. Short version:

## The hard rule

Any new or edited kernel **must pass `verify` + `verify_parity` before it is
considered done.** From Python:

```python
from xkernels import verify, verify_parity
verify("dual_rmsnorm.triton@1.0.0", arch="amd_cdna3")          # vs the op's one reference
verify_parity("dual_rmsnorm@1.0.0")                            # backends agree with each other
```

Correctness is defined **once**, centrally, in the Op Spec — never re-derived by
intuition. If you write a kernel from scratch, `verify` still gives you the
correctness + parity guarantee.

## How to find a kernel (don't grep source)

Use structured retrieval over the contract, not text search:

```python
from xkernels import find_impl
find_impl("norm", {"x1": {"dtype": "bf16", "shape": [64, 1536]}}, target_arch="amd_cdna3")
# -> ranked candidates, each with `applicable` + `reject_reasons`
```

## What lives where

- `registry/ops/*.spec.json` — **Op Specs**: backend-agnostic contract
  (constraints, numerics/tolerances, shape sweep, reference). One per op.
- `registry/impls/*.card.json` — **Implementation Cards**: backend-specific
  (arch, specialization knobs, perf.measured, provenance). Many per op.
- `registry/shape_sweeps/*.sweep.json` — mandatory correctness sweep per op.
- `registry/schema/*.schema.json` — JSON Schemas (vendor-neutral, the bottom
  consumption tier is "just read the JSON").
- `src/xkernels/registry/` — loader, validation, constraint evaluator.
- `src/xkernels/retrieval.py`, `src/xkernels/verify.py` — the agent surfaces.
- `.agents/skills/*/SKILL.md` — authoring/porting/tuning playbooks (SKILL.md standard).
- `src/xkernels/mcp_server.py` — the MCP server exposing the above as tools.

## Adding a kernel

Follow `docs/adding-a-kernel.md` (now card-driven): write the Op Spec + a
reference + a shape sweep, then an Impl Card per backend, then run `verify`.
Every successful tuning is written back to the card's `perf.measured` so the
next task is cheaper (the compounding loop, §6.2).

## Portability stance

Portability lives in the **contract, not the source.** One Op Spec; native Impl
Cards per backend, each validated against the same reference. Never hardcode
`warp=32` — wave size is 32 (NVIDIA) / 64 (AMD) and belongs in the card's
`arch.wave_size`. "It runs on AMD" is not "it's good on AMD" — grade AMD perf
against the AMD roofline, never against the NVIDIA card.
