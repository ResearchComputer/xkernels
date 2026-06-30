# Adding a Kernel

`xkernels` is **agent-native**: every op is a backend-agnostic **Op Spec** plus
one or more backend-specific **Implementation Cards**, validated by a
deterministic **harness**. See `meta/docs/library.md` for the full contract. This
guide is the concrete checklist.

The existing kernel-type-first source layout (`ops/<type>/{reference,interface,
triton,cuda}.py`) is unchanged and still the runtime. The registry is a layer of
**machine-readable metadata** on top of it.

## 1. Write the Op Spec (backend-agnostic, exactly one per op)

Create `registry/ops/<op>.spec.json`. This defines the contract *once* —
constraints, numerics/tolerances, the reference, and the mandatory shape sweep.
Schema: `registry/schema/op_spec.schema.json`.

Required keys: `id` (`<name>@<semver>`), `kernel` (the dispatch key, e.g.
`"ffn"`), `op.canonical_op` (the retrieval key), `inputs`, `outputs`,
`constraints` (decidable predicates — see the mini-language below), `numerics`
(`reference` import path + `rtol`/`atol`, optionally `by_dtype`), `shape_sweep`.

**Constraint mini-language** (reject-before-compile, §1.3.2): comparisons
(`== != < <= > >=`), arithmetic (`+ - * % //`), `and`/`or`/`not`, integer/string
constants, and the `dtype(<arg>)` builtin. Anything else is **rejected at
ingest** as non-decidable. Examples: `"K % 8 == 0"`,
`"dtype(x) == dtype(w)"`, `"M >= 16 and N % 32 == 0"`.

## 2. Write the reference + shape sweep

- `src/xkernels/ops/<type>/reference.py` — the **backend-neutral** oracle (pure
  torch). Register it: `register("<kernel>", Backend.REFERENCE)(fn)`. This is the
  `numerics.reference` the harness checks every card against. Never CUDA/HIP.
- `registry/shape_sweeps/<op>.sweep.json` — the mandatory correctness sweep
  (`default_dtype` + `points`, each a `{dtype, <shape symbols>}` dict). Include
  edge cases: tiny leading dim, non-power-of-2, divisibility boundaries (§5.1).
- Add a **seeded input generator** in `src/xkernels/registry/input_gen.py`
  mapping `(op_id, point, seed, device)` -> kwargs dict. This is what makes
  `verify` work without per-test boilerplate.

## 3. Write an Implementation Card per backend

`registry/impls/<op>.<backend>.card.json` — schema
`registry/schema/impl_card.schema.json`. One per backend: `backend`, `arch`
(`family`, `requires`, `wave_size` — 32 NVIDIA / 64 AMD / 0 agnostic),
`specialization_knobs` (the declared tuning space — declare only what you
validate), `perf` (`roofline`, `regime`), `provenance.source_path`.

The card resolves to the runtime callable via `(op_spec.kernel, card.backend)`,
so also register the backend in source:
`@register("<kernel>", Backend.TRITON)` under `ops/<type>/triton/`.

## 4. Verify + record (the hard rule)

```python
from xkernels import verify, verify_parity
verify("<op>.<backend>@<ver>", arch="<target>")     # vs the op's one reference
verify_parity("<op>@<ver>")                         # backends agree with each other
```

A card **cannot publish** until `verify().correctness.passed` is true across the
shape sweep, and (for multi-backend ops) `verify_parity().agree` is true (§2.4).
On success, write the tuning back so the next task skips autotuning (§6.2):

```python
from xkernels.registry import record_measurement
record_measurement("<op>.<backend>@<ver>", arch=..., shape=..., dtype=...,
                   knobs=..., ms=..., source=<verify run_id>)
```

Every measurement must cite a reproducible `source` run id and an `arch`, or the
loader drops it (§2.4).

## 5. Re-export + test

- Re-export the public op from `src/xkernels/__init__.py`.
- Add `tests/test_<type>.py` (correctness vs reference; skip on missing HW).
- The registry itself is exercised by `tests/test_registry.py`.

## Quick reference: artifacts per op

| Artifact | Path | Purpose |
|---|---|---|
| Op Spec | `registry/ops/<op>.spec.json` | the contract (one per op) |
| Impl Card | `registry/impls/<op>.<backend>.card.json` | one per backend |
| Shape sweep | `registry/shape_sweeps/<op>.sweep.json` | mandatory correctness sweep |
| Reference | `src/xkernels/ops/<type>/reference.py` | backend-neutral oracle |
| Input gen | `src/xkernels/registry/input_gen.py` | seeded harness inputs |
| Schemas | `registry/schema/*.schema.json` | JSON Schema (vendor-neutral) |
