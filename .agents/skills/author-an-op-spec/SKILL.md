---
name: author-an-op-spec
description: >
  Write the airtight, backend-agnostic Op Spec (constraints, numerics/tolerances,
  shape sweep) plus the backend-neutral reference and a seeded input generator for
  a NEW op — i.e. author the contract that every other skill (tile-a-gemm,
  port-cuda-to-hip, autotune-knob-sweep, establish-parity) consumes as a
  precondition. This is the gateway skill: it fires when find_impl / a coverage
  sweep shows a category gap or a requested op has no Op Spec yet. It is the ONLY
  skill whose validation gate is satisfiable on a CPU-only box (the reference card
  must pass verify against the op's one reference), so it is the first productive
  step when no GPU is available.
license: Apache-2.0
x-kernel-lib:
  id: author-an-op-spec@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "find_impl returns no candidate for a canonical_op the task needs"
      - "a category in §9 (gemm / attention / norm / reduce / activation / scatter / mhc) is empty or thin"
      - "a kernel source file exists under ops/ but has no registry/ops/<op>.spec.json"
      - "an agent is asked to 'add op X' and get_spec('X@..') raises KeyError"
    preconditions:
      - "no Op Spec exists yet for this op name (this skill CREATES it; do not run on an existing op — revise the spec inline instead)"
      - "a backend-neutral reference exists OR will be authored as the last step of this skill (the reference is part of the contract, not a backend)"
  inputs_required:
    - "op name (and version, default 1.0.0)"
    - "canonical_op (the retrieval key: gemm | attention | norm | reduce | activation | scatter | mhc | ...)"
    - "the kernel's dispatch key (op_spec.kernel, e.g. 'gemm', 'moe') if a runtime backend is already registered"
    - "the numerics story: input dtypes, accumulation dtype, and an honest tolerance source (an issue doc, a reference library, or a known acceptance bar)"
  tools:
    - get_op_spec
    - get_impl_card
    - verify
    - verify_parity
    - find_impl
  validation:
    must_pass:
      - "registry/ops/<op>.spec.json ingests (loader validation: JSON Schema + constraint mini-language is decidable)"
      - "every constraint in spec.constraints evaluates True on the sweep points and False on at least one deliberately-bad shape (reject-before-compile is real, §1.3.2)"
      - "the reference card verify('<op>.reference@<ver>', arch='any').correctness.passed == true across the shape sweep (CPU gate — the ONLY gate available without a GPU)"
      - "the seeded input generator yields byte-identical operands for any quant/dequant path so reference and backends consume the same bits"
      - "verify_parity(<op>@<ver>) is structurally wired (no load error); per_backend_runnable lists REFERENCE=True and others as the hardware allows"
    # NOTE: unlike the six kernel-layer skills, this skill's gate is satisfiable
    # on a CPU-only box. That is intentional — see the Honest no-GPU branch below.
  references:
    - "meta/docs/adding-a-kernel.md (the card-driven checklist this skill operationalizes)"
    - "meta/docs/library.md §1.3.2 (constraint mini-language), §2.4 (publish gate), §5.1 (backend-neutral reference), §10 (CUDA-shaped reference is an anti-goal)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-06-25T00:00:00Z"
    supersedes: []
---

## Why this skill exists

Every other skill in the library (`tile-a-gemm`, `port-cuda-to-hip`,
`tune-for-cdna`, `autotune-knob-sweep`, `establish-parity`) lists "Op Spec
written" or "a backend-neutral reference exists" as a **precondition**. None of
them tell you how to produce it. Before this skill, the library's execution chain
had no first link — an agent asked to "add op X" had only `meta/docs/adding-a-kernel.md`
(a doc, not a procedure-with-pitfalls) and had to re-derive the numerics
judgment calls from scratch each time. This skill is that first link, and it is
deliberately the one skill that pays off even on a CPU-only box: its validation
gate is the reference card passing `verify`, which needs no GPU.

## Procedure

> **DSL fast-path.** If the op is expressible as a fixed DAG of pointwise /
> reduce / MMA (gemm / norm / reduce / activation categories — see the routing
> table in [`author-a-kernel-with-dsl`](../author-a-kernel-with-dsl/SKILL.md)),
> author it with the **vkl DSL** instead: one `@kernel` body builds the math IR
> that lowers to BOTH the auto-reference (torch, bit-exact) AND a generated
> Triton kernel, so the spec + reference + cards are **emitted, not hand-written**
> — the same CPU gate, a fraction of the boilerplate. This hand path below is the
> fallback for ops the math IR cannot express (attention masking, scatter/gather,
> collectives, fp8 block-scale dequant not expressible as a pointwise cast). The
> two skills are peers; the contract they produce is interchangeable.

1. **Confirm the gap is real.** `get_op_spec('<op>@<ver>')` should raise, and
   `find_impl('<canonical_op>', ...)` should return no applicable card for the
   target arch/dtype. If a spec already exists, *revise it inline* — don't run
   this skill (its precondition is "no Op Spec exists yet"). Note the
   `canonical_op` you are seeding: §9 milestone coverage is tracked per category,
   and filling an empty category is the canonical trigger.

2. **Read the runtime, if any.** If a kernel source already lives under
   `ops/<type>/`, read the reference + interface + each backend entry to extract:
   the exact input/output names and dtypes, the dispatch key (`op_spec.kernel`),
   the accumulation dtype, and which backends are already `@register`-ed. The Op
   Spec must match the runtime's signatures *exactly* — a mismatch between the
   spec's `inputs` and the callable is the #1 silent harness failure.

3. **Author the Op Spec** at `registry/ops/<op>.spec.json` (schema:
   `registry/schema/op_spec.schema.json`). The hard parts are **constraints** and
   **numerics** — treat both as load-bearing, not boilerplate:
   - **Constraints** use the decidable mini-language only (§1.3.2): comparisons
     `== != < <= > >=`, arithmetic `+ - * % //`, `and/or/not`, int/str constants,
     and the `dtype(<arg>)` builtin. *Anything else is rejected at ingest.* Encode
     every shape rule that decides applicability — block sizes (`K % 128 == 0`),
     derived dims (`hc_mult3 == 2*hc_mult + hc_mult*hc_mult`), dtype equality
     (`dtype(q) == dtype(kv)`), capacity (`topk <= Kv`). A good constraint set
     rejects wrong shapes from metadata alone, before any compile.
   - **Numerics** carry `reference` (import path to the backend-neutral oracle),
     `rtol`/`atol` (per-dtype via `by_dtype`), `reduce_dtype`, `cross_backend_rtol`,
     and a `notes` string. Set tolerances from an **honest source** — an issue
     doc, a reference library's acceptance bar, or the mixed-precision math — and
     say *which* in `notes`. Never tighten to bit-equality (§5.4: fp16/bf16
     accumulation order legitimately differs) and never loosen to "just pass".
     Quantized/dequant ops deserve the most thought: the dequant must be bit-
     identical across backends (see step 5), so the only real divergence is fp32
     accumulation order.

4. **Write the backend-neutral reference** at `src/xkernels/ops/<type>/reference.py`
   and `@register("<kernel>", Backend.REFERENCE)` it. Pure torch, fp32
   materialization if needed, written for **clarity not speed**. It is the one
   source of truth every backend card is checked against (§5.1). It must NOT be
   CUDA- or HIP-shaped — a backend-shaped reference silently tilts correctness
   toward that backend (§10 anti-goal). For quantized ops, put the exact-inverse
   quant/dequant helpers here (or in a sibling module) so every consumer imports
   the *same* packing/quantization, guaranteeing byte-identical operands.

5. **Seed the input generator** in `src/xkernels/registry/input_gen.py`: a
   `_<op>(point, seed, device)` function returning the kwargs dict, registered in
   `_GENERATORS`. This is what makes `verify` work without per-test boilerplate.
   Critical rule: **operands that pass through a quant/dequant or packing path
   must be produced by the reference's own helpers**, not by independent torch
   randomness, so the reference and every backend consume byte-identical bits
   (e.g. fp8 block-scale operands via the reference's `per_block_quant_fp8`;
   W4A16 weights via `make_w4a16_weights`; MLA indices via a seeded generator).

6. **Write the shape sweep** at `registry/shape_sweeps/<op>.sweep.json`:
   `default_dtype` + `points` (each `{dtype, <shape symbols>}`). Cover: a tiny
   leading dim, a non-power-of-2 size, the divisibility boundary of your hardest
   constraint, and an fp32 point. **Pin any parameter the op is only
   sum/aggregate-invariant over** to the value where the per-element tensor
   equals the aggregate (e.g. split-K GEMM: sweep `n_splits=1` so the per-split
   output is element-wise comparable to the sum — the sum-invariance itself
   belongs in a unit test, not the correctness sweep).

7. **Author the reference Impl Card** at
   `registry/impls/<op>.reference.card.json`: `backend: reference`,
   `arch.family: any`, `specialization_knobs: {}`, a `perf.regime` string saying
   "pure-torch oracle", and `provenance.source_path` pointing at the reference
   file. Empty `specialization_knobs` is correct here — a reference oracle has no
   tuning space.

8. **Run the CPU gate.** `verify('<op>.reference@<ver>', arch='any')`. Must
   return `compiled=True, correctness.passed=True` with `max_abs_err` at the
   reference-vs-itself floor (0, or tiny fp noise). This is the correctness gate
   that *is* available on CPU — treat it as the hard rule (AGENTS.md).

9. **Author the triton/cuda/hip Impl Card(s)** that the runtime already
   `@register`s. **Declare `specialization_knobs` honestly**: only knobs the
   entry callable actually accepts belong here. Internal `@triton.autotune`
   tile configs (BLOCK_M/N/K, num_warps, num_stages) are selected *inside* the
   JIT kernel and are NOT accepted by the entry signature — declaring them would
   make the harness report them "unapplied" (§1.2 validity surface). Put those
   in the card's `notes` instead. Set `perf.measured: []` and
   `perf.roofline`/`perf.regime` from the op's character (gemm→compute_bound,
   tall-skinny fused gemm / decode attention→memory_bound).

10. **Wire parity structurally.** `verify_parity('<op>@<ver>')` should load
    without error. On a CPU-only box it returns `agree=True` trivially because
    only the reference is `compiled=True`; the real cross-backend gate fires on
    GPU. That is expected and honest — do NOT fake `compiled: true`.

11. **Update tests that assumed the gap.** `grep -rn` the repo for assertions that
    encoded "this category is empty" — e.g. `find_impl("<canonical_op>") == []`,
    `len(specs) >= <old>`, "not seeded" comments. Seeding a category *legitimately*
    invalidates them; update them to reflect the new coverage (use a genuinely-
    unseeded canonical_op as the "absent" probe, or bump the `>=` count).

12. **Hand off to the kernel-layer skills, gated on hardware.** Once a GPU is
    available, the downstream chain becomes productive in this order:
    `tile-a-gemm` (first native dense GEMM for a new family) →
    `port-cuda-to-hip` + `tune-for-cdna` (the §9 portability milestone) →
    `autotune-knob-sweep` (record winners) → `establish-parity` (the real
    cross-backend numeric gate). Record this op in the skill's
    `provenance.skill_used` once one of them lands a measured card.

    **Running those GPU gates — ds5 via rcc + docker.** The downstream skills'
    `verify` / `verify_parity` / `measure_perf` calls run on the GB10 (`arch=
    "nvidia_sm121"`) inside the NGC container: `rcc --profile ds5 push && rcc
    --profile ds5 run --docker -s '<python snippet>'` (`-s` for shell snippets /
    heredocs; `--docker` sets `PYTHONPATH=/workspace/src`). AMD/gfx942 →
    `scripts/cluster.sh run --host beverin`. DSL ops not yet imported by
    `ops/<x>/__init__.py` need `register_dsl(spec_of(<body>),"triton")` first.
    Full recipe + stand-up: `meta/docs/usage/ds5-testbed.md`.

## The honest no-GPU branch (read this if there is no GPU)

A box with torch but no CUDA can still satisfy **every must_pass above** by
shipping: a CPU-verified reference card (`compiled=True, passed=True`, exact) plus
triton/cuda/hip cards that honestly report `compiled=False`. This is not a
degraded deliverable — it is the *correct* state for the contract layer, and it
is the only state from which the GPU-gated skills can later fire without
re-authoring the contract. Concretely: do all 12 steps; at step 9 the non-
reference cards will read `compiled=False` in `verify` and that is the honest
result, not a failure. Do not set `compiled: true` in metadata to "look done" —
the harness is the source of truth, and a `compiled:false` card is publishable
exactly because it is truthful.

## Pitfalls

- **Declaring a knob the entry callable can't accept.** If the triton entry
  takes `(a, b, n_splits)` but you declare `BLOCK_M` as a `specialization_knob`,
  the harness reports it unapplied and the card is dishonest (§1.2). Either plumb
  the knob through the entry signature, or document it in `notes`.
- **A reference that drifts toward one backend.** A reference written with
  CUDA-style accumulation/tiling silently tilts "correct" toward NVIDIA (§10).
  Author it as the simplest pure-torch expression of the math.
- **Quantized operands generated independently per backend.** If each backend
  re-randomizes fp8/int4 operands, "parity" measures your random generator, not
  the kernel. Route all quant/packing through the reference's exact-inverse
  helpers (step 4/5) so every consumer starts from identical bits.
- **Tolerances set by feel.** `rtol`/`atol` chosen to "just pass" carry no
  information. Cite a source (issue doc, reference-library bar, or the
  mixed-precision math) in `numerics.notes`. For MoE/grouped GEMM bf16, ~2e-2 is a
  defensible bar; for fp32-dot refs, ~1e-3 rel. Cross-backend `rtol` is looser
  than single-backend `rtol` on purpose — never collapse them (§5.4).
- **Sweeping an aggregate-invariant parameter at non-invariant values.** A split-
  K GEMM's per-split tensor only equals the sum at `n_splits=1`; sweeping other
  values in the *correctness* sweep false-fails. Pin it; test the invariant
  separately.
- **Forgetting the "category was empty" tests.** After seeding `gemm` (or any
  category), tests asserting `find_impl("gemm") == []` now fail. This is a real,
  expected update — see step 11. Not fixing it looks like a regression you caused.
- **Treating `verify_parity.agree == True` on CPU as evidence of numeric parity.**
  With only the reference compiled, agreement is trivial. The gate is meaningful
  only once ≥2 backends are `compiled=True` (on GPU). Say so in the card `notes`.
- **Editing the Op Spec to "make a backend work."** The contract is invariant
  across backends (§7.2, §10). If a backend can't meet the spec, fix the backend;
  the spec moves only when the *contract* was wrong, and then for all backends.
