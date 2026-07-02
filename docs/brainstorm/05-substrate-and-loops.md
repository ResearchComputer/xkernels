# 05 — Substrate fit & the compounding loops

`02` said the DSL is a *producer*, the substrate is the *product*. This doc
spells out exactly how the DSL feeds the three compounding loops from
`library.md` §6.2/§7.4, and — just as importantly — what it must **not** break.

## 1. What the DSL emits, and who consumes it (unchanged)

```
                         DSL source (one .py)
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
   registry/ops/<op>.spec   reference.py    registry/impls/<op>.<backend>.card
              │                 │                  │
              └──────────┬──────┴──────────────────┘
                         ▼
          ┌──────────────────────────────┐
          │  UNCHANGED substrate surface  │
          │  find_impl()  verify()        │   ← these read JSON, not DSL
          │  verify_parity()  register()  │
          │  the JSON schemas             │
          └──────────────────────────────┘
```

Every arrow into the substrate is **JSON the schemas already validate**. The DSL
compiles to the same JSON a human would type; the registry's ingest
(`src/xkernels/registry/`) validates it identically. This is the property that
makes the DSL *safe to experiment with*: deleting it leaves a valid corpus.

## 2. Provenance: the DSL is traceable, not invisible

A card produced by the DSL carries provenance so the loops can reason about it:

```json
"provenance": {
  "authored_by": "dsl",                 // or "hybrid" if a human tuned it
  "skill_used": ["tile-a-gemm"],        // the skill that DROVE the authoring/tuning
  "derived_from": null,
  "source_path": "src/xkernels/ops/.../gemm.vkl.py",   // the DSL source, not a .cu
  "lowering": {                          // NEW namespaced field, ignored by old consumers
    "dsl_version": "0.1.0",
    "targets_emitted": ["triton","cuda","hip"]
  },
  "created": "2026-06-30T00:00:00Z"
}
```

The `authored_by: "dsl"` value is a new enum member; everything else is existing
schema. `lowering` is a *namespaced extension* — exactly the §8 "put our content
in extension fields, keep the standard core thin" discipline. An old consumer
that doesn't know `dsl` just treats the card as `authored_by: agent`.

This provenance lets the §7.4 question — *"which skills produce the
highest-quality cards?"* — be asked *across authoring mode*: are DSL-authored
cards more or less likely to pass parity first-try than hand-authored ones? That
is a measurable hypothesis, and the provenance field is what makes it answerable.

## 3. The three compounding loops, with the DSL in them

### Loop A — Cards accumulate `perf.measured` (§6.2)

**Unchanged in shape; richer in kind.** The DSL emits *starting* cards (correct,
untuned): single-kernel cards *and* graph cards. The autotune sweep
(`autotune-knob-sweep` skill) then searches the declared knob space — which the
DSL's `@targets(..., knobs={...})` block *is* — and writes the winner to
`perf.measured`. For a **graph card**, the winning knob fixes the per-node
compiled kernels and the graph is instantiated once with them; runtime
shape/arg variation rides parameter nodes (`07` §5). The DSL's contribution is
that the knob space is declared in the same file as the kernel, so it can't go
stale relative to the source.

The design discipline from the skills (the `autotune-knob-sweep` description:
*"record the winner to the card's perf.measured so the next task skips
autotuning"*) is fully preserved. The DSL just makes the *cold start* of that
loop cheaper: a correct card exists immediately, so autotuning can begin without
a porting/authoring round-trip first.

### Loop B — Skills accumulate outcome records (§7.3)

**Unchanged, but enriched.** Skills still run, still emit outcome records, still
get scored on `success_rate` / `median_iterations`. The DSL enriches the signal:
an outcome record can now carry `authoring_mode: dsl|hand|hybrid`, so we can
detect e.g. "`port-cuda-to-hip` succeeds in fewer iterations when the source
card is DSL-authored, because the contract is guaranteed non-drifting." That's a
new, cheap measurement that could justify the DSL on its own.

### Loop C — Provenance links cards ↔ skills ↔ measurements (§7.4)

**Strengthened.** `card.provenance.skill_used` ↔ `skill.metrics` already exists.
The DSL adds `card.provenance.authored_by: dsl` and the `source_path` pointing at
a regeneratable DSL source, so "fix the source card and flag its ports for
revalidation" (§2.3's `derived_from` lineage) becomes *mechanical*: re-run
`vkl build` and re-emit. A bug fix in a DSL primitive re-emits every card that
uses it — the §4.2 "a bug in a swizzle helper flags every card that uses it"
traceability, but now with a single regenerate command.

## 4. The reference loop — the structural win

This deserves its own section because it's the strongest argument for the DSL.

Today (§5.1): the reference is a *separate* hand-written torch function. Drift
between it and the device kernel is caught only by `verify` at runtime, on a
GPU. Every numerics bug that survives to `verify` is expensive.

With the DSL: the reference is the **same compute layer**, run on CPU tiles.
The contract between "what the kernel computes" and "what the reference
computes" is not a discipline — it is *identity*. Concretely:

- `reduce_dtype: fp32` is honored in both paths because both call the same
  `.to(fp32)`.
- `numerics.rtol` can be *checked against* the precision path the author
  declared, not guessed.
- A change to the compute body changes the reference on the next `vkl build`.
  There is no "I forgot to update reference.py" failure mode.

This closes the most expensive drift gap in the whole substrate. It is the
feature most worth prototyping first, *independent* of the multi-target question.

## 5. What the DSL must NOT do (to stay safe)

1. **Must not become a gatekeeper.** A hand-written `.cu`/Triton/CUTE card with
   a hand-written spec remains first-class. If `verify` ever required a DSL
   source, we'd have violated §8.4's bottom tier.
2. **Must not introduce a second contract semantics.** The DSL header is a
   *spelling* of the JSON spec, validated by round-trip through the existing
   ingest. If the DSL ever grew a constraint/numerics notion the JSON didn't
   have, it would fork the contract — the cardinal sin (§10).
3. **Must not hide the arch vocabulary.** `wave_size`, `requires`, `scratch.kind`
   must appear *in the emitted JSON* exactly as a human would write them, so
   `find_impl`'s reject-reason logic (§3.2) keeps working unchanged. The DSL
   names them in source; it must not rename them in the artifact.
4. **Must not lower the correctness bar.** A DSL-authored card (single-kernel *or*
   graph) passes the *same* shape sweep and the *same* `verify_parity` gate. No
   "DSL cards get a looser tolerance" — that would make portability (§5.3) a lie.
5. **Must not ship a graph that's slower than sequential launches.** Graph
   capture is a perf lever; a graph card that doesn't beat its sequential baseline
   (e.g. captured on a single big kernel, or re-instantiated every call) is a
   bug. `verify`'s perf block should flag it (`07` §8).
6. **Must not claim performance portability it didn't earn.** A native card's
   `perf.measured` cites a measured roofline fraction graded against the *vendor*
   ceiling (§10). "It ran on both backends" is not perf portability.

## 6. The agent-loop view (§6.1)

The per-task loop is unchanged in shape; the DSL only changes where step
"SPECIALIZE"/"VERIFY" *start from*:

```
1. SPEC      — unchanged
2. RETRIEVE  — unchanged (find_impl reads JSON)
3. SELECT    — unchanged
4. SPECIALIZE — STARTS from a DSL-emitted correct card (cold start removed)
5. VERIFY    — unchanged
6. DIAGNOSE  — unchanged; skills operate on the emitted card identically
7. RECORD    — unchanged; provenance now records authoring_mode
```

The win is concentrated in step 4's *entry condition*: instead of "no card
exists → run `tile-a-gemm` to produce one," it's "no card exists → `vkl build`
emits a correct starting card → autotune from there." Whether that's actually
faster end-to-end is the empirical question §9-style metrics would answer (`06`).
