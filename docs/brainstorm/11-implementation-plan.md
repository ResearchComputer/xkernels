# 11 — Implementation plan

> This is the engineering translation of `06` (roadmap) + `10` (data structures)
> into concrete modules, deliverables, tests, and effort. It is **not** more
> brainstorming — every section names files, functions, and the existing code it
> reuses. Read `10` first; this doc assumes the math/schedule IR split.

## 0. Placement and the no-touch rule

**New package:** `src/xkernels/vkl/` ("vkl" = placeholder name, `06` A8).
**Tests:** `tests/test_vkl_*.py`, matching the flat `tests/` convention.
**Build probe:** `scripts/vkl_probe.py` (the `06` §D kill-experiment, Phase 0).

**The no-touch rule** (the safety property from `05`): the DSL/IR lands *additively*.
The following are **read-only consumers or extended-with-namespaces only** — no
behavior changes to existing surfaces:

| File | Touched? | How |
|---|---|---|
| `src/xkernels/verify.py` | **no** | runs unchanged on emitted cards |
| `src/xkernels/retrieval.py` | **no** | reads emitted JSON |
| `src/xkernels/_dispatch.py` | **no** | `register(...)` called by the emitter |
| `src/xkernels/registry/models.py` | **no** | `ImplCard.from_doc` ingests emitter output |
| `src/xkernels/registry/cost_model.py` | **extend** | add per-instr peaks via a new `vkl/archdb.py` that *calls* `arch_peaks()`; do not edit the existing tables |
| `src/xkernels/registry/archs.py` | **no** | reuse the arch enum |
| `registry/schema/*.schema.json` | **extend** | add `"dsl"` to `authored_by` enum; add namespaced `launch.graph` + `provenance.tuning_trace` (additionalProperties, not required) |
| `src/xkernels/ops/**` | **no** | existing kernels stay; the DSL emits *new* cards alongside |

If a phase requires touching a "no" file, the phase is wrong — re-scope.

## 1. Module breakdown (the target shape)

```
src/xkernels/vkl/
├── __init__.py              # public API: kernel, targets, graph, launch, build, load_ir, edit
├── surface.py               # @kernel / @targets / @graph / @launch decorators (Axis A1)
├── parse.py                 # restricted-Python-AST → math IR (the @kernel body)
├── ir/
│   ├── math.py              # MMA/Reduce/Pointwise/Load/Store/TensorRef (§10.1, frozen)
│   └── schedule.py          # Tile/MapTo/Stage/CopyAtom/Reduce/Knob (§10.2, editable)
├── archdb.py                # ARCH_INSTR_PEAK / ARCH_NATIVE_SHAPE / ARCH_SCRATCH_BYTES (§10.4.1)
├── cost.py                  # predict() / occupancy() (§10.4.2/.3) — calls registry.cost_model
├── edits.py                 # SetKnob/Retile/MapTo_/AddStage/... + check() + apply() (§10.3)
├── gate.py                  # the check gate: run all edits' checks, return Reject(reason)
├── reference.py             # math IR → torch (the auto-reference; §10.6 CPU lowering)
├── lower/
│   ├── triton.py            # math+schedule IR → @triton.jit  (Phase 1)
│   ├── cuda.py              #                  → CUTE/CUTLASS  (Phase 2)
│   └── hip.py               #                  → Composable Kernel / hipify (Phase 2)
├── graph.py                 # @graph → cudaGraph*/hipGraph* construction (Phase 3)
└── emit.py                  # IR → Op Spec JSON + Impl Card JSON (ImplCard.from_doc-compatible)
```

The discipline: **each module is independently testable** (parse → IR; IR →
reference; IR → cost; IR → emit; emit → `verify`). No "big bang" integration —
each arrow is a unit test.

## 2. Phased plan (effort-tagged)

Effort tags: 🟢 = days, 🟡 = 1–2 weeks, 🔴 = multi-week / research. Phases are
sequential; each has a hard gate before the next.

### Phase 0 — The kill-experiment *(🟢, CPU-only, ~3 days)*

**Goal:** de-risk the *whole* thesis before building any lowering. This is `06`
§D made literal. No `vkl/` package yet — just a probe script.

**Deliverable:** `scripts/vkl_probe.py` that, for two hand-written ops, does by
hand what the DSL would do automatically, and confirms the substrate accepts it:

1. **GEMM multi-target probe:** hand-write `gemm_bf16.spec.json` (copy the
   in-repo pattern), a torch `gemm_ref`, and three Impl Cards
   (`gemm_bf16.triton/cuda/hip.card.json`) where the `cuda` card has a per-target
   override noted in `provenance`. Confirm:
   - `ImplCard.from_doc` ingests all three (no schema rejection);
   - `verify(...)` passes the triton card on an available GPU;
   - the round-trip holds: emit a spec from a Python dict → JSON → `op_spec_from_doc` → re-emit → byte-identical JSON.
2. **Graph probe:** hand-write a 2-kernel `rmsnorm_gemm` card with a namespaced
   `launch.graph` field; confirm `verify` runs it (sequentially is fine for the
   probe) without choking on the unknown field.
3. **Edit-sequence probe:** on paper (in the script's docstring), sketch the
   `09` §8 edit trace for the GEMM and confirm each edit's preconditions are
   decidable from hand-constructed schedule-IR-shaped dicts.

**Gate:** (1) round-trip byte-identical, (2) `verify` passes, (3) edits locally
decidable. If any fails, execute the `06` §D scope-correction *before* Phase 1.
**This phase is the cheapest way to kill the project early if it's going to die.**

### Phase 1 — IR + Triton lowering + knob edits *(🟡, ~2 weeks)*

**Goal:** the smallest end-to-end slice that proves contract-native authoring.
Triton-only lowering (the always-available `arch: any` target), plus the two
cheapest edit primitives. This is `06` Phase 1 minus the native ceilings (those
are Phase 2).

**Modules:** `surface.py`, `parse.py`, `ir/math.py`, `ir/schedule.py`,
`reference.py`, `lower/triton.py`, `emit.py`, `edits.py` (SetKnob + Retile only),
`gate.py`, `archdb.py` (scalar peaks + scratch budgets; matrix peaks stubbed).

**Deliverables:**
- `@kernel` + `@targets(triton=...)` parses `dual_rmsnorm` (the simplest in-repo
  op) into a math IR; `reference.py` lowers it to torch and it matches the
  existing `dual_rmsnorm_ref` bit-for-bit on the sweep.
- `emit.py` produces `dual_rmsnorm.spec.json` + `dual_rmsnorm.triton.card.json`
  that `verify` passes **with zero hand-editing** (the `04` Ex.1 claim).
- `lower/triton.py` produces a registered callable that `dispatch("dual_rmsnorm",
  backend="triton")` runs.
- `SetKnob` + `Retile` edits: a 5-line script does a knob sweep on a GEMM-shaped
  op and writes the winner to `perf.measured` (programmatic
  `autotune-knob-sweep`, `09` §7).

**Tests:** `tests/test_vkl_roundtrip.py` (spec ⇄ JSON byte-identical),
`tests/test_vkl_reference.py` (auto-ref == hand-ref on the sweep),
`tests/test_vkl_emit_dual_rmsnorm.py` (`verify` passes first try),
`tests/test_vkl_edits.py` (gate rejects: tile not divisible by L5 shape, knob
out of choices, scratch overflow).

**Gate:** the `04` Ex.1 promise — `vkl build dual_rmsnorm` emits artifacts that
pass `verify` with no JSON hand-editing, and the auto-reference matches the
hand-written one. **Do not proceed to Phase 2 until the reference-equivalence
test is bit-exact on the full sweep.**

### Phase 2 — Multi-target lowering + matrix-engine edits *(🔴, ~4–6 weeks)*

**Goal:** the cross-platform + perf claim. Triton + CUDA + HIP from one source,
with per-target overrides reaching the ceiling. This is `06` Phase 1's native
half + the H1/H2 measurement.

**Modules:** `lower/cuda.py`, `lower/hip.py`, full `archdb.py` (wgmma/mfma peaks
+ native shapes), `edits.py` (`MapTo_`, `AddStage`, `SetCopyAtom`, `ReduceLevel`,
`PromoteOverride`), `cost.py` (predict + occupancy).

**Deliverables:**
- One GEMM family (`gemm_bf16`) lowers to **three** cards (triton/cuda/hip) from
  one `@kernel` + `@targets` + one `@gemm.target("cuda", ...)` override (`04`
  Ex.2). All three pass `verify`; `verify_parity` passes.
- **The native cards hit a credible roofline fraction** (`06` A2 gate). Concretely:
  the sm_90 cuda card ≥ 70% of the `ARCH_INSTR_PEAK["nvidia_sm90"]["wgmma"]`
  ceiling; the cdna3 hip card ≥ 70% of the mfma ceiling. Graded against the
  *vendor* ceiling (§10), never against each other.
- **H1/H2 measured** (`06` A1, `08` §5): for each of ~3 GEMM/attention ops,
  record whether the ceiling needed a primitive-swap (H2) or a full-body
  override (H1). This number shapes Phase 3's scope.

**Tests:** `tests/test_vkl_parity.py` (`verify_parity` on the three-target GEMM),
`tests/test_vkl_cost.py` (predict() within 30% of measured on the GEMM sweep —
the honest cost-model bar), `tests/test_vkl_override.py` (override body
type-checked against the math IR; math-IR-untouched invariant holds after every
edit).

**Gate:** the roofline fraction. If the native DSL cards can't reach ~70% of the
vendor ceiling after the autotune sweep, **execute the `06` A2 scope reduction**:
the DSL becomes the cold-start producer (portable card + reference) and native
ceilings stay hand-written skills. This is an honest outcome — record it and
stop, don't push Phase 3.

### Phase 3 — Graph capture *(🔴, ~3–4 weeks)*

**Goal:** the `07` capability. Multi-kernel compositions lower to CUDA/HIP
graphs on both backends. Conditional nodes probed on the MoE/sparse family.

**Modules:** `graph.py` (the `@graph` decorator + explicit-construction
emission, G1 in `03` Axis G), parameter-node wiring, `emit.py` extension for
namespaced `launch.graph`.

**Deliverables:**
- A 2–3 kernel dense chain (`rmsnorm → gemm → activation`, `04` Ex.3) captured on
  **both** CUDA and HIP; `verify` runs the whole graph as one launchable unit;
  `verify_parity` passes.
- **The captured graph beats the sequential-launch baseline** (`06` §B checklist;
  `07` §8). Measure on a chain of small kernels where launch overhead dominates.
- **Conditional-node probe** (`06` A3): try to capture one MoE/sparse op
  (`moe_sum_reduce` is the simplest). If conditional nodes cover it, ship; if
  not, **reject at build time** (the `07` §4.3 honesty rule) and document the
  boundary — scope graphs to dense chains.

**Tests:** `tests/test_vkl_graph_capture.py` (graph beats sequential baseline),
`tests/test_vkl_graph_conditional.py` (conditional node either captures or
build-time rejects; never silently degrades).

**Gate:** the graph beats sequential on dense chains. If conditional coverage is
poor, graphs ship as dense-chains-only — also an honest outcome.

### Phase 4 — The compounding loop + agent wiring *(🟡, ~2 weeks)*

**Goal:** make the win *compound* across tasks. The `tuning_trace` provenance
field + the agent-loop instrumentation (`09` §6).

**Modules:** `edits.py` (trace serialization), `emit.py` (write
`provenance.tuning_trace`), an MCP tool surface in `mcp_server.py` exposing
`load_ir` / `apply_edit` / `record_trace`.

**Deliverables:**
- Every autotune/tune run appends a `tuning_trace` to the card's provenance; the
  next run reads it and skips rejected edits (the `09` §6 step-7 win).
- Instrument the §9 acceptance metric: median-iterations-to-verify-pass for
  DSL-authored vs hand-authored cards, on a frozen replay set.

**Gate:** the §6.2 compounding property — a second agent facing the same
`(op, arch, shape)` reaches the tuned result in strictly fewer iterations than
the first, because it read the trace. If this doesn't hold, the trace format is
wrong, not the idea — iterate on the trace schema.

## 3. Dependency graph (what blocks what)

```
Phase 0 (probe) ──kill?──► stop
      │ pass
      ▼
Phase 1 (IR + Triton + knob edits) ──────────────────┐
      │ gate: auto-ref bit-exact                      │ these two are the
      ▼                                               │ real go/no-go for
Phase 2 (CUDA + HIP + matrix edits + cost model) ◄──┘ the whole thesis
      │ gate: native ≥ 70% vendor roofline
      │        (else: A2 scope reduction, stop here, honest)
      ▼
Phase 3 (graphs)          Phase 4 (compounding/agent)
      │ gate: graph > sequential     │ gate: trace compounds
      └──────────────────────────────┴──► breadth (rest of op families)
```

Phases 3 and 4 can partly overlap (different modules). **Phase 2 is the crux** —
it's where multi-target + perf either works or doesn't, and the A2 scope
reduction is the pre-committed honest fallback.

## 4. Testing strategy (round-trip is the spine)

The substrate already has `tests/test_registry.py` enforcing
schema↔Python-vocab sync. The DSL extends the same discipline:

- **Round-trip tests** (the invariant from `06` Phase 0): `@kernel` header →
  `emit.py` → JSON → `op_spec_from_doc` → re-`emit` → byte-identical JSON. This
  is the formal statement of "the header is a spelling of the JSON." Run it for
  every supported op.
- **Reference-equivalence tests**: auto-reference vs the existing hand-written
  `reference.py` for every migrated op, bit-exact on the full sweep. This is the
  `05` §4 structural guarantee made into a CI gate.
- **Edit-gate tests**: each edit primitive has a passing case and a
  reject-with-reason case (tile-not-divisible, scratch-overflow,
  illegal-instruction-for-arch). The reject reasons are asserted, not just the
  pass cases — the reasons are the training signal.
- **Lowering tests**: per target, the lowered callable passes `verify` against
  the auto-reference. The test does *not* re-derive correctness — it relies on
  the math-IR-untouched invariant + `verify`.
- **Frozen-replay** (§7.3 discipline, applied to the producer): a frozen set of
  `(op, edit-sequence)` pairs; a DSL change may not regress the set. Mirrors the
  skill-replay discipline already in the substrate.

## 5. What is explicitly NOT built (scope discipline)

Restating the `06` "not in scope" as engineering guardrails:
- **No MLIR / StableHOL lowering.** Bespoke IR (`09` §11). Revisit only if a
  4th target or a polyhedral pass demands it.
- **No standalone `.vkl` grammar** (Axis A2). Embedded Python only.
- **No primitive-authoring** (Axis D2). Primitives are library-provided in v1.
- **No stream-capture as default graph mode** (Axis G2). Explicit construction
  (G1) is the default; stream-capture is an opt-in escape hatch.
- **No making the DSL mandatory.** Hand-written cards remain first-class; the
  bottom tier ("just read JSON") keeps working (§8.4).
- **No looser tolerance for DSL cards.** Same `verify` gate, same sweep.

## 6. Effort/risk summary

| Phase | Effort | Risk | Mitigation |
|---|---|---|---|
| 0 (probe) | 🟢 3 days | low | it's a script; if it fails, stop |
| 1 (IR + Triton) | 🟡 2 wk | **medium — the auto-reference must be bit-exact** | Phase-0 reference probe de-risks this |
| 2 (CUDA + HIP) | 🔴 4–6 wk | **high — native ceiling may not be reachable from the IR** | the A2 scope reduction is pre-committed; Phase 1 stands alone if 2 fails |
| 3 (graphs) | 🔴 3–4 wk | medium — conditional-node coverage uneven | scope to dense chains; build-time reject the rest |
| 4 (compounding) | 🟡 2 wk | low | trace schema is iteratable |

**The honest headline:** Phase 0 + Phase 1 (🟢🟡, ~2.5 weeks) delivers a real
product — contract-native Triton authoring with an auto-reference and programmatic
autotuning — *even if every later phase fails*. Phase 2 is where the
cross-platform + perf ambition lives or doesn't. Phases 3–4 are upside. **The
plan is designed so that the cheapest phases are the ones most likely to ship
value, and the expensive phase has a pre-committed fallback.**

## 7. The first commit (concrete, to start Phase 0)

A single PR that adds `scripts/vkl_probe.py` + `tests/test_vkl_probe.py` and
*nothing else*. It:
1. Defines the round-trip assertion (dict → JSON → `op_spec_from_doc` → JSON, byte-identical) for one hand-built op.
2. Hand-builds a `gemm_bf16` spec + 3 cards and asserts `ImplCard.from_doc` ingests them.
3. Documents the `09` §8 edit trace as a docstring and asserts each edit's
   preconditions are decidable from hand-built schedule-IR dicts.

If that PR is green, Phase 1 starts with the round-trip invariant already
proven. If it's red, the thesis is weaker than `02` claims and the scope shrinks
before any lowering is written. **~3 days, no infrastructure, maximum
information.** That is the entire point of doing Phase 0 first.

## 8. Phase 0 outcome (shipped 2026-06-30)

**Status: DONE — `scripts/vkl_probe.py` + `tests/test_vkl_probe.py`, all checks green,
full suite 100 passed / 210 GPU-skipped, ruff clean, no substrate files touched.**

The probe verified all four theses *against the real substrate validators*
(`validate_op_spec`, `validate_decidable`, `op_spec_from_doc`, `ImplCard.from_doc`):

- **A (round-trip) holds, byte-identical.** A `KernelHeader` dataclass projects to
  a dict that passes real JSON-Schema validation, has only decidable constraints,
  ingests via `op_spec_from_doc`, and re-emits byte-identically. The header *is* a
  spelling of the spec — the central thesis of `02` is confirmed.
- **B (multi-target ingest) holds.** Three cards (triton `any` / cuda `nvidia_sm90`
  with `tensor_cores+tma+clusters`+`smem` / hip `amd_cdna3` with `matrix_cores+mfma`+`lds`)
  all pass `validate_impl_card` + `ImplCard.from_doc`. The `@targets` block projects
  cleanly into the existing arch vocabulary.
- **C (edit decidability) holds — with a load-bearing insight the probe caught.**
  Edit gates are **stateful**: a `retile`'s divisibility check only bites once an
  L5 matrix-engine map is present in the IR. The first probe run falsely passed
  `retile(M=96)` (should reject) because the test's starting schedule had no L5
  map. Fixing it confirmed the right model: a gate is a pure function of
  *(edit args, current IR, arch)*, so the caller must hand it the IR state where
  the constraint applies. This is a real Phase-0 finding about the `09` design,
  not a bug in the thesis.
- **D (schema finding) confirmed.** The closed schema (`additionalProperties: false`
  on `provenance`; `authored_by` enum lacks `"dsl"`) rejects exactly the namespaced
  extensions the plan wants: `authored_by: "dsl"`, `provenance.tuning_trace`, and
  top-level `launch.graph`. The probe also confirmed the positive precedent:
  `specialization_knobs.<knob>._doc` IS allowed (that sub-schema omits
  `additionalProperties: false`). **Action for Phase 1's first task:** the three
  schema edits from §0 are required and now precisely scoped.

**The probe is the seed Phase 1 grows.** Its `KernelHeader` / `emit_spec` /
`Schedule` / `gate_*` / `ARCH_DB` are honest minimal slices of the future
`vkl/surface.py`, `vkl/emit.py`, `vkl/ir/schedule.py`, `vkl/gate.py`,
`vkl/archdb.py`. Phase 1 promotes them from the script into the `vkl/` package
and wires the Triton lowering + `verify` end-to-end on `dual_rmsnorm`.

## 9. Phase 1 outcome (shipped 2026-06-30)

**Status: DONE (CPU-satisfiable subset) — `src/xkernels/vkl/` (1428 LOC) + 4 test
files (542 LOC), all green. Full suite 133 passed / 210 GPU-skipped / 0 failures,
ruff clean. The only existing file touched is `impl_card.schema.json` (the 3
planned namespaced edits — the no-touch rule holds).**

### What landed

- **The 3 schema edits (§0).** `authored_by` enum += `"dsl"`;
`provenance.tuning_trace` added (append-only edit log);
top-level `launch` object added (graph cards). All three now ACCEPT the
namespaced extensions (Phase 0 probe check D flipped from reject→accept).
- **The `vkl/` package** (`surface.py`, `ir/{math,schedule}.py`, `archdb.py`,
`reference.py`, `emit.py`, `edits.py`, `gate.py`, `auto.py`, `tiles.py`) —
honest seeds of the §1 module breakdown, promoted from the Phase 0 probe.
- **`examples/dual_rmsnorm.py`** — the Phase 1 worked example: one `@kernel` +
`@targets(triton=...)` source spells the whole contract that today lives across
8 hand-written artifacts.

### What was proven (CPU, CI-enforced — the 3 Phase 1 gates)

- **Gate A (round-trip):** `@kernel` header → `emit_spec`/`emit_card` → real
JSON-Schema validation (`validate_op_spec`/`validate_impl_card`) → dataclass
ingest (`op_spec_from_doc`/`ImplCard.from_doc`). The header IS a spelling of the
spec. Emitting twice is byte-identical (canonical projection).
- **Gate B (auto-reference equivalence):** the `@kernel` body — written
*independently*, kernel-flavored (`sum(x*x)/d + eps`, `rsqrt`) — matches the
hand-written `dual_rmsnorm_ref` (`pow(2).mean`) **bit-exact**
(`torch.equal`, maxdiff 0.00e+00) across all 5 sweep points × 3 seeds, including
the awkward `d2=33` and `T=1` cases. The two formulations are arithmetically
identical; the equality is CHECKED, not assumed.
- **Gate C (contract faithfulness):** the DSL-authored `dual_rmsnorm@1.0.0`
agrees with the hand-written spec on EVERY contract field (constraints,
numerics, tensor decls, canonical_op, fusions, shape_sweep, arch, roofline).
The ONLY diffs are by-design: `numerics.reference` (the DSL owns its
auto-path) and `perf.measured` (the DSL card starts empty).
- **Gate D (edit decidability):** `SetKnob`/`Retile` (Phase 1) + `MapTo_`/
`AddStage` (Phase 2 checks realized) all have accept AND reject-with-reason
cases. The stateful property (Phase 0 finding) is a dedicated test:
`Retile(96)` is accepted before a `MapTo(wgmma)` and rejected after it.

### What was deferred (GPU-gated, explicitly)

- **`lower/triton.py` + end-to-end `verify`.** Phase 1's body is vectorized torch
(the reference). The per-program tiling + `@triton.jit` lowering + `register()`
+ `verify("dual_rmsnorm.triton@...")` needs a GPU. The infrastructure
(`auto.py` auto-registration, `reference_path` resolution, schema-valid card
emission) is in place so the GPU path is wiring, not redesign.
- **Body → math IR derivation.** The math IR exists as frozen dataclasses (the
oracle edits respect), but is NOT yet parsed from the body. A future test
(Phase 1.5) will check the body lowered through the math IR matches the
declarative one. Phase 1's reference is the body directly — honest, not
hand-waved.
- **`cost.py` (predict/occupancy).** `archdb.py` carries the per-instruction
peaks + native shapes + scratch budgets the gate uses; the roofline aggregate
+ occupancy model land with Phase 2's matrix-engine edits.

### The headline

Phase 1 delivers the product the cheap-phases-promise: **contract-native
authoring where one source spells the whole contract, the body IS the
bit-exact reference, and the edit gate is locally decidable for an agent.** The
GPU lowering (Phase 1.5) and multi-target ceilings (Phase 2) build on this
without touching the contract surface — the no-touch rule held end to end.

**Next entry point:** `lower/triton.py` on a GPU box — emit a `@triton.jit` from
the `dual_rmsnorm` body (per-program tiling), `register()` it, and run
`verify("dual_rmsnorm.triton@1.0.0")` against the auto-reference. That closes
the `04` Ex.1 loop ("`verify` passes with zero hand-editing").

---

## 10. Phase 1.5 outcome (shipped 2026-06-30)

**Status: DONE — Triton lowering on H100 (sm_90). `src/xkernels/vkl/lower/`
(rowreduce.py + triton.py, ~330 LOC) + `tests/test_vkl_lower_triton.py` (3 GPU
gates), all green on sgs-gpu07. Local CPU suite still 40 passed / 3 GPU-skipped,
ruff clean. The `04` Ex.1 loop is CLOSED.**

### The design call: trace body, not direct body

Phase 1's body was *direct* torch (vectorized). Phase 1.5 needed a body that
lowers to BOTH torch (reference) AND Triton (device) — and the two lowerings
must be guaranteed to agree (docs/brainstorm/02 §1: "one computation, two
lowerings"). The clean way to buy that guarantee is to make the body BUILD AN
IR once, then have two interpreters over that IR:

- **`lower/rowreduce.py`** — a tile-DSL row-reduce IR: a DAG of frozen expr
  nodes (`LoadRow`, `Cast`, `BinOp`, `SumRow`, `Rsqrt`, `StoreRow`, ...) built
  via a `RowReduceCtx` in build mode. The body takes only `(ctx)` and references
  inputs/outputs BY NAME (symbolic in shape/dtype — the IR carries no concrete
  shapes). Two interpreters: `_TorchEval` (vectorized torch, the reference) and
  `_TritonGen` (codegen'd `@triton.jit`, with CSE via an id-keyed cache). The
  cast order is chosen to mirror the REFERENCE (`(x*inv).to(out) * w`), so the
  torch evaluator is bit-exact with the hand `dual_rmsnorm_ref`.
- **`lower/triton.py`** — `lower_to_triton(spec)` dispatches on
  `spec.launch.pattern`; Phase 1.5 supports `rowwise` (one program per token
  row, `BLOCK_D = next_pow2(d)`, masked load/store). `register_dsl(spec)` then
  registers the lowered callable under `(kernel, Backend.TRITON)`, so the
  unchanged substrate `verify` can run it.
- **`reference.py` rewritten** — `run_reference` now handles BOTH direct bodies
  (Phase 1) and trace bodies (Phase 1.5): for trace bodies it builds the IR
  once (memoized in a side dict, since `KernelSpec` is frozen) and evaluates on
  torch. `trace_ir(spec)` is the public entry the lowering consumes.

### What was proven (GPU, the 3 Phase 1.5 gates — `test_vkl_lower_triton.py`)

- **The generated kernel COMPILES + RUNS** on H100 and matches the hand
  reference `dual_rmsnorm_ref` across all 5 sweep points within the op's
  declared per-dtype tolerances (fp32 1e-5/1e-6; bf16 1.6e-2/1e-2).
- **The two-lowerings guarantee holds on the GPU:** the same trace IR lowered via
  torch (`run_reference`) and via Triton (`lower_to_triton`) agrees on the
  device within tolerance — the structural promise of docs/brainstorm/02 §1,
  now machine-checked.
- **End-to-end `verify` PASSES** with the DSL kernel registered: the substrate's
  OWN correctness gate (the one every hand-written card passes) is green,
  driven by ONE `@kernel` source with zero JSON hand-editing. The `04` Ex.1 loop
  is closed.

### The accidental win: the generated kernel is FASTER than the hand one

The DSL kernel runs at **0.099 ms** on H100 for the canonical shape
(T=8192, d1=1536, d2=512, bf16) vs ~0.18 ms for the hand-written
`dual_rmsnorm_triton`. The reason is honest, not magic: the generated cast order
(`(x*inv).to(bf16) * w`, mirroring the reference) avoids one fp32 mul in the
reduction tail that the hand kernel's fp32 `x*inv*w` performs. Both are within
tolerance; the DSL one is faithful to the reference. This is the first data
point for docs/brainstorm/02 §1's claim that a single honest source can reach
the vendor ceiling — here it cleared the hand baseline by accident.

### What was learned (load-bearing gotchas, named for the next agent)

- **Triton's `@triton.jit` needs a real file.** `inspect.getsourcelines(fn)`
  fails with `OSError: could not get source code` if the code object's
  `co_filename` is `<string>` (i.e. `exec(src, ...)` without a file backing).
  Fix: write the generated source to a real file under a generated/ dir and
  `compile(src, str(path), "exec")` — then `inspect`/`linecache` can read it.
  (`_get_kernel` in rowreduce.py.)
- **`verify` calls `fn(**inputs)` (keyword dict), not positionally.** A launcher
  that binds `*args, strict=True` fails with `zip() argument 2 is shorter`. The
  DSL launcher accepts EITHER form: positional (input order) for ergonomic
  direct use, OR keyword for `verify`/`dispatch`/`generate_inputs`.
- **The mask dim is the arange's source dim, not the tensor's.** A `LoadRow` on
  a rank-1 weight `w[d]` reuses the input's `cols` arange (built from the
  reduction dim), so the mask bound is `d_x1`, NOT `(w, 1)` (weights don't have
  a DimRef). The codegen/eval pull the mask dim from the `cols` arg, not the
  loaded tensor. This was the first real body-parse stress on the IR.
- **`KernelSpec` is frozen.** Caching the built trace on the spec raises
  `FrozenInstanceError`; use a side dict keyed by `id(spec)`.

### What is deferred (Phase 2, GPU-gated)

- **A second op (GEMM).** Phase 1.5 proved the row-reduce path; a GEMM stresses
  the body-parse assumptions (2D tiles, MMA, K-loop) the row-reduce never
  touches. That is the go/no-go for generalizing the lowering.
- **CUDA/HIP native overrides.** The per-target override bodies (the honest way
  per-target ceilings are bought) are Phase 2. Triton-only lowering is
  sufficient to close Ex.1; multi-target is the Phase 2 deliverable.
- **`cost.py` (predict/occupancy).** Still Phase 2; `archdb.py` carries the
  tables the gate uses.

### The headline

Phase 1.5 closes the loop the brainstorm promised: **one `@kernel` source
lowers to a generated Triton kernel that the unchanged substrate `verify`
passes — and, by an honest accident, beats the hand-written baseline.** The
contract is the product; the DSL is now a *generator* of cards + kernels, not
just a spelling of the header. Phase 2 (multi-target ceilings, GEMM,
cost-model predict) builds on the two-interpreter IR without touching the
contract surface.

---

## 11. Phase 2 plan: converge on the math IR (the go/no-go phase)

**Phase 2 is RED (§2) — the cross-platform + perf claim, with the 70% roofline
gate (§2 Phase 2). But before any native CUDA/HIP or roofline work, there is a
cheaper go/no-go that gates ALL of it: does the DSL's body-parse + lowering
generalize from row-reduce to a 2D-tiled GEMM?** The Phase 1.5 row-reduce IR
(`LoadRow`/`SumRow`/`Rsqrt`) genuinely cannot express a GEMM — no 2D tiles, no
K-loop, no MMA. The doc-10 **math IR** (`MMA`/`Reduce`/`Pointwise`, `ir/math.py`)
was built for exactly this but is currently UNUSED (Phase 1 deferred
body→math-IR derivation). Phase 2's GEMM is where that deferred decision bites.

**The plan: converge the body-IR onto the math IR, in two never-break-working-
code sub-steps.** No grab-bag — the end state is ONE body-IR (the math IR) +
two interpreters (torch reference, Triton device), the doc-10 design realized.

### Phase 2.0a — the GEMM math-IR proof (the go/no-go)

New module `lower/mathbody.py`: a body-DSL ctx (`MathBodyCtx`) that builds the
math IR, with two interpreters — `_TorchEval` (MMA → `torch.matmul` in fp32 =
the auto-reference) and `_TritonGen` (MMA → the tiled `tl.dot` K-loop from
`08` §3 / the hand `mm_fp8_blockscale_kernel.py` idiom). Author
`examples/gemm_bf16.py` (bare bf16 GEMM, fp32 accum, bf16 out; the cleanest
2D-tile stress — no fusion, no dequant).

**Gate:** `verify("gemm_bf16.triton@1.0.0")` passes on H100 — the DSL lowers a
2D-tiled GEMM correctly. The launch pattern (`Launch.tiled_2d()`) + the math
node types (MMA → K-loop; Reduce → row reduce) together drive the codegen
structure. The row-reduce IR is UNTOUCHED — `dual_rmsnorm` still passes.

### Phase 2.0b — migrate dual_rmsnorm to the math IR; retire row-reduce

Rewrite `examples/dual_rmsnorm.py` to build a math IR (Load, Reduce(sum),
Pointwise(rsqrt/mul/div/cast), Store). Extend `mathbody.py`'s lowering to
handle `Launch.rowwise()` over the math IR. **Gate: all Phase 1/1.5 tests still
pass.** Then delete `lower/rowreduce.py` — its expr set folds into the math IR.

### Phase 2.1–2.3 (after the IR is unified)

- **2.1 CUDA/HIP native overrides:** the per-target override body path (`08` §3
  second body; `lower/cuda.py`, `lower/hip.py`). The honest way per-target
  ceilings are bought.
- **2.2 schedule IR + edits:** the editable overlay (`MapTo`, `AddStage`,
  `Retile`) for GEMM tuning — the agent-editable surface, now over the unified
  math IR.
- **2.3 `cost.py` + the 70% roofline gate:** predict/occupancy (§4.3) and the
  §2 Phase 2 gate (native cards ≥ 70% vendor ceiling, or execute the A2 scope
  reduction).

**The headline for 2.0:** the go/no-go is IR expressiveness, and it's testable
on Triton alone (no native backends needed). Proving the math IR + lowering
handles GEMM de-risks the entire phase before any CUDA/HIP investment.

---

## 12. Phase 2.0a outcome (shipped 2026-06-30)

**Status: DONE — the math-IR convergence proven on a 2D-tiled GEMM (the go/no-go).
`src/xkernels/vkl/lower/mathbody.py` (~460 LOC) + `tests/test_vkl_lower_gemm.py`
(4 gates) + the DSL-emitted `registry/ops/gemm_bf16.spec.json` &
`registry/impls/gemm_bf16.triton.card.json`. 47 GPU tests green; local CPU
41 passed / 6 skipped; ruff clean. The row-reduce IR is UNTOUCHED — dual_rmsnorm
still passes on its own path.**

### The design call: converge on the doc-10 math IR

The Phase 1.5 row-reduce IR (``LoadRow``/``SumRow``/``Rsqrt``) genuinely cannot
express a GEMM — no 2D tiles, no K-loop, no MMA. The doc-10 **math IR**
(``ir/math.py``: ``MMA``/``Reduce``/``Pointwise``/``Load``/``Store``) was built
for exactly this but was UNUSED (Phase 1 deferred body→math-IR derivation).
Phase 2.0a is where that deferred decision was paid: a new ``MathBodyCtx`` builds
the math IR from the body, and two interpreters lower it:

- **``_TorchEval``** — walks the nodes; ``MMA`` → ``torch.matmul`` in fp32 (the
  bit-exact auto-reference, matching ``a.float() @ b.float()``).
- **``_TritonGen``** — walks the nodes; ``MMA`` → the tiled ``tl.dot`` K-loop
  (``08`` §3 / the hand ``mm_fp8_blockscale_kernel.py`` idiom). The tiling comes
  from the launch pattern (``Launch.tiled_2d()``) + the math IR's ``subscript``
  (Einstein labels): the output dims are the 2D grid; the contracted dim is the
  K-loop. No wave size, no instruction, no L5 shape named — the body stays
  portable (above L3).

### What was proven (the 4 Phase 2.0a gates)

- **The math-IR torch reference is bit-exact** with ``a.float() @ b.float()``
  across bf16/fp32, including ``[Load,Load,MMA,Pointwise,Store]`` node shapes —
  the structural promise that the body IS the reference, now over the GEMM.
- **The generated tiled Triton kernel matches** the auto-reference across the
  full sweep, within the op's declared per-dtype tolerances.
- **The two-lowerings guarantee holds** on GPU: same math IR, torch eval vs
  Triton codegen agree within tolerance.
- **End-to-end ``verify`` passes** on the DSL-emitted card (``gemm_bf16.triton@
  1.0.0``), with zero JSON hand-editing — the substrate gate driven by ONE
  ``@kernel`` source. The emitted spec + card were schema-validated and committed
  to ``registry/`` (the DSL is now a *generator* of committed artifacts).

### The perf data point (the Phase 2.2 starting line)

The DSL GEMM hits **~25–28% of the H100 bf16 wgmma ceiling** at large shapes
(2048³/4096³), dropping to ~1% at tiny shapes (launch-overhead-bound). This is
the honest, expected result for a **single-config, non-autotuned, non-pipelined**
Triton GEMM with fixed ``BLOCK_M=BLOCK_N=64, BLOCK_K=32``. The 70% roofline
gate (§2 Phase 2) is NOT met by the baseline — and that's by design: closing the
gap is exactly what Phase 2.2 (the schedule IR + ``autotune-knob-sweep`` +
software pipelining) is for. **2.0a proved the DSL can *express and correctly
lower* a GEMM; 2.2 is where it reaches the ceiling.** This is the clean
division: expressiveness (go/no-go, Triton-only, done) vs ceiling-reaching
(Phase 2.2+).

### What was learned (load-bearing gotchas)

- **The math IR's ``subscript`` IS the tiling source.** The contracted dim of an
  ``MMA(a[m,k], b[k,n])`` is the K-loop; the output dims are the 2D grid. This is
  why ``ir/math.py``'s ``TensorRef`` carries ``subscript`` (Einstein labels) —
  it's not decorative, it's what makes the lowering decidable from the IR alone.
- **``Load`` nodes must be emitted by ``ctx.load``.** A body that reads ``a`` in
  an MMA but never calls ``ctx.load('a')`` builds an IR whose MMA references an
  undefined name. ``ctx.load`` is now the dataflow entry point (deduped per name).
- **``verify`` needs a registered input generator.** The substrate's
  ``input_gen._GENERATORS`` is a hardcoded dict with no public register path; a
  DSL-emitted op has none → ``verify`` keyerrors. Fix: an additive
  ``register_input_gen(op_id, fn)`` (the no-touch "extend-with-namespaces" path)
  that ``register_dsl`` now calls, delegating to ``vkl.reference.make_inputs``.
  The DSL is now self-contained — emitting + registering a kernel wires its
  input gen with no hand-editing of the substrate.
- **The perf-script divisor.** TFLOPS = FLOPS / ``1e12`` (not ``1e9``). A wrong
divisor printed 2794× of ceiling — a 1000× error that's invisible unless you
sanity-check against the roofline. Always cross-check against the vendor
ceiling before believing a perf number.
- **Rsync source-arg footgun.** ``rsync src/ tests/ host:dir/`` creates a STRAY
  ``dir/xkernels/`` (the contents of ``src/``, so ``src/xkernels/`` →
  ``dir/xkernels/``) that shadows the real package and silently breaks
  ``registry_root()`` (``__file__`` resolves to the stray copy). Always
  ``rsync ./ host:dir/`` from the repo root.

### What is next (Phase 2.0b — unify, then 2.1–2.3)

- **2.0b:** migrate ``dual_rmsnorm`` to the math IR (``Reduce``/``Pointwise``),
  add the ``rowwise`` lowering path to ``mathbody.py``, retire ``rowreduce.py``.
  Gate: all Phase 1/1.5 tests still pass. One body-IR, two interpreters.
- **2.1:** CUDA/HIP native overrides (the per-target ceiling path).
- **2.2:** the schedule IR + edits over the unified math IR + autotune — the
  25% → 70% gap.
- **2.3:** ``cost.py`` + the 70% roofline gate.

### The headline for 2.0a

The go/no-go is answered: **the math IR + lowering generalizes from
row-reduce to a 2D-tiled GEMM, ``verify`` passes end-to-end with zero
hand-editing, and the baseline lands at an honest ~25% roofline that frames
Phase 2.2's tuning work.** No CUDA/HIP investment was needed to de-risk the
phase — expressiveness was provable on Triton alone. The DSL now has TWO ops
lowered through one body-IR design (row-reduce + math), and 2.0b collapses
them to one.

---

## 13. Phase 2.0b outcome (shipped 2026-06-30)

**Status: DONE — the two trace lowerings collapsed onto ONE body-IR.
``dual_rmsnorm`` now builds the doc-10 math IR (``Reduce``/``Pointwise``) the GEMM
already used, and ``lower/rowreduce.py`` is DELETED. One body-IR design, two
interpreters (torch + Triton), two launch patterns (``tiled_2d`` + ``rowwise``).
86 tests green (47 vkl + 39 registry); local CPU 41 passed / 6 skipped; ruff clean.**

### What was done

- **``dual_rmsnorm`` rewritten on the math IR.** The body now spells the
  arithmetic with ``ctx.load`` / ``ctx.reduce_sum`` / overloaded ``T`` operators
  / ``ctx.dim`` / ``ctx.lit`` / ``ctx.rsqrt`` / ``ctx.cast`` — the SAME
  ``MathBodyCtx`` the GEMM uses. The cast order is unchanged (``(v*inv).to(out)
  * w``), so the torch evaluator is STILL bit-exact with the hand reference
  (``test_auto_ref_bit_exact_hand_ref`` passes unchanged).
- **``_TritonGenRowwise`` added to ``mathbody.py``.** One program per leading-dim
  row; the ``Reduce(sum, axis=last)`` node becomes the program-local 1D tile
  (padded to ``next_pow2(d)``, masked ``cols < d``); rank-1 weights share the
  input's reduction dim (linked by the decl ``subscript`` symbol). Both latents
  compose in one kernel. The tiling is read off the math IR's ``subscript`` —
  no wave size, no instruction named (portable, above L3).
- **``lower/rowreduce.py`` DELETED.** ``reference.py`` folds ``rowwise`` into the
  math-body path; ``lower/triton.py`` routes both patterns through
  ``mathbody.launch(..., pattern=...)``. The bespoke row-reduce expr IR
  (``LoadRow``/``SumRow``/``Rsqrt``/...) is gone — one body-IR, two interpreters.
- **``ctx.dim`` + ``_DimRefMarker``** added: a symbolic ``inputs[t].shape[axis]``
  scalar (e.g. the reduction width for ``ss / d``). Resolves to the concrete
  shape value in the torch evaluator; the rowwise codegen lowers it to the
  runtime dim arg ``d_<symbol>`` (one per reduction axis, shared by every load
  on it).

### The proof (the 2.0b gate)

The gate was "all Phase 1/1.5 tests still pass on the unified IR" — and they do:

- ``test_auto_ref_bit_exact_hand_ref``: the math-IR torch evaluator is bit-exact
  with ``dual_rmsnorm_ref`` across the full sweep (the arithmetically-identical
  formulation ``sum(v*v)/d + eps``).
- ``test_dsl_triton_matches_reference`` / ``test_dsl_triton_matches_run_reference``:
  the rowwise math-IR kernel matches BOTH the hand ref (within tolerance) AND the
  torch auto-ref (two-lowerings agreement) on H100.
- ``test_register_then_verify_passes``: ``verify("dual_rmsnorm.triton@1.0.0")``
  PASSES with the kernel lowered from the math IR.
- ``dual_rmsnorm`` (math IR) runs at **~0.03 ms** on H100, ~1150 GB/s effective
  (memory-bound) — at parity-or-better with the retired row-reduce kernel.

### The bonus fix (Phase 2.0a's incomplete commit)

Running the FULL suite (not just ``test_vkl_*``) surfaced that the Phase 2.0a
commit of ``gemm_bf16`` artifacts was incomplete: every op in the registry must
have a **reference card** (the substrate invariant ``test_every_seeded_op_has_
reference_and_sweep``), and the auto-ref must resolve from a fresh ``import
xkernels`` (``test_reference_callables_resolve``). Both failed for gemm_bf16.

- **``emit_reference_card(spec)`` added to ``emit.py``**: every DSL op's body IS
  the auto-reference, so every DSL op has a reference card for free
  (``backend=reference``, ``arch=any``). Emitted ``gemm_bf16.reference.card.json``
  to the registry — the DSL is now a complete generator of committed artifacts
  (spec + reference card + per-backend cards).
- **``auto.get_auto`` lazy-imports ``xkernels.vkl.examples`` on a miss.** The DSL
  package stays side-effect-free at import (``__init__`` registers nothing), but
  the reference path (``xkernels.vkl.auto:<name>``) must resolve from a process
  that only did ``import xkernels``. The lazy import makes DSL ops symmetric
  with hand ops (whose reference module self-registers on the import
  ``_import_attr`` performs). Returns the ``_ref(**inputs)`` wrapper, not the raw
  ``(ctx)`` body.

Both fixes are CPU-doable and land additively (no substrate file touched beyond
the Phase 1 ``input_gen.register_input_gen`` extension). ``test_registry.py`` now
passes fully (the one remaining failure was a missing ``pyyaml`` on the GPU
venv — environmental, fixed by ``uv pip install pyyaml``).

### What was learned (load-bearing gotchas)

- **Run the FULL suite, not just the feature suite.** Phase 2.0a's "47 GPU tests
  pass" was only ``test_vkl_*``; the gemm_bf16 commit had been breaking
  ``test_registry.py``'s invariants since it landed, invisible because the vkl
  suite never exercises them. A feature suite proves the feature; only the full
  suite proves you didn't break the substrate's contract invariants.
- **Every committed op needs a reference card.** The registry invariant
  (``test_every_seeded_op_has_reference_and_sweep``) is load-bearing: one spec +
  a reference card + per-backend cards. The DSL's "body IS the reference" thesis
  means the reference card is FREE to emit — and mandatory to commit.
- **``id()``-keyed caches + ``lru_cache`` survive a single feature suite but
  interact with test ordering in the full suite.** The probe runs IN-PROCESS
  (``spec.loader.exec_module``), so it inherits any registry mutation from an
  earlier test. The lazy-import fix (above) is the robust answer: resolve-on-
  demand, not resolve-on-import.
- **The math IR's ``subscript`` does double duty.** For the GEMM it named the
  contracted K-loop dim; for the rowwise reduce it names the tile-width symbol
  that links a rank-2 input to its rank-1 weight (``x1[T,d1]`` and ``w1[d1]``
  share ``d1`` → one BLOCK + one mask bound). One field, two lowering duties.

### The headline for 2.0b

**One body-IR, two interpreters, two launch patterns.** The Phase 1.5 bespoke
row-reduce IR is gone; both ``dual_rmsnorm`` (rowwise) and ``gemm_bf16``
(tiled_2d) lower from the SAME doc-10 math IR via the SAME ``mathbody.py``.
This is the convergence ``04`` Ex.1 / ``11`` §11 anticipated — the math IR is
now the single lowering entry point, and Phase 2.1 (CUDA/HIP overrides) and 2.2
(schedule IR + autotune) build on it directly.

---

## 14. Phase 2.2a outcome (shipped 2026-06-30)

**Status: DONE — the schedule-IR-driven autotune sweep closed the gap to the
*practical* ceiling. The autotuned Triton GEMM hits 460 TFLOPS on a 4096³ bf16
problem: 1.7× the Phase 2.0a default (268 TFLOPS) and 96.6% of cuBLAS (477
TFLOPS). The Phase 1 ``ir/schedule.py`` + ``edits.py`` + ``gate.py`` modules —
built but unused since Phase 1 — are now LIVE: the sweep enumerates the card's
declared ``specialization_knobs`` via ``SetKnob``/``run_gate`` (the agent-editable
primitive) and measures each via the substrate's own ``verify(measure_perf=True)``.
94 tests green (8 new sweep gates); the winner + the full 108-config history are
written to the committed card's ``perf.measured`` + ``provenance.tuning_trace``
(the compounding loop, live).**

### What was done

- **The launcher accepts tile knobs (the plumbing).** ``lower/triton.py``'s
  launcher now separates input-kwargs from knob-kwargs (using the target's
  declared knob names) and threads them to ``mathbody.launch(..., **knobs)``.
  Because the launcher declares ``**kwargs``, the substrate's ``_apply_knobs``
  passes every requested knob through — so ``verify(impl_card_id, knobs={...})``
  *actually retargets the compiled kernel*. ``BLOCK_M/N/K`` are ``tl.constexpr``
  (Triton recompiles per value, internally cached); ``num_warps``/``num_stages``
  are Triton launch metas (the software-pipeline-depth lever that hides
  global-memory latency).
- **The GEMM declares its search space.** The Target's ``knobs`` field names
  ``BLOCK_M ∈ {64,128,256}``, ``BLOCK_N ∈ {64,128,256}``, ``BLOCK_K ∈ {32,64}``,
  ``num_warps ∈ {4,8}``, ``num_stages ∈ {2,3,4}`` (108 configs). The emitter
  turns these into the Triton card's ``specialization_knobs`` — no hand-editing.
- **``sweep.py`` (~300 LOC, new).** The DSL's programmatic
  ``autotune-knob-sweep``: ``schedule_from_card(card)`` rebuilds a ``ScheduleIR``
  from the card's declared knobs; ``enumerate_configs`` yields the Cartesian
  product; each config is bound via ``SetKnob`` edits through ``run_gate`` (the
  decidability check — value ∈ choices; trivially Ok for pure knobs, ready to
  reject once ``MapTo``/``AddStage`` bring L5-divisibility / scratch constraints,
  Phase 2.2b); each gate-passing config is measured by the unchanged
  ``verify(measure_perf=True, knobs=<config>)`` (the substrate's own ``do_bench``
  median, not a DSL reinvention); the min-ms winner is written to
  ``perf.measured`` (via the substrate's ``record_measurement``) and the whole
  sweep is appended to ``provenance.tuning_trace``.
- **The compounding loop is live.** The committed ``gemm_bf16.triton.card.json``
  now carries the real 4096³ winner in ``perf.measured`` (ms=0.298, knobs=
  BLOCK_M=128/BLOCK_N=256/BLOCK_K=64/num_warps=8/num_stages=4) AND a 108-entry
  ``tuning_trace`` (106 passed, 2 failed — the dead-ends the next agent skips).

### The numbers (H100 NVL, sm_90, bf16, 4096³)

| config | ms | TFLOPS | vs default | vs cuBLAS |
|---|---|---|---|---|
| Phase 2.0a default (64/64/32) | 0.512 | 268 | 1.0× | 56% |
| **swept winner (128/256/64, w8, s4)** | **0.298** | **460** | **1.7×** | **97%** |
| cuBLAS (``torch.matmul``) | 0.288 | 477 | — | 100% |

Graded against the *vendor* ceiling (the design's bar, §10): 47% of H100 SXM
bf16 dense (989 TFLOPS), 61% of H100 NVL/PCIe dense (756 TFLOPS). cuBLAS itself
reaches ~48% of the theoretical dense peak on this shape — so the autotuned
Triton GEMM is at ~97% of what the vendor library achieves, i.e. *at the
practical ceiling* for a dense GEMM on this backend.

### What this reframes about Phase 2.1 (native overrides)

The Phase 2 gate (§2) is: native cards ≥ 70% of the *vendor* ceiling after the
autotune sweep, else execute A2 scope reduction. Two reframings from this result:

1. **The autotune machinery is backend-agnostic and now exists.** ``sweep.py``
   drives the substrate's ``verify`` regardless of backend — it will sweep a
   CUDA or HIP card's knobs identically once those lowerings exist (Phase 2.1).
   "After the autotune sweep" is no longer a prerequisite to build; it's built.
2. **For dense GEMM, autotune alone reaches cuBLAS parity on Triton.** That
   directly informs the H1/H2 question (§2 deliverable): a dense bf16 GEMM's
   ceiling does NOT need a primitive-swap (H2) or a full-body override (H1) on
   the Triton backend — the schedule-IR sweep closed the gap. A native override
   body (CUTE/CUTLASS) would chase the last ~3% to cuBLAS and the path to the
   *theoretical* vendor peak (47%→70%), which is diminishing-returns territory
   for dense GEMM. Native overrides are better spent on ops where Triton's
   codegen genuinely can't reach the ceiling (e.g. attention with custom shapes,
   fp8 with non-trivial dequant) — that's where Phase 2.1's effort compounds.

This is *not* the A2 scope reduction: the native-override path stays open
(especially for AMD/HIP and for ops Triton can't express well). It's a data
point that for the worked example (dense bf16 GEMM), the designed ceiling-reaching
lever (schedule IR + autotune) is sufficient, and Phase 2.1 should target ops
where it isn't.

### What was learned (load-bearing gotchas)

- **``ms`` is milliseconds, not seconds — the REVERSE of the §12 TFLOPS gotcha.**
  ``flops / ms / 1e12`` printed 0.5 TFLOPS (off by 1000×); the fix is
  ``flops / (ms * 1e-3) / 1e12``. ``verify``'s ``perf.ms`` and ``do_bench`` both
  return milliseconds. Always cross-check against cuBLAS before believing a
  roofline fraction — cuBLAS is the sanity floor (if you "beat" it 2×, the math
  is wrong, not the kernel).
- **``record_measurement`` writes to the REAL committed card.** A sweep with
  ``record=True`` mutates the committed artifact (intended — that's the
  compounding loop), but it means TESTS must use ``record=False`` or write to a
  temp copy, or they pollute the committed card with test data on every run. The
  sweep tests use ``record=False`` for the sweep logic + a temp-copy path
  (monkeypatched ``_card_path``) for the trace-writeback unit test.
- **The substrate's autotune path is the right measurement layer.** Driving the
  sweep through ``verify(measure_perf=True, knobs=<cfg>)`` — not a DSL-reinvented
  timer — means the winner is a real ``do_bench`` median and the
  correctness-check happens for free (fast-but-wrong configs are discarded, the
  autotune skill's pitfall). The DSL's contribution is the *search driver*
  (schedule-IR edits), not the *measurement*.
- **``num_stages`` is the load-bearing knob.** The winner uses ``num_stages=4``
  (the deepest pipeline); configs with ``num_stages=2`` were uniformly slower,
  and the two FAILs were ``BLOCK_M=BLOCK_N=256, BLOCK_K=64, num_stages=4`` (smem
  overflow at the largest tile × deepest pipeline). This is exactly the
  ``AddStage`` scratch-budget constraint the ``gate.py``/``edits.py`` machinery
  is built to catch — Phase 2.2b wires it so the gate rejects those before
  launch instead of the kernel crashing.
- **The schedule IR is the source of truth, but the launcher doesn't read it
  directly.** The flow is: schedule IR → ``SetKnob``/``run_gate`` validates →
  ``verify(knobs=<cfg>)`` → ``_apply_knobs`` → launcher ``**kwargs`` →
  ``mathbody.launch(**knobs)``. The schedule IR never touches the launcher; the
  knob VALUES flow through ``verify``'s existing ``knobs=`` path. This keeps the
  lowering substrate-faithful (the launcher is a plain registered callable) and
  the schedule IR the *editable* representation (Phase 2.2b: Retile/MapTo edits
  change which configs the gate even admits).

### The headline for 2.2a

**The agent-editable schedule IR is no longer decorative.** Phase 1 built
``edits.py``/``gate.py`` and unit-tested their decidability, but nothing *used*
them — the GEMM ran at one hardcoded config. Phase 2.2a wired them into a real
autotune loop that took the DSL GEMM from 25% to 97% of cuBLAS, recorded the
winner + the full search history to the committed card, and validated the whole
path with the substrate's own ``verify``. The 25%→70% roofline gap that §11
named as "the designed next step" is, for the dense-GEMM worked example,
substantially closed by the designed lever (schedule IR + autotune) — not by a
native override. Native overrides (Phase 2.1) are redirected to where they
actually compound.

---

## 15. Phase 2.2b + 2.3 + 2.1 outcome (shipped 2026-06-30)

**Status: DONE — the cost model predicts before measuring, the gate rejects
scratch-overflow configs before launch (the 2.2a kernel crashes are now clean
rejects), and the per-target override mechanism lands (CPU-doable half). Three
phases in one pass because they share a foundation: ``cost.py`` (2.3's module) is
the scratch-footprint predictor 2.2b's gate needs, AND the roofline/occupancy
2.3's gate needs; Phase 2.1's override mechanism is independent. 123 GPU tests
green (20 cost + 9 sweep + 8 override + 86 prior); ruff clean; the committed card
carries a real ``perf.measured`` + ``tuning_trace`` + the Phase 2 gate verdict.**

### Phase 2.2b — the gate's cost-model half (scratch overflow, predicted)

The 2.2a sweep had two 4096³ FAILs: ``BLOCK_M=BLOCK_N=256, BLOCK_K=64,
num_stages=4`` crashed with a smem overflow (256 KB > 228 KB budget). The fix is
not in the launcher — it's in the gate. ``cost.overflows_scratch(pattern, config,
dtype, arch)`` is the closed-form predictor: ``tiled_2d`` scratch =
``num_stages × (BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N) × dtype_bytes``; the gate now
checks it before launch.

**Validated on H100:** the re-run sweep reported ``overflow pre-checked: 2 |
crashes: 0`` — the two configs that crashed in 2.2a are now clean trace entries
with ``reason: scratch overflow: 256 KB > arch budget``. The decidable gate is
no longer decorative for tiles; it bites. (The smem-overflow case is the
``AddStage`` scratch-budget check the Phase 1 ``edits.py`` was built for, now
wired through ``cost.predict_scratch`` instead of the un-annotated Phase 1 stub.)

### Phase 2.3 — ``cost.py`` + the formal roofline gate

``cost.py`` (~340 LOC, new) composes with the substrate's ``cost_model.py``
(no-touch: DSL ops carry their models HERE, hand ops delegate). Four signals:

- **workload** — ``workload(op_id, point) -> (flops, bytes)``. DSL ops (``gemm_bf16``
  ``_gemm_bf16``, ``dual_rmsnorm``) in an vkl table; hand ops via the substrate's
  ``cost_model``. Validated: ``2*M*N*K`` flops + ``2*(MK+KN+MN)`` bytes for the
  GEMM (bf16).
- **roofline** — ``roofline(op, point, arch, instr) -> Roofline{tflops, ceil, ai,
  bottleneck}``. At 4096³ bf16: AI=1365, compute-bound, ceiling 989 (wgmma). The
  ``bottleneck`` label is the skill-routing signal (memory→diagnose-memory-bound;
  compute+low-occ→diagnose-low-occupancy).
- **occupancy** — warps/SM from smem pressure + warps/CTA (the closed-form half;
  the register half is profile-calibrated, honestly ``unknown`` cold-start). On
  the 2.2a winner: 8 warps/SM, smem-limited (192 KB / 228 KB → 1 CTA/SM × 8 warps).
- **roofline_gate** — ``roofline_gate(measured_ms, op, point, arch, instr, frac=0.70)
  -> GateVerdict``. The §2 decision rule. On the 2.2a winner (461 TFLOPS): **46.6%
of the 989 wgmma ceiling → BELOW_BAR** — the honest verdict. cuBLAS reaches 48%
of the same ceiling on this shape, so the autotuned Triton GEMM is at 97% of
*cuBLAS* but below the *theoretical* bar; the verdict triggers the Phase 2.1
native-override conversation, recorded in the card's tuning_trace (``_gate``
entry) so the next agent reads "don't bother re-sweeping; this backend is at its
practical ceiling — go native or accept A2."

The gate verdict is now part of every ``SweepResult`` and persisted to the
committed card's ``provenance.tuning_trace`` (the compounding loop compounds the
*verdict*, not just the configs).

### Phase 2.1 — the override MECHANISM (CPU-doable half; GPU codegen TBD)

The full Phase 2.1 (native CUDA/CUTE + HIP/CK codegen reaching the vendor
ceiling) is GPU-gated and **environment-blocked** on this node (system has only
CUDA 13.0; torch is cu124; no root/module-system to install cu12 nvcc). What IS
CPU-doable — and load-bearing — is the *mechanism*, now shipped:

- **``@gemm.target("cuda", arch="nvidia_sm90")``** — the per-target override
decorator (``surface.py``). Attaches an ``OverrideBody`` to the ``KernelSpec``;
``KernelSpec.override_for(backend, arch)`` resolves the most-specific match.
- **``check_override_math_ir(spec, override)``** — the oracle-property gate
(``override.py``). An override must build the SAME math IR (same op-kind
signature: Load/MMA/Reduce/Pointwise/Store) as the portable body. A body that
drops the MMA (computes a different op) is rejected with "route to
author-an-op-spec". **This makes the oracle property *enforced*, not hoped-for**
— the single most load-bearing invariant in the whole design (§10 anti-goals).
- **``emit_override_card(spec, override)``** — projects an override to its own
schema-valid Impl Card (cuda/hip backend; ``arch.requires`` the native features:
tensor_cores+tma+clusters on sm_90, matrix_cores+mfma on cdna3; ``wave_size`` 32
vs 64; ``scratch.kind`` smem vs lds; ``provenance.derived_from`` the portable
Triton card).

The GPU half (``lower/cuda.py`` wgmma+TMA+cluster codegen, ``lower/hip.py`` MFMA
codegen) is the future Phase 2.1 work; it lands on top of this mechanism
unchanged. The decorator + invariant + card emission are the foundation.

### What was learned (load-bearing gotchas)

- **A shared GPU node makes autotune non-deterministic.** The 2.2a sweep found
the 460-TFLOPS winner on an unloaded node; a 2.2b re-run under contention (all 4
GPUs at 100% from other users) picked a *different* winner (267 TFLOPS) because
``do_bench`` medians were inflated ~2× and the relative ordering shifted. This is
the autotune-knob-sweep skill's "re-time noisy winners at higher iteration
count" caveat, hit in the wild. **Mitigation:** the committed ``perf.measured``
carries the clean (node-unloaded) measurement with an honest source marker; the
contention run's winner is a distinct point (different config), both legitimate.
- **``verify``'s ``perf.ms`` is a median under current load, not a property of
the kernel.** Two sweeps on the same config can disagree by 2× if the node
changed state between them. Always cross-check against cuBLAS (the sanity floor)
before believing a roofline fraction — and record the *source* honestly (clean
re-time vs contention).
- **The oracle property is a structural invariant, checkable from the IR alone.**
``check_override_math_ir`` compares op-kind signatures — no GPU needed. This is
why the math IR's small algebra (~6 node kinds) is load-bearing: it makes "same
computation" decidable. An override that adds a Reduce the portable body lacks
fails the check; an override that spells the MMA with wgmma instead of tl.dot
passes (same op kind). The check is the difference between "the override is
verified" and "we hope the override is equivalent."
- **``provenance`` is ``additionalProperties: false``** — the override card can't
carry an ``override_kind`` field; ``derived_from`` (already schema-legal) records
the derivation. Schema constraints shape what the emitter may emit; don't fight
them.
- **The cost model is only useful if it would have predicted the outcome.** Every
``cost.py`` test asserts against a Phase 2.2a MEASURED number (the two overflow
FAILs, the 460-TFLOPS winner, the 47% gate fraction). A cost model validated only
against its own predictions is circular; validating against prior measurements
is the honest bar.

### The headline for 2.2b/2.3/2.1

**The DSL now has the three pillars the design named as load-bearing, all live:**
(1) a decidable gate that rejects bad configs before launch (not after a crash);
(2) a cost model that predicts the outcome + the skill to fire next + the formal
70%-roofline Phase 2 decision rule; (3) an override mechanism with an *enforced*
oracle property. What remains GPU-gated is the native codegen itself (Phase 2.1
GPU half) — and that's now unblocked to land on a stable, validated foundation
whenever the cu12-nvcc environment issue is resolved or an AMD node is available.

## 16. Phase 2.1 GPU half + Phase 3 outcome (shipped 2026-07-01)

The native-codegen + graph-capture phases, both closed on real hardware via the
**ds5** box (aarch64 DGX Spark, NVIDIA GB10 / sm_121) and cross-checked on
**sgs-gpu07** (H100 / sm_90). All transport now via **rcc** (the remote cluster
controller), which killed both rsync footguns (the `--delete` that wiped
tracked scripts, and the multi-source-dir error): `.rcc/config.toml` carries the
ds5 + sgs-gpu07 profiles; `.rcc/rccignore` carries the exclusions.

### Phase 2.1 GPU half — native CUDA override (`vkl.lower.cuda`)

`@gemm_bf16.target("cuda", arch="nvidia_sm121")` + `lower/cuda.py` lowers the
override's math IR to a REAL native nvcc kernel (via `load_inline`), registered
as the `cuda` backend. The oracle invariant holds (same `[Load,Load,MMA,
Pointwise,Store]` signature as the portable body, checked by
`check_override_math_ir`). Both bf16 and fp32 sweep points PASS `verify` on GB10.

**The big correction (a misdiagnosis overturned on the chip).** The override was
motivated by "triton 3.6 degrades fp32→tf32 on Blackwell." Measuring every
compute path directly on GB10 refuted it:
```
cpu_true vs out_triton : 0.000000   ← triton kernel IS true fp32 (bit-exact)
cpu_true vs ref_raw    : 0.004260   ← the REFERENCE was tf32
```
The triton kernel was never the bug. `run_reference` ran `torch.matmul` on GPU
with the global `allow_tf32=True`, silently making the **oracle itself tf32** —
so a correct true-fp32 kernel "failed" against a tf32 reference. **Fix at the
source: `run_reference` now disables tf32 for the duration of eval** (the oracle
must be exact; only kernels approximate). This took ds5 from 120 pass + 3 fail
to **135 pass / 1 skip**. The native override's honest value today is
**mechanism validation** (it compiles, registers, passes verify on a real chip) —
not a correctness fix over triton (both do true fp32). Corrected all stale
"fixes the tf32 divergence" claims in the docstrings.

The native kernel launches on `at::cuda::getCurrentCUDAStream()` (not raw stream
0) — the stream-discipline fix for a latent read-before-write hazard that
CUDAGraph capture's stream changes expose (intermittent garbage output;
`max_abs_err` 23.4 at bf16 512³ when it triggered).

### Phase 3 — graph capture (`vkl.graph`)

`graph.py` captures a composition of `register_dsl`-ed kernels into ONE
`torch.cuda.CUDAGraph` replay via explicit construction (works on both NVIDIA
and AMD with no native ext — the nodes ARE DSL launchers). The `@graph` body
declares *what* composes (`ctx.call(name, ...)` per node); `capture()` records
it; `replay(new_inputs)` copies params into static buffers (the §4.2
parameter-node discipline — one graph serves many args) and replays.

**The §8 perf gate, measured honestly on both archs** (3-node GEMM chain,
`examples/gemm_chain.py`):

| shape regime | GB10 (sm_121) | H100 (sm_90) | §8 verdict |
|---|---|---|---|
| small (launch-bound) | **6.51×** WIN | **5.02×** WIN | graphs win big |
| medium | 1.83× WIN | 2.95× WIN | graphs win |
| large (compute-bound) | 0.98× LOSS | 0.90× LOSS | graphs lose (correctly) |
| correctness \|cap−seq\| | 0.00 | 0.00 | bit-exact |

This is the §8 honesty table made real: graphs win exactly where the doc says
(small/launch-bound chains) and lose exactly where it says they shouldn't
(large/compute-bound). `measure()` reports the honest ratio; the test gate
asserts the win only on the launch-bound regime.

**The §4.3 conditional-node boundary (open question B3).** A captured graph is a
static DAG; host-side `if` on device data is a GPU→CPU sync, illegal under
capture. Probed via `test_vkl_graph_conditional.py`: capture is **rejected**
(`cudaErrorStreamCaptureInvalidated`), never silently degraded — the honesty
rule holds. The wrinkle: an invalidated capture **poisons the CUDA context
process-wide** (verified across `capture_error_mode` global/thread_local/relaxed
— none recover). So the provoked-capture test runs in a **subprocess**: the
child asserts the rejection and its poisoned state dies with it. The honest v1
boundary: **dense chains only**; conditional nodes ship when torch/CUDA grows
`cudaGraphAddCond` support (the test is the canary — it flips to "captures").

### What was learned (load-bearing gotchas)

- **The oracle must be exact, and "exact" means disabling tf32 in the
  reference path.** A GPU `torch.matmul` with the global `allow_tf32=True` is
  tf32, not fp32. If the reference runs on GPU, it inherits that. Pin the oracle
  to true fp32; let the kernels approximate. Diagnosing "the kernel is wrong"
  when the REFERENCE is wrong wastes a whole override phase — always measure
  kernel-vs-CPU-true directly before blaming the kernel.
- **Native CUDA kernels must launch on `getCurrentCUDAStream()`, not stream 0.**
  Raw `<<<>>>` binds stream 0; torch ops use the current stream. After
  CUDAGraph capture (or any non-default-stream work) they diverge → read-before-
  write hazard → intermittent garbage, worse under contention. This is invisible
  in isolation (same stream) and only surfaces under graph-capture ordering.
- **An invalidated graph capture is process-fatal.** Once a sync hits during
  `torch.cuda.graph(g)`, `cudaErrorStreamCaptureInvalidated` poisons the whole
  context — `empty_cache`, `synchronize`, stream reset, and every
  `capture_error_mode` all fail to recover. Isolate provoked-capture tests in a
  subprocess; never let one poison the pytest process.
- **Captured graphs must be `close()`d.** `torch.cuda.CUDAGraph` holds
  capture-pool state; `close()` drops the graph + static buffers + `empty_cache`.
  Without it, `measure()`'s graphs leak pool state into later tests
  ("Offset increment outside graph capture").
- **rcc over raw rsync.** `rcc --profile <host> push` is non-destructive by
default (no `--delete` wiping tracked scripts); `push <subpath>` handles
multi-file transfer (no multi-source-dir error); `.rcc/rccignore` owns
exclusions. Both GPU boxes are rcc profiles now.

### The headline for 2.1-GPU + 3

The DSL now produces: (1) a **native CUDA kernel** from a per-target override
(compiled by nvcc on the chip, verified against the exact oracle) — the
mechanism that carries future CUTLASS/wgmma ceiling work; (2) a **captured
CUDA/HIP graph** from a composition of DSL kernels (one replay beats N launches,
measured honestly, wins only where the doc says it should). Both are live on two
archs (GB10 sm_121 + H100 sm_90), 135/131 pass, bit-exact correctness. The
override's native ceiling (tensor cores) and the graph's conditional nodes are
the documented open boundaries — honest scope, not silent gaps.
