---
name: author-a-kernel-with-dsl
description: >
  Author a NEW op's WHOLE contract (Op Spec + backend-neutral reference + Impl
  Cards + shape sweep) from ONE `@kernel` source in the vkl DSL — the Vibe Kernel
  Language (`src/xkernels/vkl/`). One body builds the frozen math IR that lowers
  to BOTH torch (the auto-reference, bit-exact) AND a generated Triton device
  kernel, so the spec + reference + cards are EMITTED, not hand-written across
  eight artifacts. This is the DSL fast-path twin of `author-an-op-spec` (the
  hand path): same CPU-satisfiable gate, far less boilerplate, but only for ops
  the math IR can express (pointwise / reduce / MMA — gemm / norm / reduce /
  activation categories). If the op is NOT math-IR-expressible (custom softmax
  masking, scatter/gather, collectives), fall back to `author-an-op-spec`. The
  DSL is a spelling of the contract, NEVER a gatekeeper (docs/brainstorm/02 §1,
  §10 anti-goals).
license: Apache-2.0
x-kernel-lib:
  id: author-a-kernel-with-dsl@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "task asks to add a NEW op AND the op is expressible as pointwise/reduce/MMA over the inputs (gemm / norm / reduce / activation / elementwise — the §9 categories the math IR covers)"
      - "an agent wants the spec + reference + Triton card from one source (the @kernel body), not hand-written across registry/ops + ops/<type>/reference.py + ops/<type>/triton/*.py + input_gen.py"
      - "no GPU is available but a CPU-verifiable reference + a Triton card that will compile on GPU are both wanted (the body IS the auto-reference; emit is free)"
      - "an op has a hand-written spec already and the task is to consolidate it into one @kernel source (the emitted artifacts must round-trip the existing spec's fields)"
    preconditions:
      - "no Op Spec exists yet for this op name, OR an existing hand spec is being consolidated (this skill REPLACES the hand authoring of spec+reference+cards with one source; do not hand-write the cards it emits)"
      - "the op IS math-IR-expressible — i.e. its computation is a fixed DAG of load/store/mma/reduce_sum/pointwise(*/+//-/cast/rsqrt). If it needs data-dependent control flow, masking, scatter/gather, or collectives, route to author-an-op-spec (the hand path) instead."
      - "the numerics story is known: input dtypes, the fp32 (or wider) accumulation dtype, and an honest tolerance source (issue doc / reference library / empirical bar) — the header's Numerics carries these 1:1"
  inputs_required:
    - "op name (and version, default 1.0.0)"
    - "canonical_op (the retrieval key: gemm | norm | reduce | activation | ...)"
    - "the math of the op, as a fixed DAG of the math-IR ops above"
    - "the launch pattern: rowwise (one program per leading-dim row, reduction over the last axis) or tiled_2d (2D output grid + K-loop). These are the only two patterns today (Phase 2.0b)."
  tools:
    - xkernels.vkl.kernel
    - xkernels.vkl.emit_spec
    - xkernels.vkl.emit_card
    - xkernels.vkl.emit_reference_card
    - xkernels.vkl.register_dsl
    - xkernels.vkl.run_reference
    - xkernels.vkl.autotune
    - xkernels.verify
  validation:
    must_pass:
      - "emit_spec(spec) is schema-valid (validate_op_spec passes) and round-trips the header fields 1:1"
      - "emit_reference_card(spec) + emit_card(spec, spec.targets['triton']) are schema-valid (validate_impl_card passes)"
      - "run_reference(spec, make_inputs(spec, point)) executes on torch (CPU) — the body IS the auto-reference; this is the CPU correctness gate"
      - "verify('<op>.triton@<ver>', ...) on a GPU returns compiled=True, correctness.passed=True (the generated Triton kernel vs the auto-reference); the generated kernel runs with zero JSON hand-editing"
      - "the body's cast/reduction order matches the reference formulation you intend — the auto-reference is bit-exact with the BODY'S arithmetic, so the body must encode the numerics you want (fp32 reduction before cast, etc.)"
    # NOTE: like author-an-op-spec, this skill's emit+reference gate is CPU-
    # satisfiable. The Triton-compile gate needs a GPU. See the no-GPU branch.
  references:
    - "src/xkernels/vkl/examples/dual_rmsnorm.py (the rowwise worked example)"
    - "src/xkernels/vkl/examples/gemm_bf16.py (the tiled_2d worked example)"
    - "docs/brainstorm/04-strawman.md (the two-source example the DSL operationalizes)"
    - "docs/brainstorm/02-core-idea.md §1 (one computation, two lowerings)"
    - "meta/docs/library.md §10 (the DSL is NEVER a gatekeeper; portability lives in the contract)"
    - "author-an-op-spec/SKILL.md (the hand-path twin — fall back here when the op is not math-IR-expressible)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-06-30T00:00:00Z"
    supersedes: []
---

## Why this skill exists

`author-an-op-spec` (the hand path) is correct and complete, but it spreads one
op across **eight hand-written artifacts** in four paradigms: the Op Spec
(`registry/ops/<op>.spec.json`), the reference
(`src/xkernels/ops/<type>/reference.py`), the shape sweep
(`registry/shape_sweeps/<op>.sweep.json`), the input generator (a literal in
`input_gen.py`), and one Impl Card per backend (`registry/impls/<op>.*.card.json`).
Each must be kept consistent with the others by hand — and the #1 silent harness
failure is a mismatch between the spec's `inputs` and the backend callable.

The vkl DSL collapses that to **one `@kernel` source**. The header projects 1:1
to the Op Spec; the body builds a frozen **math IR** (`load`/`store`/`mma`/
`reduce_sum`/pointwise) that **two interpreters** lower over the SAME nodes:
torch (the auto-reference) and a generated Triton kernel (the device card). The
computation is written once; "the reference matches the kernel" becomes a
structural property, not a hope. `emit_spec` / `emit_card` / `emit_reference_card`
project the one source to the schema-valid JSON the substrate already consumes —
the cards are contract-identical to hand-written ones (`provenance.authored_by:
"dsl"` is the only tell).

This is the **gateway fast-path** when the op is math-IR-expressible. Its gate,
like `author-an-op-spec`'s, is CPU-satisfiable (emit + validate + the body-as-
reference runs on torch). It is NOT a gatekeeper (§10): if the op needs something
the math IR cannot express, fall back to the hand path — the two skills are
peers, and the contract they produce is interchangeable.

## The routing decision: DSL vs hand path (decide BEFORE step 1)

The math IR today expresses a fixed DAG of these ops (verify against this list):

| math-IR op | `ctx` call | covers |
|---|---|---|
| load | `ctx.load("a")` | read an input by name |
| store | `ctx.store("out", val)` | write an output by name |
| MMA | `ctx.mma(a, b, accum_dtype="fp32")` | matrix-multiply-accumulate (GEMM inner product) |
| reduce_sum | `ctx.reduce_sum(x, axis=1, accum_dtype="fp32")` | sum reduction over an axis |
| pointwise | overloaded `* + /`, `val.cast(dtype)`, `ctx.rsqrt(x)`, `ctx.lit(c)`, `ctx.dim(t, axis)` | scale / bias / cast / activate |

**Use this skill (DSL path)** if the op is a fixed DAG of the above. Today that
covers the **gemm, norm, reduce, activation** §9 categories cleanly (see the two
worked examples). **Fall back to `author-an-op-spec` (hand path)** if the op
needs anything outside this list — e.g.:

- **attention with custom masking** (data-dependent control flow, online-softmax)
- **scatter / gather / indexing** (`out[idx] = ...`, top-k, MoE routing)
- **collectives** (all-reduce, all-to-all — `ops/comm/`)
- **fp8/int4 dequant with block scales** not expressible as a pointwise cast
- **a fusion whose output shape/semantics change** (route to `author-an-op-spec`
  first, then `add-epilogue-fusion` — the DSL emits the UN-fused base)

> **Caveat — the data-ADDRESSING family (`gather`/`slice`/`concat`/`unsqueeze`)
> verifies on device, but its codegen had a modulo-sign gotcha (now fixed).**
> These nodes (docs/brainstorm/06 A4 case (a)) lower via `_TritonGenMultiDim`, a
> flat-tiled grid that decomposes each lane's offset into per-axis coords and
> lets every node compute its own address. `apply_rope` (#68) is the worked
> example and is now verified on GB10 (`verify` 5/5, parity agrees). The ONE
> gotcha to know: the codegen emits `coord % shape` for broadcast indices, and a
> `Concat` b-branch shifts its coord *negative* (`c{ax} - len_a`) — so the modulo
> MUST be floored (`((x%n)+n)%n`), because CUDA `%` follows C sign (`-1%64 ==
> -1`, not Python's `63`). That bug (now fixed via `_floored_mod` in
> `lower/mathbody.py`; diagnosis: `meta/docs/wiki/04-gotchas.md` §14) was the
> one that made an earlier session wrongly declare the addressing family
> "unverifiable on device." It is verifiable; just floor the modulo. The
> pointwise/reduce/MMA categories (gemm/norm/reduce/activation) are unaffected
> (their coords are always non-negative).

If you are unsure, sketch the op as the DAG above on paper; if every arrow maps
to a row in the table, the DSL path is faster.

## Procedure

1. **Pick the launch pattern.** It decides how the math IR maps to programs. The
   two patterns today (Phase 2.0b):
   - `Launch.rowwise()` — one Triton program per leading-dim row; reductions run
     over the **last axis**. Use for **norm / layernorm / rmsnorm / activation /
     any per-row reduce** (see `examples/dual_rmsnorm.py`).
   - `Launch.tiled_2d()` — a 2D program grid over the output's two leading dims;
     the MMA's contracted dim becomes the **K-loop**. Use for **GEMM and any
     2D-output inner-product** (see `examples/gemm_bf16.py`).

2. **Author the header (`@kernel(...)`).** Every field projects 1:1 to the Op
   Spec — copy the field names from an example. The load-bearing ones:
   - `inputs` / `outputs` — `TensorDecl(rank=, dtype=, symbols=...)`. `symbols`
     is the shape-symbol tuple (`("M","K")`); the math IR's `dim(tensor, axis)`
     resolves these symbolically (they are NOT python ints).
   - `constraints` — the decidable mini-language only (`K % 16 == 0`,
     `dtype(a) == dtype(b)`); same rules as the hand spec (§1.3.2). A good set
     rejects bad shapes from metadata alone.
   - `numerics=Numerics(rtol=, atol=, reduce_dtype=, ...)` — cite an honest
     tolerance source in `notes`; do NOT set `reference` (it defaults to
     `AUTO_REFERENCE`, meaning the body — overriding it defeats the whole point).

3. **Author the body.** The body takes ONLY `ctx` and references every input and
   output **by name string**. Build the math IR with the `ctx` calls in the table
   above. Three rules that are load-bearing for bit-exactness:
   - **Cast to the accumulation dtype BEFORE the heavy op.** `ctx.mma(...,
     accum_dtype="fp32")` and `ctx.reduce_sum(..., accum_dtype="fp32")` make the
     fp32 (or wider) accumulation explicit — this is the numerical invariant that
     matches the hand reference (`a.float() @ b.float()`; variance of fp32
     squares). Forgetting it silently degrades to input-precision accumulation.
   - **Mirror the reference's cast order at the output.** The auto-reference IS
     the body, so the body must encode the numerics you intend. If the reference
     is `(x*inv).to(out_dtype) * w` (cast before the weight multiply), spell it
     that way — `(v*inv).cast(ctx.out_dtype()) * ctx.load(w)` — not `v*inv*w`
     then cast. Order changes ULPs; the gate is "body matches your intent," and
     your intent is the reference formulation.
   - **`ctx.dim(tensor, axis)` for the reduction width, not a python int.** The
     reduction width is symbolic in the input's shape symbol; `dim` emits the
     node that lowers to `tl.num_programs` / a runtime value. Hardcoding `d`
     breaks the sweep (the IR is shape-symbolic).

4. **Stack the decorators bottom-up: `@targets(...)`, then `@launch(...)`, then
   `@kernel(...)`.** Python applies decorators bottom-up, so `@kernel` (topmost)
   runs last and must READ the metadata the lower decorators attached. Get the
   order wrong and `@kernel` won't see the `@launch`/`@targets` — a silent
   "no lowering" failure. Declare at least a `triton` target; its `knobs` dict is
   the **autotune search space** that becomes the card's `specialization_knobs`.

5. **Emit the artifacts.** From the spec:
   ```python
   from xkernels.vkl import spec_of, emit_spec, emit_reference_card, emit_card
   spec = spec_of(<your kernel fn>)
   emit_spec(spec)              # -> registry/ops/<op>.spec.json
   emit_reference_card(spec)    # -> registry/impls/<op>.reference.card.json  (MANDATORY)
   emit_card(spec, spec.targets["triton"])  # -> registry/impls/<op>.triton.card.json
   ```
   Write each dict to its registry path. The **reference card is mandatory**
   (the registry invariant `test_every_seeded_op_has_reference_and_sweep` requires
   one spec + a reference card + per-backend cards); the DSL gives it to you free
   because the body IS the reference — do not skip it.

6. **Run the CPU gate (no GPU needed).**
   ```python
   from xkernels.registry.schemas import validate_op_spec, validate_impl_card
   from xkernels.vkl import run_reference, make_inputs
   validate_op_spec(emit_spec(spec))
   validate_impl_card(emit_reference_card(spec))
   validate_impl_card(emit_card(spec, spec.targets["triton"]))
   out = run_reference(spec, make_inputs(spec, {"dtype":"bf16", ...}, device="cpu"))
   ```
   All three validate, and `run_reference` executes the body on torch. This is
   the correctness gate available on a CPU-only box — the same one
   `author-an-op-spec` ships. If `run_reference` raises, the body has a bug (a
   misspelled input name, a cast on a non-existent dtype, a reduce over an axis
   the launch pattern can't lower).

7. **Write the shape sweep** at `registry/shape_sweeps/<op>.sweep.json`
   (`default_dtype` + `points`). Same rules as the hand path: cover a tiny leading
   dim, a non-power-of-2 size, the divisibility boundary of your hardest
   constraint, and an fp32 point. The DSL does not author the sweep for you — it
   is a data file the spec references by name (`shape_sweep="<op>"`).

8. **On a GPU, close the loop.** Register the generated Triton kernel and verify:
   ```python
   from xkernels.vkl import register_dsl
   from xkernels import verify
   register_dsl(spec, backend="triton")          # registers dispatch + input gen
   r = verify("<op>.triton@1.0.0", arch="nvidia_sm121")
   assert r["compiled"] and r["correctness"]["passed"]
   ```
   `register_dsl` wires BOTH the dispatch callable AND the input generator (from
   the spec's shape symbols), so `verify` runs end-to-end with zero hand-editing
   of `input_gen.py`. The generated Triton kernel is checked against the
   auto-reference — the structural "reference matches kernel" property, measured.

   **Run it on ds5 (rcc + docker).** The above is a device call — execute it on
   the GB10 test bed (`arch="nvidia_sm121"`) via rcc's Docker target:
   ```bash
   rcc --profile ds5 push
   rcc --profile ds5 run --docker -s 'python - <<PY
   from xkernels.vkl import register_dsl, spec_of
   from xkernels.vkl.examples import <your_module> as m   # the file holding @kernel
   from xkernels import verify, verify_parity
   register_dsl(spec_of(m.<your_kernel>), "triton")
   r = verify("<op>.triton@1.0.0", arch="nvidia_sm121", measure_perf=True)
   print(r["correctness"]["passed"], r["correctness"]["max_rel_err"], r["perf"]["ms"])
   print("parity", verify_parity("<op>@1.0.0", archs=["nvidia_sm121"])["agree"])
   PY'
   ```
   `--docker` runs in the NGC container (`PYTHONPATH=/workspace/src` auto-set —
   no venv needed); `-s` takes a shell snippet (heredocs/pipes ok). ds5 is NVIDIA
   only; the AMD/gfx942 ceiling runs on beverin
   (`scripts/cluster.sh run --host beverin`). Stand-up + internals:
   `meta/docs/usage/ds5-testbed.md`.

9. **Autotune (GPU).** The `Target.knobs` you declared is the search space:
   ```python
   from xkernels.vkl import autotune
   res = autotune("<op>.triton@1.0.0", arch="nvidia_sm121",
                  point={"dtype":"bf16","M":4096,"N":4096,"K":4096})
   ```
   `autotune` enumerates the configs, gates each through the scratch-budget
   predictor (rejecting smem/lds overflows BEFORE launch — no crash), times the
   survivors via `verify(measure_perf=True)`, and writes the winner to the card's
   `perf.measured` + the full history to `provenance.tuning_trace` (the
   compounding loop). Hand off to the `autotune-knob-sweep` skill for the full
   procedure.

10. **Native-ceiling override (GPU, only if Triton can't reach the ceiling).** If
    the roofline gate reports `BELOW_BAR` (the winner is under ~70% of the arch's
    instruction ceiling), attach a per-target override body:
    ```python
    @<your kernel>.target("cuda", arch="nvidia_sm90")
    def kernel_cuda(ctx):
        # SAME math-IR signature as the portable body, spelled for native
        # (TMA + wgmma + clusters). The oracle-property gate enforces this.
        ...
    ```
    `check_override_math_ir(spec, override)` is the CPU-doable gate: the override
    must build the SAME op-kind signature as the portable body, or it is a
    different op (route to `author-an-op-spec`). `emit_override_card` projects it
    to its own card. See `map-to-matrix-cores` / `port-across-arch` for the
    native codegen the override body ultimately needs.

## The honest no-GPU branch (read this if there is no GPU)

A CPU-only box satisfies **every must_pass through step 7**: the emitted spec +
cards are schema-valid, and the body runs as its own auto-reference on torch. The
Triton card will honestly read `compiled=False` in `verify` (no GPU to compile
on) — that is the correct, publishable state for the contract layer, identical to
`author-an-op-spec`'s no-GPU branch. The generated Triton kernel source exists
and will compile unchanged once a GPU is available; `register_dsl` + `verify` at
step 8 are the only GPU-gated steps. Do not set `compiled: true` to "look done".

## Pitfalls

- **The body takes ONLY `ctx`; inputs are referenced BY NAME.** A common copy-
  paste error is `def body(x1, w1, ctx)` — the signature is `def body(ctx)`, and
  inputs are `ctx.load("x1")`. Extra parameters are silently ignored; the IR
  never sees them and `run_reference` raises on the missing input.
- **Forgetting `accum_dtype="fp32"` on the heavy op.** `ctx.mma(a,b)` without it
  (or `reduce_sum` without it) compiles but accumulates in input precision — the
  numerics drift, and the tolerance that was calibrated for fp32 accumulation no
  longer holds. The fp32 (or wider) accumulation is the load-bearing invariant;
  spell it every time.
- **Hardcoding a shape int instead of `ctx.dim`.** `mean = ss / 1536` "works" on
  one sweep point and silently fails the sweep at `d=4096`. The IR is shape-
  symbolic; `ctx.dim(x, axis=1)` emits the node that lowers to the runtime width.
- **A cast order that differs from the reference you intend.** The auto-reference
  IS the body, so "the body is wrong" and "the reference is wrong" are the same
  bug. If the published reference casts before the weight multiply, spell it that
  way; the bit-exact torch eval will follow whatever DAG you built.
- **Overriding `numerics.reference`.** It defaults to `AUTO_REFERENCE` (the
  body). Pointing it at a hand-written reference re-introduces the exact
  mismatch risk the DSL eliminates. Only override it if you are consolidating a
  hand op and the body genuinely cannot be the reference yet.
- **Skipping the reference card.** `emit_reference_card` is not optional — the
  registry invariant requires one spec + a reference card + per-backend cards.
  Because the body IS the reference, emitting it is free; skipping it breaks
  `test_every_seeded_op_has_reference_and_sweep`.
- **Declaring a `Target.knob` the launcher can't accept.** The launcher accepts
  `BLOCK_M/N/K`, `num_warps`, `num_stages` (the tile + pipeline knobs). Declaring
  a knob outside that set makes the card's `specialization_knobs` dishonest (the
  harness reports it unapplied, §1.2). Same pitfall as the hand path.
- **Using the DSL for an op it can't express, then hand-patching the emitted
  kernel.** If `run_reference` or the generated Triton source needs a fix the
  math IR can't carry (masking, gather), you are on the wrong skill — fall back
  to `author-an-op-spec` and write the reference + kernel by hand. The emitted
  kernel is generated; hand-editing it is discarded on the next emit.
- **An override body that computes a different op.** `check_override_math_ir`
  will reject it (op-kind signature mismatch) with "route to author-an-op-spec".
  An override is MORE native code for the SAME computation (wgmma instead of
  tl.dot), not a DIFFERENT computation. If you genuinely need a different op
  (extra output, fused residual), that is a new Op Spec.
- **Decorator order.** Bottom-up: `@targets` and `@launch` attach metadata;
  `@kernel` (topmost) consumes it. Reorder and `@kernel` builds a spec with no
  lowering — the failure is "no @launch declared" at `register_dsl` time.
