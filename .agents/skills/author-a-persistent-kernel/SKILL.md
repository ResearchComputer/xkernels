---
name: author-a-persistent-kernel
description: >
  Author a whole-sub-block MEGAKERNEL — one launch running heterogeneous
  ops (GEMMs, attention, norms, activations, residuals) with intermediates
  kept ON-CHIP (registers/LDS) across a persistent grid, killing each DRAM
  round-trip. The MOST aggressive fusion and LAST RESORT (tension with
  library.md §10): fires ONLY after (i) graph capture (launch.graph) is in
  place, (ii) narrow fusion (add-epilogue-fusion / fuse-elementwise-chain)
  is exhausted, AND (iii) a profile (rocprof/ncu) shows the on-chip-staged
  regime winning — usually decode-only (M=1..few). The on-chip dataflow
  contract is declared via the PINNED `persistent` schema block
  (residency, warp_roles, budgets, pipeline_stages, defused_edges).
  GPU-gated (compile + verify); contract + budget math is CPU-doable.
  Attention can't be DSL-emitted (online softmax, causal masking) — a
  whole-layer megakernel is HAND-WRITTEN NATIVE (CUDA/HIP), gated on
  blocker (c). Use when an issue or profile demands a true persistent
  megakernel.
license: Apache-2.0
x-kernel-lib:
  id: author-a-persistent-kernel@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "an issue or benchmark explicitly requests a whole-layer / whole-sub-block persistent megakernel (not a single epilogue fusion)"
      - "a profile (rocprof/ncu) on a decode-regime workload (M=1..few) shows launch/latency + intermediate-DRAM-bandwidth as the dominant cost AFTER graph capture is already in place"
      - "the interim path (graph capture + add-epilogue-fusion + fuse-elementwise-chain) has been measured and STILL leaves a gap the on-chip-staged regime would close"
    preconditions:
      - "graph capture (launch.graph) of the per-op cards is ALREADY in place and measured — the megakernel must beat THAT baseline, not the naive launch-per-op baseline"
      - "a profile exists naming the specific DRAM round-trips the megakernel would eliminate (not a vibe — §1.3.2 measure-don't-guess)"
      - "the target shape regime is identified: decode (M small, latency-bound — megakernel favored) vs prefill (M large, compute-bound — megakernel usually LOSES to tiled separate GEMMs at the roofline)"
  inputs_required:
    - "the ordered list of ops in the sub-block (e.g. dual_rmsnorm -> qkv -> qk_norm -> rope -> paged_attention -> o_proj -> residual_add)"
    - "target arch + dtype + the decode shape regime (M=1..few) where the megakernel is claimed to win"
    - "a profile (rocprof/ncu) of the graph-captured baseline naming the DRAM round-trips to eliminate"
  tools:
    - get_impl_card
    - get_op_spec
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "the megakernel compiles and verify(<persistent_card>).correctness.passed == true on the target arch (GPU-gated)"
      - "the persistent block declares the on-chip contract honestly: residency (which intermediates never hit DRAM), defused_edges (which do), warp_roles, register_budget, lds_budget, pipeline_stages — all filled, none hand-waved"
      - "the budget MATH is shown: sum of staged-tile bytes <= arch scratch budget (vkl/cost.py predict_scratch); regs/thread <= the register half of occupancy (profile-calibrated). If an edge had to be defused (written to DRAM) because it didn't fit, it is named in defused_edges — never silently spilled"
      - "the card's perf.ms < graph-captured-baseline perf.ms on the target shape regime (else the megakernel wasn't worth it — route back to narrow fusion)"
      - "the card is shape-GATED: its applicability constraint restricts it to the regime where it wins (decode); it must NOT be offered for prefill where tiled GEMMs win"
      - "verify_parity agrees against the shared reference (the megakernel's numerics must match the graph of separate ops at the op's tolerance — the reference is the graph's reference, composed)"
  references:
    - "meta/docs/design/megakernel-blockers.md (the blocker (b) substrate + (c) native-attention path this skill realizes)"
    - "registry/schema/impl_card.schema.json persistent block (the pinned on-chip dataflow contract)"
    - "src/xkernels/vkl/cost.py predict_scratch / overflows_scratch (the LDS budget gate)"
    - "src/xkernels/vkl/schedule.py Stage(id, producer_ref, space, depth) (the pipeline_stages vocabulary)"
    - "registry/ops/residual_add.spec.json (the on-chip residual step as a first-class contract)"
    - "meta/docs/library.md §10 (the one-mega-kernel-with-a-thousand-flags anti-goal this skill must respect)"
    - ".agents/skills/add-epilogue-fusion/SKILL.md (the NARROWER fusion path to exhaust FIRST)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-07-08T00:00:00Z"
    supersedes: []
---

> **This is the LAST RESORT skill. Before you touch it, confirm the interim path
> is exhausted.** A persistent megakernel is the most fusion you can do and the
> hardest to verify, retune, and reason about applicability for (§10). The
> honest path is: graph-capture the per-op cards (`launch.graph`) to kill launch
> overhead → pick off DRAM round-trips with `add-epilogue-fusion` /
> `fuse-elementwise-chain` → profile → ONLY if a measured gap remains in the
> decode regime does this skill fire. Read `meta/docs/design/megakernel-blockers.md`
> first; it is the design note this skill operationalizes.
>
> **GPU-gated.** The megakernel must compile + `verify` on a GPU. The contract
> (Op Spec), the `persistent` block, and the budget math are CPU-doable; ship
> `compiled:false` honestly on a CPU-only box and defer the kernel + perf gate.

## Procedure

1. **Confirm the last-resort gate (do NOT skip).** A persistent megakernel is
   worth authoring ONLY if all three hold:
   - **Graph capture is already in place and measured.** The baseline the
     megakernel must beat is the `launch.graph` of the per-op cards, NOT the
     naive launch-per-op path. If graph capture isn't done, do that first
     (lighter, no validity-surface explosion).
   - **Narrow fusion is exhausted.** `add-epilogue-fusion` /
     `fuse-elementwise-chain` have already collapsed every chain they honestly
     can. If a short pointwise/reduction op still trails a heavy kernel, fuse
     THAT first — it is strictly easier.
   - **A profile names the gap.** `use-rocprof-compute` / `use-nsight-compute`
     on the decode-regime workload shows intermediate-DRAM bandwidth +
     launch/latency as the dominant remaining cost that ONLY on-chip staging of
     heterogeneous ops would close. If the profile says compute-bound (prefill,
     M large), STOP — tiled separate GEMMs win at the roofline; a megakernel
     there is an anti-pattern.
   If any of these fails, route AWAY: graph capture, or `add-epilogue-fusion`,
   or `fuse-elementwise-chain`. Document the routing decision in the card's
   `notes` or the issue, so the next agent doesn't re-derive it.

2. **Decide the op boundary — it is always a NEW Op Spec.** A megakernel emits
   a multi-op block; it is never a card under an existing single-op spec
   (that would overload the spec's output contract — the `add-epilogue-fusion`
   case-(b) trap, amplified). Run `author-an-op-spec` for the new op FIRST:
   - `canonical_op`: the existing vocabulary has no entry for a persistent
     multi-op block; for now use the **dominant leaf canonical_op** of the block
     (e.g. an attention sub-block → `attention`; an FFN sub-block → `activation`
     or `gemm`) with rich `fusions` tags listing every fused leaf. A dedicated
     `canonical_op` for persistent blocks is an OPEN question (library.md §11);
     do NOT invent one ad hoc. Document the choice in `notes`.
   - `composes_with`: the leaf ops fused in (the `residual_add`, the `rmsnorm`,
     the attention), so the composition graph stays honest.
   - The reference is the **graph of the per-op references, composed** (e.g.
     `residual_add(rmsnorm(x), r)` then attention, then `o_proj`, then a second
     `residual_add`). The megakernel's numerics must match THIS, not a
     hand-rolled formula.

3. **Do the register/LDS budget MATH before writing the kernel.** This is the
   CPU-doable core and the thing that makes the `persistent` block honest, not
   decorative:
   - **LDS budget:** sum the staged-tile bytes for every intermediate kept
     on-chip (use `src/xkernels/vkl/cost.py` `predict_scratch`). Compare to the
     arch's scratch budget (LDS on AMD, smem on NVIDIA — from the card's
     `arch.scratch.bytes`). If `overflows_scratch(...)` is true, you MUST defuse
     an edge (write that intermediate to DRAM) and name it in `defused_edges`.
   - **Register budget:** `register_budget.regs_per_thread` is the half the
     occupancy model keeps profile-calibrated (vkl/cost.py). Declare your
     target; `spills: true` is a red flag — if the budget forces register
     spills to scratch, the megakernel may lose to separate kernels. This is
     where a profile is load-bearing: measure the actual regs/thread with
     rocprof/ncu before pinning the number.
   - **Heterogeneous-GEMM trap:** an Apertus layer has 5 GEMMs in 4 shapes
     (q/k/v/o_proj ~4096-wide; up_proj 21504-wide; down_proj 21504×4096).
     Keeping BOTH the 4096 activations AND the 21504 FFN hidden resident across
     the whole layer is almost certainly infeasible. The honest move is to
     bound the megakernel to ONE sub-block (attention OR FFN), not a whole
     layer, and defuse the cross-block edge to DRAM.

4. **Fill the `persistent` block honestly.** Every pinned sub-field is a
   load-bearing claim, not documentation:
   - `residency`: the intermediate names that NEVER round-trip to DRAM (the
     whole point of the megakernel — if this is empty, you wrote a graph, not a
     persistent kernel).
   - `defused_edges`: the producer→consumer edges that DO hit DRAM (the
     complement of `residency`; empty `defused_edges` + non-empty `residency` =
     fully on-chip).
   - `warp_roles`: the named roles (producer / consumer / epilogue / reducer).
   - `register_budget` / `lds_budget`: the numbers from step 3.
   - `pipeline_stages`: the staged tiles in the vkl schedule IR vocabulary
     (`Stage(id, producer_ref, space, depth)`); `depth` is the pipeline
     buffering depth (int or a knob name resolved at emit).
   A hand-waved `persistent` block defeats the purpose — it is what lets an
   agent (and `find_impl`) reason about applicability from metadata (§1.3.2).

5. **Shape-gate the applicability constraint.** The card's `constraints` MUST
   restrict it to the regime where the profile said it wins — decode (M ≤ a
   small bound). A megakernel offered for prefill is a correctness-and-perf
   landmine (it loses at the roofline). Express this as a constraint the
   constraint evaluator can check (e.g. `M <= 8`), so `find_impl` rejects it
   for prefill shapes BEFORE compile (reject-before-compile, §1.3.2).

6. **Write the kernel source.** Two hard facts constrain this:
   - **Attention CANNOT be emitted by the DSL** (online softmax, causal /
     data-dependent masking, paged-KV gather — blocker (c)). A whole-layer
     megakernel is therefore a **hand-written native CUDA/HIP** project, OR a
     carefully crafted persistent Triton grid that calls into the existing
     hand-written attention leaf. There is no `@kernel`-one-source path here.
   - If the sub-block is attention-free (e.g. a fused FFN sub-block), the DSL
     MIGHT express it — but a persistent grid across multiple GEMMs is beyond
     the DSL's current lowering (it lowers ONE kernel per `@kernel`). Route to
     `author-a-kernel-with-dsl` only if the block is genuinely one math-IR DAG;
     otherwise hand-author.
   Set `provenance.source_path` to the hand-written source.

7. **Register + verify + verify_parity + record_measurement.** `@register` the
   megakernel under the new op's dispatch key + backend. `verify(<card>,
   arch=...)` must pass correctness on the target arch. `verify_parity(<op>)`
   must agree against the composed reference. `measure_perf=True`; the
   validation gate is `perf.ms < graph-captured-baseline perf.ms` on the
   shape-gated regime. `record_measurement` the winner so the next task skips
   re-tuning (the compounding loop, §6.2).

## Pitfalls

- **Firing this skill before exhausting the interim path.** Graph capture +
  narrow fusion kill most of the win at a fraction of the complexity /
  validity-surface cost. If you haven't measured the graph-captured baseline,
  you can't claim the megakernel wins.
- **A whole-LAYER megakernel.** The heterogeneous-GEMM register/LDS budget
  (4096-wide activations + 21504-wide FFN hidden resident simultaneously) is
  almost certainly infeasible. Bound to ONE sub-block; defuse the cross-block
  edge. A whole-layer megakernel is the §10 anti-goal in disguise.
- **Offering the card for prefill.** A decode megakernel loses at the prefill
  roofline. Shape-gate the constraint; let `find_impl` reject it for large M.
- **A decorative `persistent` block.** If `residency` is empty or the budget
  numbers are guesses, the block is documentation, not a contract — and
  applicability can't be reasoned about from metadata. Fill every field from
  the budget math (step 3) and the profile.
- **Trying to emit attention from the DSL.** It can't (online softmax, masking,
  paged gather — blocker (c)). A whole-layer megakernel is native-authoring,
  full stop. Do not route to `author-a-kernel-with-dsl` for an attention block.
- **Overloading an existing single-op spec's outputs to host the megakernel.**
  It is always a NEW op (author-an-op-spec first). Overloading breaks every
  card already validating against that spec.
- **Silent register spills.** If `register_budget.spills` is true, the
  megakernel may lose to separate kernels (spill bandwidth eats the fusion
  win). The profile is load-bearing here — measure regs/thread, don't guess.
