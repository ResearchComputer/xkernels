---
name: add-epilogue-fusion
description: >
  Attach a fused epilogue (bias, activation, residual-add, norm, scale-cast) to
  an existing kernel WITHOUT touching its main compute loop, by staging the
  epilogue into the kernel's output store (registers/LDS) instead of a separate
  launch. Covers two cases the agent must distinguish up front: (a) the fusion
  does NOT change the op's output shape/semantics -> a new Impl Card under the
  SAME Op Spec; (b) the fusion DOES change outputs (e.g. residual+rmsnorm emits
  BOTH the normalized x and the new residual) -> a NEW Op Spec (route to
  author-an-op-spec first). Kernel-layer skill: the fused card must compile and
  pass verify on a GPU; the contract/tolerance analysis is the CPU-doable part.
  Use when a profile shows a short pointwise/reduction op chained right after a
  memory/compute-heavy kernel, or when a benchmark calls for a fused variant.
license: Apache-2.0
x-kernel-lib:
  id: add-epilogue-fusion@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "a profile/trace shows a short pointwise or reduction kernel immediately after a GEMM/attention/reduce kernel, reading its output back from DRAM"
      - "an op family has a benchmarked 'fused' variant (e.g. fused SwiGLU inside FFN, residual+rmsnorm after all-reduce, MHC pre/post sigmoid+sinkhorn+residual)"
      - "a benchmark or issue requests 'fuse X into the epilogue of Y'"
    preconditions:
      - "the host kernel exists as an Impl Card and verify(host_card).correctness.passed == true"
      - "the epilogue is pointwise or a row/element reduction over the host kernel's own output tile (fusible into the store)"
  inputs_required:
    - "host impl_card_id (the kernel getting the epilogue)"
    - "epilogue spec: which ops, their parameters, and whether they add outputs"
    - "target arch + dtype"
  tools:
    - get_impl_card
    - get_op_spec
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "case (a): new fused card verifies under the SAME Op Spec (verify().correctness.passed == true), because the fused output equals unfused+epilogue bit-for-bit at the op's tolerance"
      - "case (b): a NEW Op Spec is authored (author-an-op-spec) and the fused card verifies against ITS new reference; never overload the old spec's outputs"
      - "verify_parity still agrees for multi-backend ops (an epilogue fusion is numerically relevant -> re-check)"
      - "the fused card's perf.ms <= host card's perf.ms + standalone-epilogue perf.ms on the target arch (else the fusion wasn't worth it)"
  references:
    - "src/xkernels/ops/comm/triton/add_rmsnorm_kernel.py (residual-add + RMSNorm epilogue, case b: two outputs)"
    - "src/xkernels/ops/ffn/triton/ffn_kernel.py (fused SwiGLU silu(g)*u inside FFN, case a)"
    - "src/xkernels/ops/mhc/triton/pre_post_kernel.py (sigmoid heads + sinkhorn + residual combine, case b)"
    - "meta/docs/library.md §10 (one-mega-kernel-with-a-thousand-flags anti-goal: fuse narrowly, don't flag-bomb)"
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

> **Run it on a GPU — ds5 via rcc + docker.** `verify` / `verify_parity` are
> device calls. Sync and run them inside the NGC container on the GB10
> (`arch="nvidia_sm121"`):
> ```bash
> rcc --profile ds5 push
> rcc --profile ds5 run --docker -s 'python -c "from xkernels import verify; print(verify(\"<fused_card>@1.0.0\", arch=\"nvidia_sm121\")[\"correctness\"][\"passed\"])"'
> ```
> `-s` = shell snippet (heredocs/pipes ok); `--docker` uses the profile container
> (`PYTHONPATH=/workspace/src` set, no venv). DSL ops not yet imported by
> `ops/<x>/__init__.py` need `register_dsl(spec_of(<body>),"triton")` first.
> AMD/gfx942 → `scripts/cluster.sh run --host beverin`. Full recipe:
> `meta/docs/usage/ds5-testbed.md`.

## Procedure

1. **Decide case (a) vs case (b) FIRST.** This is the load-bearing decision and
   it determines every following step:
   - **Case (a) — same outputs, just fused:** the epilogue transforms the host
     kernel's output in place and emits the *same* tensor (e.g. add bias, apply
     SwiGLU to the gate). The Op Spec's `outputs` are unchanged → you add an Impl
     Card under the existing Op Spec. Example: fused SwiGLU inside the FFN
     projections.
   - **Case (b) — new/extra outputs:** the fusion emits something the host op did
     not (e.g. residual+rmsnorm emits BOTH `out` and the updated `residual`; an
     epilogue that also writes a squared-sum for a downstream norm). The Op
     Spec's `outputs` change → **stop and run `author-an-op-spec`** for a new op,
     then come back. Never overload an existing spec's output contract to sneak
     in an extra tensor; that breaks every card already validating against it.
   Read the host Op Spec's `outputs` and compare to what your fused kernel emits.

2. **Pull the host card and its kernel source** (`get_impl_card`, read
   `provenance.source_path`). Locate the output-store path — that is where the
   epilogue stages. The rule: the epilogue computes in the *same* registers / LDS
   the output tile already lives in; it must NOT reload the host output from DRAM
   (reloading is just two kernels with a launch in between, i.e. no fusion at all).

3. **Write the epilogue into the store.** Concretely, in Triton: the `tl.store`
   of the host kernel becomes `tl.store(out_ptr, epilogue(tile), ...)`. A
   residual-add folds in `res_ptr` with one extra load; a row-reduction epilogue
   (rmsnorm) accumulates a per-row statistic in the same program. Keep the
   reduction in fp32 (`tl.float32`) regardless of storage dtype — this is the
   numerics-relevant change (step 5).

4. **Author the new Impl Card.** Case (a): `<op>.<backend>.fused.card.json` under
   the existing Op Spec, with `provenance.derived_from: <host_card_id>` and a
   `specialization_knobs` entry only if the epilogue adds a tunable (e.g. a
   second tile size). Case (b): a card under the NEW op from step 1. In both,
   `perf.regime` should read "fused <host> + <epilogue>, single launch, no DRAM
   round-trip for the intermediate."

5. **Re-derive tolerances honestly (numerics-relevant).** A fused epilogue can
   change numerics in two ways: (i) a row-reduction epilogue (rmsnorm/softmax)
   accumulates in fp32 internally — make sure the card's `reduce_dtype` reflects
   that, and the reference does too; (ii) fusing can change *association* of the
   math (e.g. bias added before vs after a cast). Confirm the fused output still
   matches `unfused + standalone epilogue` at the op's existing tolerance; if a
   reduction dtype widened, the tolerance is unchanged but `numerics.notes` should
   say so. **Do not** widen a tolerance to make a buggy fusion pass.

6. **Register + verify.** `@register` the fused kernel under the same dispatch
   key + backend (case a) or the new key (case b). `verify(<fused_card>, arch)`
   must pass correctness. `verify_parity(<op>)` must still agree (a fusion is a
   numerics-relevant change on every backend — §7.2 / establish-parity). On GPU,
   `measure_perf=True` and confirm the fused card beats host+standalone (the
   validation gate); `record_measurement` the winner.

7. **Honest no-GPU branch.** Without a GPU the fused card reports
   `compiled=False` and the perf gate in step 6 cannot be satisfied — that is
   expected. You CAN still complete the case-(a)/(b) decision, author the card,
   set tolerances, and run the reference-vs-reference CPU gate. Ship
   `compiled:false` honestly; the perf justification is deferred to a GPU run.

## Pitfalls

- **Overloading an existing Op Spec's outputs to "add" the epilogue tensor
  (case b done as case a).** Every card already validating against that spec
  silently breaks. Case (b) is a new op — full stop.
- **"Fusing" by reloading the host output from DRAM in the same kernel.** That's
  a fused launch, not a fused kernel — no bandwidth win, only launch overhead
  saved. The epilogue must stage in the tile that's already in registers/LDS.
- **A flag-bomb epilogue.** One kernel with `do_bias`, `do_act`, `do_norm`,
  `do_residual` booleans explodes the validity surface (§10 anti-goal). Fuse a
   *specific* epilogue into a *specific* host; emit a card per real combination,
  not a universal fused kernel.
- **Forgetting that a reduction epilogue needs fp32 accumulation.** A fused
  rmsnorm/softmax that reduces in the storage dtype silently loses precision;
  the reference (fp32) still passes in isolation but cross-backend parity drifts
  once the fused card lands on a second backend.
- **Reporting the fusion as a win without measuring the standalone baseline.**
  On a memory-bound host the epilogue is nearly free; on a compute-bound host it
  can *lose* to two kernels if the host's matrix engine stalls. The validation
  gate (fused <= host + standalone) is what makes "it's a win" truthful.
