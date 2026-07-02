---
name: autotune-knob-sweep
description: >
  Search an Implementation Card's declared specialization_knobs space for the
  best point on a target arch, then record the winner to the card's perf.measured
  so the next task skips autotuning. Use when a card has non-empty
  specialization_knobs and no measured entry matches the target (arch, shape, dtype).
license: Apache-2.0
x-kernel-lib:
  id: autotune-knob-sweep@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "no perf.measured entry matches the target arch/shape/dtype"
      - "card has non-empty specialization_knobs"
    preconditions:
      - "verify(card).correctness.passed == true"
  inputs_required:
    - "impl_card_id"
    - "target arch"
    - "concrete shape + dtype"
  tools:
    - get_impl_card
    - verify
    - record_measurement
  validation:
    must_pass:
      - "every swept point passes correctness"
      - "winner recorded with a reproducible source run id"
  references: []
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

> **Maturity note — harness perf fields.** `verify(..., measure_perf=True)`
> returns `perf = {ms, tflops, achieved_bw_pct}`, but today only **`ms`** is
> populated (median wall-clock via `do_bench`). `tflops` and `achieved_bw_pct` are
> stubbed to `None` in the harness; compute them yourself and pass to
> `record_measurement(..., tflops=, achieved_bw_pct=)`. **The FLOP/byte model + the
> roofline gate now exist** in [`xkernels.vkl.cost`](../../src/xkernels/vkl/cost.py)
> (Phase 2.3): `workload(op_id, point)` -> `(flops, bytes)`, `roofline(...)` ->
> `{tflops, ceiling, ai, bottleneck}`, `roofline_gate(ms, op, point, arch, instr)`
> -> the §2 70%-of-ceiling verdict. The normalized `stall_reasons`/`occupancy`
> vocabulary §10 describes still needs an external profiler (`rocprof` AMD /
> Nsight Compute NVIDIA — see `use-rocprof-compute` / `use-nsight-compute`);
> `vkl.cost.occupancy()` gives the closed-form (smem/warp) half, profile-calibrated
> for the register half.
>
> **DSL-native driver.** If the card was emitted by the
> [`author-a-kernel-with-dsl`](../author-a-kernel-with-dsl/SKILL.md) skill, its
> `specialization_knobs` came from the `@kernel` body's `Target.knobs` dict, and
> **`xkernels.vkl.autotune(impl_card_id, arch, point)`** is the higher-level
> driver over this same `verify(measure_perf=True)` path: it enumerates the
> declared space, **rejects smem/lds-overflow configs before launch** (via
> `vkl.cost.overflows_scratch` — no crash), times the survivors, writes the winner
> to `perf.measured`, appends the full history to `provenance.tuning_trace`, and
> records the roofline-gate verdict. Prefer it for DSL-emitted cards; the raw
> per-point `verify(knobs=...)` loop below is the general (hand-card) path.

## Procedure

1. `get_impl_card(impl_card_id)`. Read `specialization_knobs` — this is the
   *declared* search space (§4/§2.2). The agent does not freestyle knobs outside it.
2. Enumerate the Cartesian product of each knob's `choices` (or `min..max`).
   Skip points that violate the card's arch constraints.
3. For each point, `verify(impl_card_id, arch, knobs=<point>, shapes=[<concrete point>])`.
   Discard any point where `correctness.passed` is false — a fast-but-wrong kernel
   is never the winner.
4. Among passing points, pick the min `perf.ms` — the only objective `verify()`
   populates today (see the maturity note above; `tflops`/`achieved_bw_pct` are
   stubbed to `None`). If you compute tflops/bandwidth yourself from an external
   FLOP/byte model, you may optimize on that instead and record it in step 5.
5. `record_measurement(impl_card_id, arch, shape, dtype, knobs=<winner>,
   ms=..., source=<verify run_id>)`. The next task with the same
   (arch, shape, dtype) is now served from cache — autotuning is skipped (§6.2).

## Pitfalls

- Sweeping outside the declared space "just to try" — that defeats the
  reason-able validity surface (§1.2). Extend `specialization_knobs` on the card
  first, with justification.
- Recording a winner without a `source` run id — un-sourced measurements are
  dropped by the loader (§2.4). Always pass `verify`'s `artifacts.run_id`.
- Accepting a high-variance point — the harness reports only the **median** ms
  (a single number; no IQR/variance). If a point looks like a lucky-low outlier,
  re-time it at a higher iteration count before crowning it: a noisy winner is
  useless to the next task that skips autotuning off it.
