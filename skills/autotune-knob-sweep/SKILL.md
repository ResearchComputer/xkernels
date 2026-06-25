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

## Procedure

1. `get_impl_card(impl_card_id)`. Read `specialization_knobs` — this is the
   *declared* search space (§4/§2.2). The agent does not freestyle knobs outside it.
2. Enumerate the Cartesian product of each knob's `choices` (or `min..max`).
   Skip points that violate the card's arch constraints.
3. For each point, `verify(impl_card_id, arch, knobs=<point>, shapes=[<concrete point>])`.
   Discard any point where `correctness.passed` is false — a fast-but-wrong kernel
   is never the winner.
4. Among passing points, pick the min `perf.ms` (or max `tflops` / `achieved_bw_pct`
   if an op-specific model exists — §11).
5. `record_measurement(impl_card_id, arch, shape, dtype, knobs=<winner>,
   ms=..., source=<verify run_id>)`. The next task with the same
   (arch, shape, dtype) is now served from cache — autotuning is skipped (§6.2).

## Pitfalls

- Sweeping outside the declared space "just to try" — that defeats the
  reason-able validity surface (§1.2). Extend `specialization_knobs` on the card
  first, with justification.
- Recording a winner without a `source` run id — un-sourced measurements are
  dropped by the loader (§2.4). Always pass `verify`'s `artifacts.run_id`.
- Accepting a high-variance point — the harness reports median+IQR; a point with
  high variance is flagged, not silently averaged (§5.4).
