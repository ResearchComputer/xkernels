---
name: establish-parity
description: >
  Given an op with two or more backend Implementation Cards, run the
  cross-backend parity check (verify_parity) and, if the backends diverge
  outside cross_backend_rtol, localize which backend's numerics drifted from
  the shared reference and route to the right fix. Use whenever a new card is
  staged or a numerics-relevant change lands on any backend of a multi-backend op.
license: Apache-2.0
x-kernel-lib:
  id: establish-parity@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "new impl card staged for a multi-backend op"
      - "reduce_dtype or accumulation path changed on one backend"
      - "verify_parity reports diverging pair"
    preconditions:
      - "op has >=2 implementation cards"
  inputs_required:
    - "op_id"
  tools:
    - verify_parity
    - verify
    - get_op_spec
  validation:
    must_pass:
      - "verify_parity(op_id).agree == true"
      - "each card verify(card).correctness.passed == true"
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

1. Call `verify_parity(op_id)`. Read `agree`, `max_pairwise_rel_err`, and the
   `diverging` list (each entry names the offending `pair` + `point` + `rel_err`).
2. If `agree` is true, the op passes the portability gate (§5.3) — done.
3. If it diverges, the shared reference is the source of truth. For **each**
   backend card, call `verify(card_id, arch)` and compare
   `correctness.max_rel_err` against the op's `numerics.rtol`.
   - The backend that *also* fails its single-backend `verify` is the one that
     drifted from the reference. The other is faithful to the reference; the
     divergence is genuinely cross-backend numeric (accumulation order / dtype).
4. Route the fix:
   - A backend failing `verify` alone → its kernel is buggy; fix the kernel.
   - Both pass `verify` but `verify_parity` diverges → accumulation/dtype
     mismatch (e.g. one backend accumulates in fp16). Check
     `numerics.reduce_dtype` and align the backends, or — if the gap is inherent
     to the vendors — propose widening `cross_backend_rtol` with justification
     (open question §11: setting cross_backend_rtol).
5. Re-run `verify_parity`. A card may not publish until `agree` is true (§2.4).

## Pitfalls

- Don't tighten `cross_backend_rtol` to chase bit-equality — fp16/bf16
  accumulation order legitimately differs across vendors (§5.4). Cross-backend
  agreement uses the looser tolerance, never bit-equality.
- Don't "fix" parity by editing the reference — the reference is backend-neutral
  by construction (§5.1); a CUDA-shaped reference is an anti-goal (§10).
