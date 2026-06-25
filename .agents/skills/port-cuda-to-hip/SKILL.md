---
name: port-cuda-to-hip
description: >
  Produce a functionally-correct HIP (AMD) Implementation Card from an existing
  CUDA (NVIDIA) one for the same Op Spec. Step 1 may use HIPIFY for a
  correctness-only starting point, but hipified output is a DRAFT, not a
  deliverable. Sets provenance.derived_from. Use when find_impl returns an op
  with a CUDA card but missing_backend for the AMD target.
license: Apache-2.0
x-kernel-lib:
  id: port-cuda-to-hip@1.0.0
  backend_scope: [cuda, hip]
  when_to_use:
    triggers:
      - "find_impl returned missing_backend for an amd target"
      - "cuda card exists, hip card does not"
    preconditions:
      - "Op Spec exists (the contract is backend-agnostic and already defined)"
      - "a CUDA Implementation Card passes verify on its target arch"
  inputs_required:
    - "source cuda impl_card_id"
    - "target amd arch (e.g. amd_cdna3)"
  tools:
    - get_impl_card
    - get_op_spec
    - verify
    - verify_parity
  validation:
    must_pass:
      - "new hip card verify(...).correctness.passed == true on the amd arch"
      - "verify_parity(op).agree == true (functional portability gate)"
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

This is the **functional port** — it produces "it runs correctly on AMD", NOT
"it's fast on AMD". Performance is a separate skill (`tune-for-cdna`). Conflating
them is the classic way "AMD support" ships as "AMD-compatible but 4x slow"
(§7.2, §10).

1. `get_op_spec(op_id)` — read the contract, constraints, and numerics. The Op
   Spec does NOT change; you are adding a backend to it.
2. `get_impl_card(<cuda_card_id>)` — read `source_path`, `arch`, `uses_primitives`.
3. Produce a correctness-only HIP source. HIPIFY is acceptable **only** as a
   starting draft (§7.2). Expect to fix:
   - `warpSize` (32) → wavefront (64): tiling, reduction tree depth, occupancy
     arithmetic all change. This is the #1 silent-occupancy-halver (§4.1, §10).
   - shared memory → LDS API.
   - warp shuffle → wavefront shuffle (64-wide).
   - tensor-core (wmma/wgmma) → MFMA intrinsics — but for a *functional* port
     you may first map to generic FMA and let `map-to-matrix-cores` do the
     matrix-engine port.
4. Author the new Impl Card: `backend: hip`, `arch.family: amd_cdna3` (or cdna2),
   `arch.wave_size: 64`, `provenance.derived_from: <cuda_card_id>`,
   `provenance.skill_used: [port-cuda-to-hip]`. `specialization_knobs` can be
   empty for the functional port.
5. `verify(<new_hip_card_id>, arch=amd_cdna3)` — must pass correctness.
6. `verify_parity(op_id)` — the AMD and NVIDIA backends must agree with each
   other within `cross_backend_rtol`. If not, run `establish-parity`.

## Pitfalls

- Shipping the hipified draft as the deliverable. It runs; it's usually slow and
  often still assumes warp=32. Always follow with `tune-for-cdna`.
- Reporting AMD perf relative to the NVIDIA card (§10). Grade against the AMD
  roofline only.
- Changing the Op Spec to "make it work on AMD" — the contract is invariant;
  fix the implementation, not the spec.
