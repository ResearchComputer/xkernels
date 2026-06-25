---
name: diagnose-memory-bound
description: >
  Improve a functionally-correct kernel whose verify() perf profile or roofline
  says memory-bound: improve coalescing, switch to vectorized loads, stage
  through shared memory (NVIDIA) or LDS (AMD) with async copy (cp.async / TMA on
  NVIDIA; global->LDS DMA on AMD). Use when a card passes correctness but its
  achieved bandwidth is below the architecture's roofline.
license: Apache-2.0
x-kernel-lib:
  id: diagnose-memory-bound@1.0.0
  backend_scope: [cuda, hip]
  when_to_use:
    triggers:
      - "card correct but achieved_bw_pct below target"
      - "perf.roofline == memory_bound and regime misses"
    preconditions:
      - "verify(card).correctness.passed == true"
      - "arch.wave_size known (32 NVIDIA / 64 AMD) — do not assume"
  inputs_required:
    - "impl_card_id"
    - "target arch"
    - "failing perf regime or achieved_bw_pct"
  tools:
    - verify
    - get_impl_card
    - record_measurement
  validation:
    must_pass:
      - "correctness sweep still passes after the change"
      - "verify_parity still agrees (if multi-backend)"
      - "achieved_bw_pct improved (or ms dropped) vs the recorded baseline"
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

1. `verify(impl_card_id, arch, measure_perf=True)`. Read `perf.ms` and
   `correctness` (must be passing). Pull the card's `perf.roofline` and
   `arch.wave_size`.
2. Confirm memory-bound: the op's byte traffic dominates. For a GEMM that's
   usually compute-bound — if a GEMM shows memory-bound, suspect redundant
   reloads or a transpose materialization.
3. Pick the fix by backend (do **not** assume warp=32; read `arch.wave_size`):
   - **NVIDIA**: coalesce loads, vectorize (`.vec4`), stage through smem with
     `cp.async`, or TMA on sm_90.
   - **AMD (CDNA)**: coalesce, vectorize, stage through LDS via the
     global->LDS DMA path. Tune `waves_per_eu` if occupancy is the real issue.
4. Re-run the correctness sweep + `verify_parity` (§5.3). If both hold and perf
   improved, `record_measurement(impl_card_id, arch, shape, dtype, source=<run_id>, ms=..., achieved_bw_pct=...)`.

## Pitfalls

- Re-staging through smem/LDS without async copy just adds latency — the copy
  must overlap compute (pipeline depth).
- A "memory-bound" diagnosis on a kernel that *should* be compute-bound usually
  means the matrix engine isn't being used — escalate to `map-to-matrix-cores`
  (AMD) or tensor-core targeting (NVIDIA) instead.
