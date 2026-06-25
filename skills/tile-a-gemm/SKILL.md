---
name: tile-a-gemm
description: >
  Build a tiled GEMM from primitives for a given dtype and arch, wave-size aware
  (32 on NVIDIA, 64 on AMD). The workhorse authoring skill. Use when find_impl
  returns no applicable GEMM card for the target arch/dtype, or when seeding a
  new op family's first dense GEMM.
license: Apache-2.0
x-kernel-lib:
  id: tile-a-gemm@1.0.0
  backend_scope: [cuda, hip]
  when_to_use:
    triggers:
      - "no gemm card matches target arch + dtype"
      - "seeding the first dense GEMM for a new op family"
    preconditions:
      - "Op Spec written (canonical_op: gemm, constraints, numerics, shape sweep)"
      - "backend-neutral reference exists and passes its own sweep"
  inputs_required:
    - "op_id"
    - "target arch + dtype"
  tools:
    - get_op_spec
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "correctness sweep passes vs the op reference"
      - "verify_parity agrees (once a 2nd backend exists)"
      - "achieved compute is a reasonable fraction of the arch roofline"
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

1. `get_op_spec(op_id)`. Confirm `op.canonical_op == "gemm"`, read `constraints`
   (e.g. `K % 8 == 0`) and `numerics` (accumulate in fp32 for fp16/bf16 inputs).
2. Read `arch.wave_size`. Tile so that one tile's output is produced by an
   integral number of warps/wavefronts:
   - NVIDIA: `BLOCK_M`, `BLOCK_N` in {64,128,256}; warp = 32 threads.
   - AMD CDNA: same tile choices but wavefront = 64 — recompute the
     threads-per-tile and reduction tree depth. Never hardcode 32 (§4.1, §10).
3. Stage the K-loop through smem (NVIDIA, `cp.async`/TMA on sm_90) or LDS (AMD,
   global->LDS DMA). Pipeline depth (`num_stages`) is a knob.
4. Map the inner product to the matrix engine: tensor cores / wgmma (NVIDIA) or
   MFMA (AMD). This is where dtype matters — fp8 on CDNA3 needs `e4m3fnuz`.
5. Declare the knobs you actually swept on the Impl Card's `specialization_knobs`
   (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_stages`, `waves_per_eu` on AMD). Empty
   or undeclared knobs are dishonest (§1.2).
6. `verify(card, arch)` then `verify_parity` once a second backend exists.
   `record_measurement` the winner per (arch, shape, dtype).

## Pitfalls

- Hardcoding warp=32 — breaks AMD silently. The card's `arch.wave_size` is
  authoritative.
- Declaring a huge knob space "for completeness" — the validity surface must
  stay reason-able (§1.2). Declare only what you can validate.
- Skipping the fp32-accumulation requirement for mixed precision — re-validate
  tolerances AND cross-backend parity (`mixed-precision-convert` skill).
