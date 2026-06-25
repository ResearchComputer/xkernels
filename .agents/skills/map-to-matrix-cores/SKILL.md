---
name: map-to-matrix-cores
description: >
  Replace a kernel's generic FMA inner loop with the target arch's matrix-core
  instruction (MFMA on AMD CDNA, tensor-core/wmma/wgmma on NVIDIA) where dtype +
  shape allow — the single primitive swap that turns a correct-but-slow kernel
  into one that uses the hardware's compute engine. Narrower than tune-for-cdna
  (which retiles + restages + tunes the whole CDNA kernel): this skill is JUST the
  FMA->matrix-core mapping plus the dtype/encoding prerequisites it forces (fp8
  needs e4m3fnuz on CDNA3; fp16/bf16 need the matching MFMA shape). Today the
  repo's matrix-core work is expressed through Triton (matrix_instr_nonkdim,
  kpack, instruction-dtype) rather than raw v_mfma intrinsics, since no native
  HIP/CUDA kernel source exists yet — the skill covers both the Triton-MFMA
  targeting path and the native-intrinsic path for when native cards arrive.
  GPU-gated. Use when a card is correct but compute-bound and NOT using the matrix
  engine (profiler shows low tensor/MFMA utilization).
license: Apache-2.0
x-kernel-lib:
  id: map-to-matrix-cores@1.0.0
  backend_scope: [hip]
  when_to_use:
    triggers:
      - "card is compute-bound (perf.roofline == compute_bound) but an external profiler (rocprof/omniperf) shows low MFMA / matrix-instruction utilization"
      - "an FMA-inner-loop kernel exists and the dtype + tile shape admit an MFMA op (fp8/fp16/bf16/fp32 with a supported MFMA shape)"
      - "tune-for-cdna step 3 (map the inner product to MFMA) called as a standalone procedure"
    preconditions:
      - "verify(card, amd_cdna*).correctness.passed == true"
      - "target arch.family in [amd_cdna2, amd_cdna3] (this skill is AMD-scoped; NVIDIA tensor-core targeting is a sibling skill)"
      - "dtype + inner-product shape match a native MFMA op (else fall back to FMA — do not force a shape the hardware can't decode)"
  inputs_required:
    - "impl_card_id + target amd arch"
    - "the kernel's inner-product dtype + reduction/contraction dimension"
  tools:
    - get_impl_card
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "correctness sweep still passes at the op's tolerance (MFMA fp32-accumulate must match the reference)"
      - "verify_parity still agrees (matrix-core vs FMA must be numerically equivalent at cross_backend_rtol)"
      - "external profiler shows matrix/MFMA utilization rose; perf.ms dropped toward the arch compute roofline (in-harness tflops is still None today — §11 — so the compute-fraction claim is external for now)"
  references:
    - "src/xkernels/ops/gemm/triton/configs.py (matrix_instr_nonkdim 16/32 -> 16x16x32 / 32x32 fp8 MFMA; kpack=2 VGPR packing; ~400 TFLOP/s gfx942 ceiling)"
    - "src/xkernels/ops/gemm/triton/entry.py (path='mfma' routing; e4m3fnuz vs e4m3fn -> native fp8 MFMA vs slow f16 upcast)"
    - "src/xkernels/ops/moe/triton/moe_int4_kernel.py (waves_per_eu, kpack, MFMA-shape selection for the grouped GEMM)"
    - ".agents/skills/tune-for-cdna/SKILL.md (the broader CDNA skill; this is its step 3 in isolation)"
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
> stubbed to `None` until an op-specific FLOP/byte model lands (open question
> §11). So the validation gate "matrix utilization rose and ms dropped toward the
> arch compute roofline" is checked from an **external profiler**
> (`rocprof`/`omniperf` MFMA utilization counters) plus `ms`; the achieved
> compute fraction is computed yourself from `ms` + the FLOP count and compared
> to the AMD MFMA ceiling (gfx942 ~400 TFLOP/s fp8). Pass it to
> `record_measurement(..., tflops=, achieved_bw_pct=)`, which accepts both even
> though `verify()` doesn't populate them yet.

## Procedure

1. **Confirm the matrix engine is the missing piece.** From an external profile
   (rocprof/omniperf): the card is compute-bound but MFMA utilization is low while
   VALU/FMA is busy. If instead the card is memory-bound, route to
   `diagnose-memory-bound`; if occupancy is the issue, `diagnose-low-occupancy`.
   This skill is specifically "the compute engine is idle/under-used."

2. **Check the dtype + shape admit a native MFMA op.** On gfx942 (CDNA3):
   - **fp8:** `v_mfma_*_fp8_fp8` decodes `float8_e4m3fnuz` natively; the OCP
     `e4m3fn` encoding upcasts to a slower f16 MFMA (see `entry.py`). If your
     operands are `e4m3fn`, step 3 is "switch the encoding to fnuz," not "tile."
   - **fp16/bf16:** match the MFMA shape to the tile (`matrix_instr_nonkdim` 16
     -> 16x16x32, 32 -> 32x32, etc.; `configs.py`).
   - **fp32:** only certain MFMA shapes apply; if none fits the contraction, keep
     FMA (do not force a mis-shaped MFMA — it loses, not gains).
   If the dtype/shape admits no MFMA, this skill does not apply; stay on FMA.

3. **Map the inner product to the matrix-core op.**
   - **Triton path (today's repo reality):** set `matrix_instr_nonkdim` to the
     MFMA shape that fits the contraction, set `kpack` (2 packs two K elements per
     VGPR for the MFMA feed — the ds_read/MFMA ratio matters), and let the
     `num_warps`/`waves_per_eu` recompute for 64-wide wavefronts. This lives in
     the autotune config space (`configs.py`); it is NOT an entry-callable knob,
     so it is documented on the card, not declared in `specialization_knobs`
     (author-an-op-spec / tile-a-gemm pitfall).
   - **Native-hip path (when native cards exist):** replace the FMA loop with
     `v_mfma_*` intrinsics matching the dtype, with explicit A/B VGPR layout and
     the LDS-staged feed. Accumulator stays fp32.

4. **Re-validate numerics.** MFMA accumulates in fp32 — exactly what the
   reference does — so correctness should hold at the existing tolerance. But the
   *association* of the contraction changes (tiled MFMA vs serial FMA), so
   re-run `verify` AND `verify_parity`: a cross-backend divergence here means one
   backend's MFMA tiling reordered the sum outside cross_backend_rtol (genuine,
   not a bug — §5.4 — but must be within the looser tolerance).

5. **Record.** `record_measurement(card, arch, shape, dtype, knobs=..., ms=...,
   source=<run_id>)`. State the achieved compute (vs the gfx942 ~400 TFLOP/s fp8
   ceiling for CDNA3) in the measurement note — the AMD roofline, never the
   NVIDIA card (§10).

## Pitfalls

- **Forcing an MFMA shape the dtype/contraction doesn't admit.** A mis-shaped
   MFMA is slower than the FMA it replaced. If no native MFMA fits, stay on FMA;
   this skill is conditional, not mandatory.
- **fp8 in the wrong encoding on CDNA3.** `e4m3fn` operands upcast to a slow f16
   MFMA — the card "uses MFMA" but the speedup vanishes and you'll misdiagnose it
   as tiling. Use `e4m3fnuz` (fnuz family) for the native fp8 MFMA path.
- **Declaring `matrix_instr_nonkdom`/`kpack` as specialization_knobs when the
   entry callable doesn't accept them.** They live inside the Triton autotune
   configs, not the entry signature; declaring them makes the harness report them
   unapplied (§1.2). Document on the card instead.
- **Treating MFMA as a numerics-neutral change.** It changes contraction
   association; cross-backend parity must be re-checked, not assumed.
- **Grading the win against the NVIDIA card.** AMD compute is reported against
   the AMD roofline only (§10). A "2x over the FMA baseline, 60% of gfx942 fp8
   ceiling" claim is honest; "still slower than the H100 card" is irrelevant.
- **Conflating with tune-for-cdna.** That skill is the whole-CDNA retune (tile +
   MFMA + LDS + waves_per_eu). This one is the single primitive swap. Call this
   when the diagnosis is specifically "matrix engine idle."
