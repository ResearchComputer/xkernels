---
name: tune-for-cdna
description: >
  Take a functionally-correct HIP Implementation Card and make it fast on AMD
  CDNA (gfx9xx): re-tile for 64-wide wavefronts, map the inner product to MFMA
  matrix cores, tune waves_per_eu, and restage through LDS. Use when an AMD HIP
  card passes correctness (verify) but misses its perf regime. This is the step
  that turns "it runs on AMD" into "it's good on AMD".
license: Apache-2.0
x-kernel-lib:
  id: tune-for-cdna@1.0.0
  backend_scope: [hip]
  when_to_use:
    triggers:
      - "hip card correct but slow"
      - "perf below amd roofline regime"
    preconditions:
      - "verify(hip_card, amd_cdna*).correctness.passed == true"
      - "arch.family in [amd_cdna2, amd_cdna3]"
  inputs_required:
    - "impl_card_id"
    - "target amd arch"
    - "failing perf regime"
  tools:
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "correctness sweep still passes"
      - "verify_parity still agrees"
      - "perf >= amd roofline baseline (rocBLAS/hipBLASLt or Composable Kernel)"
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
> stubbed to `None` until an op-specific FLOP/byte model lands (open question
> §11), and the normalized `stall_reasons`/`occupancy` vocabulary §10 describes
> is not emitted either. So the only in-harness objective you can read directly
> today is **min `perf.ms`**; grade tflops/achieved-bandwidth against the AMD
> roofline (rocBLAS/hipBLASLt/Composable Kernel baseline — step 1) by computing
> them yourself from `ms` + a FLOP/byte model, and use `rocprof`/`omniperf` for
> occupancy & stalls. Pass them to `record_measurement(..., tflops=,
> achieved_bw_pct=)`, which accepts both even though `verify()` doesn't populate
> them yet.

> **Run it on beverin (gfx942).** `verify(<hip_card>, amd_cdna3,
> measure_perf=True)` and the `rocprof` profile both run on beverin:
> ```bash
> scripts/cluster.sh run --host beverin -- python -c "from xkernels import verify; r=verify('<card>@1.0.0', arch='amd_cdna3', measure_perf=True); print(r['correctness']['passed'], r['perf']['ms'])"
> ```
> Any NVIDIA/CUDA parity side runs on ds5 in the NGC container (`arch=
> "nvidia_sm121"`): `rcc --profile ds5 run --docker -s '...'` (recipe + stand-up:
> `meta/docs/usage/ds5-testbed.md`). beverin runbook: `meta/docs/usage/clusters.md`;
> the rocprof/occupancy profile via `use-rocprof-compute`.

## Procedure

Separate from `port-cuda-to-hip` on purpose: functional port and performance
tuning are different procedures with different success criteria (§7.2).

1. `verify(impl_card_id, arch=amd_cdna3, measure_perf=True)`. Read `perf.ms`.
   Establish the AMD roofline baseline (rocBLAS / hipBLASLt / Composable Kernel
   for the op family — §11) as the honest bar, never the NVIDIA card.
2. Re-tile for **64-lane wavefronts**. Do not assume warp=32 (§4.1, §10):
   tiling, reduction tree depth, and occupancy arithmetic all depend on 64.
3. Map the inner product to **MFMA** matrix-core ops where dtype/shape allow
   (`v_mfma_*`). For fp8, use `float8_e4m3fnuz` operands — it's the only fp8
   encoding CDNA3 MFMA decodes natively; `e4m3fn` upcasts to a slower f16 MFMA.
4. Stage through **LDS** with the global->LDS DMA path; overlap with compute
   (pipeline depth). Tune `waves_per_eu` (1 vs 2) and LDS staging depth.
5. Sweep the declared `specialization_knobs` for the target arch; the winner is
   the point that maximizes achieved bandwidth / TFLOP/s under the roofline.
6. Re-run the correctness sweep + `verify_parity`. If both hold and perf clears
   the AMD roofline baseline, `record_measurement(impl_card_id, arch=amd_cdna3,
   shape=..., dtype=..., knobs=..., tflops=..., achieved_bw_pct=...,
   source=<run_id>)`. Next task skips autotuning (§6.2).

## Pitfalls

- Leftover warp=32 tiling from a hipified draft — silently halves occupancy.
  This is the first thing to check.
- Grading the result against the NVIDIA card. AMD perf is reported against the
  AMD roofline only (§10).
- Forgetting that fp8 needs the fnuz encoding on CDNA3 — the speedup vanishes
  and you'll misdiagnose it as a tiling problem.
