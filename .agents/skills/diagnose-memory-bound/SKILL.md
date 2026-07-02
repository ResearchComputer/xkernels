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
      - "card correct but memory-bound and far from the arch bandwidth roofline (compute achieved_bw_pct yourself — verify() stubs it to None; see maturity note)"
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

> **Maturity note — harness perf fields.** `verify(..., measure_perf=True)`
> returns `perf = {ms, tflops, achieved_bw_pct}`, but today only **`ms`** is
> populated (median wall-clock via `do_bench`). `tflops` and `achieved_bw_pct` are
> stubbed to `None` until an op-specific FLOP/byte model lands (open question
> §11), and the normalized `stall_reasons`/`occupancy` vocabulary §10 describes
> is not emitted either. So the only in-harness objective you can read directly
> today is **min `perf.ms`**. When this skill references tflops, achieved
> bandwidth, occupancy, or stalls, compute them yourself — `ms` + your own
> FLOP/byte model for `tflops`/`achieved_bw_pct`; `rocprof` (AMD) or Nsight
> Compute (NVIDIA) for occupancy and stalls — and pass them to
> `record_measurement(..., tflops=, achieved_bw_pct=)`, which accepts both even
> though `verify()` doesn't populate them yet.

> **Run `verify` on ds5 (rcc + docker); profile on bristen/beverin.** The
> `verify(arch, measure_perf=True)` call below (correctness + `ms` + the achieved-
> bandwidth numerator) runs in the NGC container on the GB10 (`arch=
> "nvidia_sm121"`):
> ```bash
> rcc --profile ds5 push
> rcc --profile ds5 run --docker -s 'python -c "from xkernels import verify; r=verify(\"<card>@1.0.0\", arch=\"nvidia_sm121\", measure_perf=True); print(r[\"correctness\"][\"passed\"], r[\"perf\"][\"ms\"])"'
> ```
> `-s` = shell snippet; `--docker` sets `PYTHONPATH=/workspace/src`. The bandwidth
> diagnosis itself needs the external profiler (`use-nsight-compute` bristen /
> `use-rocprof-compute` beverin). AMD/gfx942 verify → `scripts/cluster.sh run
> --host beverin`. DSL ops not yet imported by `ops/<x>/__init__.py` need
> `register_dsl` first. Full recipe: `meta/docs/usage/ds5-testbed.md`.

## Procedure

1. `verify(impl_card_id, arch, measure_perf=True)`. Read `perf.ms` and
   `correctness` (must be passing). Pull the card's `perf.roofline` (the card's
   *declared* metadata — real) and `arch.wave_size`. Note `perf.achieved_bw_pct`
   is stubbed to `None`: compute it yourself as
   `bytes_moved / (perf.ms * peak_bandwidth)` to see how far you are from the roofline.
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
