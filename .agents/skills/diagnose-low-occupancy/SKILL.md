---
name: diagnose-low-occupancy
description: >
  Branch on an occupancy/stall profile to a CONCRETE fix for a kernel that is
  correct but slow: register/VGPR pressure, scratch spill, block/wave size
  mismatch, or waves_per_eu. Fully GPU-gated: needs a compiled card + an external
  profiler (rocprof on AMD, Nsight Compute on NVIDIA), because verify() does not
  yet emit the §10 stall_reasons/occupancy vocabulary (see maturity note). Use
  when a card passes correctness but ms is bad and the problem is
  latency/occupancy, not bandwidth (route memory-bound cases to
  diagnose-memory-bound).
license: Apache-2.0
x-kernel-lib:
  id: diagnose-low-occupancy@1.0.0
  backend_scope: [cuda, hip]
  when_to_use:
    triggers:
      - "card passes verify but perf.ms is far off regime, AND an external profiler (rocprof/Nsight) reports low active-occupancy or a dominant stall reason in {sgpr/vgpr pressure, scratch, scheduling, instruction-cache, wait-cnt}"
      - "a tuned card regressed on a new arch without a numerics change (occupancy math shifted)"
    preconditions:
      - "verify(card, arch).correctness.passed == true (this skill fixes perf, not correctness)"
      - "a compiled card on a real GPU + profiler access (rocprof / Nsight Compute)"
      - "arch.wave_size known (32 NVIDIA / 64 AMD) — recompute occupancy from 64 on AMD, NEVER assume 32"
    # NOTE: blocked on the harness until verify() emits the §10 normalized
    # profiler vocabulary. Until then, run the external profiler yourself and map
    # its raw output onto the vocabulary below before branching.
  inputs_required:
    - "impl_card_id + target arch"
    - "an occupancy/stall profile (rocprof --stats / Nsight Compute SOL/Source) for the failing shape"
    - "arch.wave_size (read from the card; 32 NVIDIA, 64 AMD CDNA)"
  tools:
    - verify
    - get_impl_card
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "correctness sweep still passes after the fix"
      - "verify_parity still agrees (an occupancy fix must not change numerics — if it did, you changed the math, not just the scheduling)"
      - "perf.ms improved on the failing shape (measurable in-harness via ms; the occupancy/stall improvement is re-measured externally since the harness doesn't emit it yet)"
  references:
    - "meta/docs/library.md §10 (stall_reasons/occupancy normalized to a common vocabulary — aspirational; harness emits only ms today), §11 (op-specific perf model is an open question)"
    - "src/xkernels/ops/moe/triton/moe_int4_kernel.py (waves_per_eu tuning — the AMD occupancy knob), src/xkernels/ops/gemm/triton/configs.py (num_warps = wavefronts on AMD)"
    - ".agents/skills/tune-for-cdna/SKILL.md (the broader CDNA perf skill this branches into on AMD)"
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
> is not emitted either. This skill's ENTIRE input — the occupancy percentage and
> the dominant stall reason — comes from an **external profiler** (`rocprof` /
> `omniperf` on AMD, Nsight Compute on NVIDIA) that you map onto the vocabulary
> in step 1; `verify()` cannot supply any of it. (When the §10 profiler layer
> lands in `verify()`, this skill updates to read it directly.) `ms` is the only
> in-harness signal, and it cannot distinguish occupancy from bandwidth from
> compute — hence the external profile.

> **Run `verify` on ds5 (rcc + docker); profile on bristen/beverin.** The
> `verify(arch, measure_perf=True)` half (correctness + `ms`) runs in the NGC
> container on the GB10 (`arch="nvidia_sm121"`):
> ```bash
> rcc --profile ds5 push
> rcc --profile ds5 run --docker -s 'python -c "from xkernels import verify; r=verify(\"<card>@1.0.0\", arch=\"nvidia_sm121\", measure_perf=True); print(r[\"correctness\"][\"passed\"], r[\"perf\"][\"ms\"])"'
> ```
> `-s` = shell snippet; `--docker` sets `PYTHONPATH=/workspace/src`. The occupancy
> half of THIS skill still needs the external profiler — `use-nsight-compute`
> (bristen) / `use-rocprof-compute` (beverin). AMD/gfx942 verify →
> `scripts/cluster.sh run --host beverin`. DSL ops not yet imported by
> `ops/<x>/__init__.py` need `register_dsl` first. Full recipe:
> `meta/docs/usage/ds5-testbed.md`.

## Procedure

1. **Get the profile from an EXTERNAL profiler** (the harness can't give you one
   yet). On AMD: `rocprof --stats` / `omniperf` for occupancy + the
   `SQ_THREAD_CYCLES_BUSY` / VGPR-limit / scratch metrics. On NVIDIA: Nsight
   Compute's "Occupancy" + "Warp State Statistics" / "Stall Reasons" sections.
   Map the raw reason onto the normalized vocabulary before branching:
   - **VGPR/SGPR pressure** → register count exceeds the per-SIMD/per-SM budget,
     cutting resident waves/warps.
   - **scratch spill** → registers spilled to backing memory (catastrophic on
     AMD; the card's `arch.scratch.kind` should say `registers`, not `scratch`).
   - **scheduling / wait-cnt / instruction-cache** → latency-bound, often an
     un-pipelined load or a too-deep dependency chain.
   - **low achieved occupancy with NO dominant stall** → block/wave size doesn't
     tile the hardware evenly (the classic AMD warp=32 leftover).

2. **Branch to the concrete fix by reason.** Read `arch.wave_size` first (32
   NVIDIA / 64 AMD) — every occupancy calculation depends on it:
   - **VGPR pressure:** reduce live register count — split the kernel, stage one
     intermediate through LDS/smem, or drop an over-wide tile. On AMD MFMA,
     `kpack` and the MFMA shape drive VGPR count (`configs.py`).
   - **scratch spill:** the #1 catastrophic regressor. Re-tile so the working set
     fits in registers/LDS; never ship a card whose `arch.scratch.kind == scratch`.
   - **block/wave size mismatch:** re-tile so one tile is an integral number of
     wavefronts (AMD, 64) or warps (NVIDIA, 32). The classic hipified-warp=32
     leftover halves AMD occupancy — this is the first thing to check on a regressed
     AMD card (tune-for-cdna pitfall).
   - **scheduling/latency:** raise pipeline depth (`num_stages`) to overlap the
     load with compute, or unroll the K-loop. If the matrix engine is idle, route
     to `map-to-matrix-cores` instead (this is a "compute engine unused" symptom,
     not an occupancy symptom).
   - **waves_per_eu (AMD):** try 1 vs 2 — 2 hides latency at the cost of VGPR
     pressure, so it trades off with the VGPR fix above.

3. **Re-run the correctness sweep + verify_parity.** An occupancy fix changes
   scheduling, not math — if `verify_parity` now diverges, you accidentally
   changed the numerics (e.g. reordered a reduction); back it out and redo.

4. **Re-profile + record.** Confirm the stall reason dropped and `ms` improved.
   `record_measurement(card, arch, shape, dtype, knobs=<fix>, ms=..., source=<run_id>)`.
   The next task with that (arch, shape, dtype) is served from cache (§6.2).

## Pitfalls

- **Treating the harness `ms` as an occupancy signal.** It isn't — `ms` can't
   tell occupancy from bandwidth from compute. You need the external profile.
   (When the §10 normalized vocabulary lands in verify(), this skill updates to
   read it directly; until then the external-profiler path is mandatory.)
- **Assuming warp=32 on AMD.** Every occupancy/tiling number changes at 64.
   Read `arch.wave_size`; never hardcode.
- **Fixing occupancy and breaking parity.** If verify_parity moved, you changed
   the math (association/reduction order), not just scheduling. The fix is
   numerics-neutral by definition — re-derive it.
- **Shipping a scratch-spilling card.** Scratch to backing memory is a
   multi-order-of-magnitude regressor; the card's `arch.scratch` must read
   `registers` or `lds`. A scratch card should never clear review.
- **Routing a compute-idle symptom here.** If the matrix engine is idle, the fix
   is `map-to-matrix-cores` (use the hardware), not an occupancy tweak. Low
   occupancy with a saturated compute engine is fine; low occupancy with an idle
   engine is a mis-targeting.
