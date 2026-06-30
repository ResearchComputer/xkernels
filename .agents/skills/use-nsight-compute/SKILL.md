---
name: use-nsight-compute
description: >
  Run NVIDIA Nsight Compute (ncu) — and its sibling Nsight Systems (nsys) — on a
  bristen A100 (sm_80) node to get the per-kernel occupancy, warp-stall,
  memory/roofline, and pipeline-utilization numbers that the
  diagnose-low-occupancy / diagnose-memory-bound / map-to-matrix-cores skills
  branch on. This is the BRISTEN REALIZATION of "the external profiler" those
  skills name but never show how to launch — it is how you turn verify()'s lone
  perf.ms into a real diagnosis on NVIDIA. ncu is the AMD ROCm Compute Profiler's
  counterpart (same role, same vocabulary slots); nsys adds the system timeline
  and a clean steady-state duration to cross-check verify()'s ms. Use whenever a
  card is correct but slow on NVIDIA (sm_80) and you need to know WHY (latency vs
  bandwidth vs idle tensor engine) before picking a fix skill.
license: Apache-2.0
x-kernel-lib:
  id: use-nsight-compute@1.0.0
  backend_scope: [cuda]
  when_to_use:
    triggers:
      - "a card passes verify() on nvidia_sm_80 but perf.ms is off-regime and you need the occupancy/stall/bandwidth/roofline profile to pick between diagnose-low-occupancy, diagnose-memory-bound, and map-to-matrix-cores"
      - "you must compute the tflops / achieved_bw_pct that verify() stubs to None (open question §11) to grade a card against the A100 roofline"
      - "you need a clean steady-state kernel duration to cross-check verify()'s ms (nsys path)"
    preconditions:
      - "verify(card, nvidia_sm_80).correctness.passed == true (this is a perf-diagnosis skill, not a correctness skill)"
      - "a bristen A100 allocation (the login node has no GPU; everything runs on a compute node)"
      - "the NGC PyTorch container image is reachable (it bundles ncu + nsys — there is NO one-time install, unlike use-rocprof-compute)"
  inputs_required:
    - "impl_card_id or a kernel name in meta/benchmarks/probe_ncu.py"
    - "which question to answer: compute-vs-memory (roofline / SpeedOfLight), occupancy+stalls (SchedulerStats/WarpStateStats), or duration/timeline (nsys)"
  tools:
    - verify
    - record_measurement
  validation:
    must_pass:
      - "ncu ran the requested --section set across its multi-pass replay and produced an importable .ncu-rep + a human report.txt with the target dispatch"
      - "the chosen fix skill's precondition (occupancy %, dominant stall, achieved DRAM bw %, tensor-pipe utilization) is now a NUMBER read from the report, not a guess"
  references:
    - "src/xkernels/verify.py (perf stubs: only ms is populated; this skill supplies the rest)"
    - ".agents/skills/diagnose-low-occupancy/SKILL.md (consumes SchedulerStats + WarpStateStats output)"
    - ".agents/skills/diagnose-memory-bound/SKILL.md (consumes SpeedOfLight + MemoryWorkloadAnalysis output)"
    - ".agents/skills/map-to-matrix-cores/SKILL.md (consumes ComputeWorkloadAnalysis / tensor-pipe output)"
    - ".agents/skills/use-rocprof-compute/SKILL.md (the AMD/beverin twin — same diagnosis slots, different tool)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: agent
    created: "2026-06-25T00:00:00Z"
    supersedes: []
---

> **Why this and not nvprof.** nvprof is **unsupported on Volta and later**, so
> it produces no data on A100 (sm_80) even though its binary is present. The
> supported NVIDIA profiler pair is **Nsight Compute (`ncu`)** — per-kernel
> occupancy/stall/roofline via counter replay — and **Nsight Systems (`nsys`)** —
> the system timeline + per-kernel duration via passive CUPTI (no replay). This
> skill is the ncu/nsys counterpart of `use-rocprof-compute` (AMD): same
> diagnosis slots, same fix-skill routing, different toolchain.

> **ncu vs nsys — pick by question.** ncu does a multi-pass **counter replay** of
> ONE kernel to get occupancy/stalls/pipeline-utilization/roofline (slow, one
> profiled dispatch, deep). nsys does a **single passive trace** of the whole
> timeline to get kernel durations, CUDA-API gaps, and memory-op stats (fast,
> overhead-free, shallow). The diagnose skills branch on ncu data; nsys is the
> duration cross-check and the "is the host the bottleneck" tool. Profile with
> ncu when you need *why*; profile with nsys when you need *how long / how
> serialized*.

## What it is

Nsight Compute (`ncu`, 2024.3.2 in the `nvcr.io/nvidia/pytorch:24.10-py3`
container) attaches to a target process and replays each profiled kernel
dispatch many times (one pass per counter group) to collect HW performance
counters, organized into **sections**:

- **SpeedOfLight** — the one-glance roofline: DRAM/L1/L2 throughput % vs Compute
  (SM) throughput %, plus SM/DRAM frequency and elapsed cycles.
- **ComputeWorkloadAnalysis** — IPC, issue slots, and **per-pipeline utilization**
  (FMA / ALU / **tensor (HMMA, Ampere)** / …) — this is where idle-tensor-core
  shows up.
- **MemoryWorkloadAnalysis** — DRAM/L2/L1 throughput and cache behavior.
- **Occupancy** + **LaunchStats** — registered/achieved occupancy, block/grid.
- **SchedulerStats** — active/eligible warps per scheduler, "no eligible" %.
- **WarpStateStats** — warp cycles per issued instruction, broken down by **stall
  reason** (the ncu analog of rocprof's SQ stall block).

`scripts/profile-ncu-bristen.sh` wraps it; `scripts/profile-nsys-bristen.sh`
wraps nsys. Both run inside the container, driven by `meta/benchmarks/probe_ncu.py`
(warm-up + fixed iteration count → one clean steady-state dispatch — the NVIDIA
twin of `probe_omniperf.py`, same kernels, same seeded shapes, so a bristen
profile is directly comparable to a beverin one).

## Setup (none, on purpose — contrast with use-rocprof-compute)

There is **no one-time install**. The NGC PyTorch 24.10 container already ships
`ncu` (at `/opt/nvidia/nsight-compute/<ver>/ncu`) and `nsys`
(`/opt/nvidia/nsight-systems*/bin/nsys`), alongside torch 2.5.0a0 + triton 3.0.0
to run the kernels. Where the AMD path needs a source clone + a `pandas<3` pin +
a `libdw.so.1` staging dance, the NVIDIA path needs exactly **one** host-side
step — and it is a runtime gotcha, not an install one (see Pitfalls).

## The one load-bearing runtime fix: pause DCGM

**DCGM (the node monitoring daemon) holds the GPU performance counters, so `ncu`
fails with "driver resource unavailable" unless you pause it first.** ncu's
counter replay needs the same perf fields DCGM polls every second; on a shared
node DCGM wins the race and ncu is refused.

The fix is host-side and must bracket the ncu run:

```bash
/usr/bin/dcgmi profile --pause     # before ncu  (frees the perf counters)
/usr/bin/dcgmi profile --resume    # after ncu   (restores node monitoring)
```

This is why `scripts/slurm/profile_ncu_bristen.sbatch` runs on the compute-node **host**
(calling `dcgmi` directly as user `xyao` — no sudo needed) and wraps the
in-container `ncu` `srun` step, with a `trap … EXIT` that always resumes DCGM.
**`ncu` profiling goes through the sbatch; it does not work from a plain
interactive `srun --container-image`** because that would skip the pause.
`scripts/profile-ncu-bristen.sh` also attempts the pause defensively if `dcgmi`
happens to be reachable inside the container, but the sbatch pause is the
reliable one.

`nsys` uses **passive CUPTI activity tracing**, not perf counters, so it does
**not** need the DCGM pause and runs fine in-container.

## Procedure: profile → read → route one kernel

1. **Pick the question first** — each ncu mode is a separate multi-pass replay,
   so don't ask for everything at once:
   | Question | mode flag | ncu sections | feeds skill |
   |---|---|---|---|
   | compute- or memory-bound? how far from roofline? | `roof` (DEFAULT) | SpeedOfLight + ComputeWorkloadAnalysis + MemoryWorkloadAnalysis + Occupancy + LaunchStats | diagnose-memory-bound / route decision |
   | low occupancy? which stall dominates? | `sq` | Occupancy + LaunchStats + WarpStateStats + SchedulerStats | diagnose-low-occupancy |
   | tensor engine utilized? | `roof` (read ComputeWorkloadAnalysis pipeline %) | + ComputeWorkloadAnalysis | map-to-matrix-cores |
   | everything (slow) | `full` | `--set full` | — |

2. **Run it** through the sbatch (ncu needs the host-side DCGM pause):
   ```bash
   # default: roofline + full metric set on dual_rmsnorm
   KERNEL=dual_rmsnorm              scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
   # SQ scheduler/stall block
   KERNEL=moe_sum_reduce MODE=sq    scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
   rcc --profile bristen run -- tail -f ncu-<jobid>.out
   ```
   `nsys` (no DCGM pause needed) can run interactively or via sbatch:
   ```bash
   scripts/cluster.sh run --host bristen bash scripts/profile-nsys-bristen.sh dual_rmsnorm
   # or
   KERNEL=dual_rmsnorm scripts/cluster.sh submit --host bristen scripts/slurm/profile_nsys_bristen.sbatch
   ```
   Outputs land in `.ncu-workloads/<kernel>_<mode>/` (`<name>.ncu-rep` GUI-importable,
   `<name>.report.txt` section tables, `<name>.sol.csv` peak rows) and
   `.nsys-workloads/<kernel>/` (`<kernel>.nsys-rep`, `.sqlite`, `.stats.txt`).
   To profile a kernel not in `meta/benchmarks/probe_ncu.py`, add a builder there
   first (warm-up + fixed iteration count → steady-state dispatch) plus the
   kernel's Triton name fragment.

3. **Read the report into the fix skills' vocabulary.** Verified on a real
   `dual_rmsnorm` A100 profile, here is the concrete map:
   - **`Section: GPU Speed Of Light Throughput`** (roof mode) — the route
     decision. Compare **DRAM Throughput %** vs **Compute (SM) Throughput %**.
     Our run read **DRAM 68.14% vs Compute 53.87%** → memory-bound →
     `diagnose-memory-bound`. Also: SM Frequency 1.40 GHz, Duration 38.50 µs,
     L2 Cache Throughput 65.26%, L1/TEX 50.33%, SM Active Cycles 49450.
   - **`Section: Compute Workload Analysis`** (roof mode) — IPC and the
     **pipeline utilization**. **Executed Ipc Active 2.31**, SM Busy 57.90%, and
     **FMA is the highest-utilized pipeline (42.9%)**. A compute-bound kernel
     here with a **low tensor/HMMA pipeline %** is the `map-to-matrix-cores`
     trigger (it's spending compute on FMA, not the tensor engine). Read the
     named pipeline in the INF line.
   - **`Section: Scheduler Statistics`** (sq mode) — the occupancy/eligibility
     picture. Our run read **Active Warps Per Scheduler 14.99** (of 16), but only
     **Eligible Warps Per Scheduler 1.69**, **No Eligible 42.35%**. High active
     warps but few eligible → the warps exist, they're just stalled → read the
     stall section. This is the `diagnose-low-occupancy` precondition.
   - **`Section: Warp State Statistics`** (sq mode) — the **dominant stall
     reason**. **Warp Cycles Per Issued Instruction 26.00**; the OPT line names
     the winner: our run read *"11.0 cycles stalled waiting for a L1TEX …
     operation … 42.3% of the total"*. Map the stall name:
     `L1TEX`/`LG`/`MIO Throttle` → memory latency → `diagnose-memory-bound`;
     `Wait`/scoreboard → dependency latency → `diagnose-low-occupancy`;
     `Long Scoreboard` with a saturated tensor pipe → `map-to-matrix-cores`.
     (A100's `Avg. Active Threads Per Warp` is 32 — the NVIDIA wave size, vs 64
     on AMD.)
   - **`Section: Memory Workload Analysis`** (roof mode) — DRAM/L2/L1 throughput
     and coalescing. Cross-checks the SpeedOfLight DRAM %; replay/coalescing
     sub-metrics tell you *why* bandwidth is low (uncoalesced) vs capped (at the
     roofline). Feeds `diagnose-memory-bound`'s coalescing/vector-load branch.
   - The per-section OPT/WRN lines already name the dominant issue and an
     estimated speedup — read them; they encode the routing.

   **Convergence check:** the same `dual_rmsnorm` profiled in `roof` (DRAM ≫
   Compute) and `sq` (L1TEX stall = 42.3%) agree → memory-bound → one fix skill.
   If your two modes disagree, trust the stall reason (it's causal, not a ratio).

4. **Close the loop with `record_measurement`.** Compute the two fields `verify()`
   stubs, then record them so the next task skips re-profiling (§6.2):
   ```python
   # tflops = 2 * FLOPs_of_op(shape) / (duration_s)        # duration from ncu Duration or nsys
   # bw_pct = DRAM Throughput %   (ncu reports this directly in SpeedOfLight)
   ```
   A100-SXM4-80GB peaks (measure on-device to confirm; these are Ampere non-sparse
   figures): ~9.7 TF/s FP16/BF16 tensor, ~19.5 TF/s FP16 with 2:4 sparsity,
   ~1.55 TF/s FP32 (non-tensor), ~1.94 TB/s HBM2e. Hand the numbers to
   `record_measurement(impl_card_id, arch="nvidia_sm_80", shape, dtype, knobs, ms,
   tflops, achieved_bw_pct, source=<run_id>)`. Cross-check `ms` against the nsys
   steady-state kernel duration — they should match within noise.

## Pitfalls

- **Don't use this skill on a crashing / wrong-results dispatch.** ncu replays
  counters and will either die with the crash or report a stall reason unrelated
  to the bug. This skill REQUIRES `verify().correctness.passed == true`. If the
  kernel crashes or fails `verify` on GPU (e.g. interpreter-green but GPU-red),
  route to [`diagnose-wrong-results`](../diagnose-wrong-results/SKILL.md) FIRST
  to restore correctness, then come back here for perf.
- **DCGM holds the perf counters (the load-bearing gotcha).** Without
  `dcgmi profile --pause`, `ncu` dies with `==ERROR== Resource Danger: driver
  resource unavailable` / `NVDRV_WARN_EVENT_OS_INFO`. The sbatch does the pause
  host-side with a `trap --resume EXIT`; a plain interactive in-container `srun`
  skips it and fails. nsys is unaffected (passive CUPTI).
- **This ncu build (2024.3.2) rejects a bare `--` before the target** as an
  "ambiguous empty option", and **`-k` needs the `regex:` prefix** to
  substring-match the Triton kernel name. `profile-ncu-bristen.sh` uses
  `-k "regex:$FRAG"` and no `--`; if you call ncu by hand, do the same.
- **`--export` suppresses the stdout section text in this build.** The run
  produces the `.ncu-rep`, but to get the human-readable roofline/occupancy
  *tables* you must `ncu --import <name>.ncu-rep` afterward (the driver does this
  and tees it to `.report.txt`). If your `.report.txt` is 6 lines, you forgot the
  import step.
- **`nsys --force=true` is ambiguous** in this version (matches both
  `--force-overwrite` and `--force-start-capture-range`). Use
  `--force-overwrite=true`. (Already fixed in `profile-nsys-bristen.sh`.)
- **`-c 1` profiles exactly one dispatch.** The probe's warm-up already
  JIT-compiled + filled caches, so the first matching launch is the real
  steady-state kernel — don't raise `-c` to "get more data", you'll profile
  autotune/launch helpers or repeats that add nothing.
- **Profile the wrong dispatch.** A Triton kernel compiles to several device
  kernels (e.g. `tl.sprod` reductions for the rmsnorm); `torch.randn` setup shows
  up too. `-k "regex:$FRAG"` keeps ncu on the op under test. If several match,
  check `--target-processes=all` didn't pull in an unrelated process and narrow
  the regex.
- **Interactive `srun` drops mid-queue.** The `normal` partition is shared; your
  rcc connection times out while the job is `PD`. Use the sbatch wrapper for real
  profiles and `tail` the `ncu-<jobid>.out`.
- **"Data collection happened without fixed GPU frequencies."** ncu prints this
  warning on shared nodes where you can't lock clocks. It makes cross-run
  comparisons slightly noisier but does not invalidate a single profile; for
  record_measurement, take the duration from that run's ncu/`nsys`, not a
  datasheet.
- **Don't assume the number alone; read the dominant stall.** Low occupancy with
  a saturated compute engine is fine; low occupancy with an **idle tensor pipe**
  is `map-to-matrix-cores`, not an occupancy fix. Route by the *combination* of
  SpeedOfLight ratio + SchedulerStats + WarpStateStats + the named pipeline %.
- **Re-verify correctness after any fix.** This skill only diagnoses; the fix
  skills re-run `verify` + `verify_parity`. A profiler-guided change that breaks
  parity changed the math, not just the scheduling.
