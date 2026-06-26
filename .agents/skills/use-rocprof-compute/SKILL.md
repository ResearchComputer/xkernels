---
name: use-rocprof-compute
description: >
  Run AMD's ROCm Compute Profiler (the tool formerly called Omniperf — the
  MI300A equivalent of what RGP / Nsight Compute give you) on a beverin gfx942
  node to get the wave-level occupancy, stall-reason, cache/bandwidth, and
  roofline numbers that the diagnose-low-occupancy / diagnose-memory-bound /
  map-to-matrix-cores skills branch on. This is the BEVERIN REALIZATION of "the
  external profiler" those skills name but never show how to launch — it is how
  you turn `verify()`'s lone `perf.ms` into a real diagnosis. Use whenever a card
  is correct but slow on AMD and you need to know WHY (latency vs bandwidth vs
  idle matrix engine) before picking a fix skill.
license: Apache-2.0
x-kernel-lib:
  id: use-rocprof-compute@1.0.0
  backend_scope: [hip]
  when_to_use:
    triggers:
      - "a card passes verify() on amd_cdna3 but perf.ms is off-regime and you need the occupancy/stall/bandwidth profile to pick between diagnose-low-occupancy, diagnose-memory-bound, and map-to-matrix-cores"
      - "you must compute the tflops / achieved_bw_pct that verify() stubs to None (open question §11) to grade a card against the AMD roofline"
    preconditions:
      - "verify(card, amd_cdna3).correctness.passed == true (this is a perf-diagnosis skill, not a correctness skill)"
      - "a beverin MI300A allocation (the container has no GPU on the login node)"
      - "the one-time scratch install has been run (see Setup) — Omniperf is NOT on PyPI"
  inputs_required:
    - "impl_card_id or a kernel name in benchmarks/probe_omniperf.py"
    - "which question to answer: compute-vs-memory (roofline), occupancy/stalls (SQ block), or bandwidth/caches (TCC/LDS blocks)"
  tools:
    - verify
    - record_measurement
  validation:
    must_pass:
      - "profile + analyze ran and printed metric tables for the target dispatch"
      - "the chosen fix skill's precondition (occupancy %, dominant stall, achieved bw %) is now a NUMBER read from the analyze output, not a guess"
  references:
    - "src/xkernels/verify.py (perf stubs: only ms is populated; this skill supplies the rest)"
    - ".agents/skills/diagnose-low-occupancy/SKILL.md (consumes SQ-block output)"
    - ".agents/skills/diagnose-memory-bound/SKILL.md (consumes roofline + TCC/LDS output)"
    - ".agents/skills/map-to-matrix-cores/SKILL.md (consumes low-MFMA-utilization output)"
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

> **Naming reality.** AMD renamed **Omniperf** to **ROCm Compute Profiler**
> (`rocprof-compute`, package `rocprof_compute`, CLI prints `rocprofiler-compute
> 3.4.0`). Older docs, the `diagnose-*` skills, and most web results still call
> it Omniperf — same tool. This skill says "ROCm Compute Profiler" for the tool
> and keeps `omniperf` where it appears as a path/flag.

> **Why this and not RGP.** RGP (gpuopen.com/rgp) is a *desktop* GUI; its full
> per-wave capture needs the proprietary `amdgpu-pro`/PAL driver in Radeon
> Developer Mode. beverin runs the **open `amdgpu` driver** (headless,
> containerized MI300A), so RGP capture is a driver/site dead end. ROCm Compute
> Profiler is AMD's supported headless profiler for Instinct — it gives the same
> occupancy / stall / cache / roofline data on the open driver. It is the right
> tool here.

## What it is

ROCm Compute Profiler wraps the system **`rocprofv3`** (at `/usr/bin/rocprofv3`
in the `tokenspeed-rocm-aiter-myofi` container, ROCm 7.2) and:
- collects HW performance counters per kernel dispatch, organized into **blocks**
  (`SQ` = shader/scheduler, `TCC` = L2 cache, `TA`/`TD` = texture/data, `LDS`,
  `SQC` = instruction/L1 constant cache, …),
- builds a **roofline** (compute vs memory bound, achieved vs peak),
- prints normalized analysis tables (`analyze` mode) you can grep.

It is pure Python; it shells out to the system `rocprofv3`. **Nothing is on
PyPI** — install is a source clone (one-time, scripted).

## Setup (one-time; done on beverin, validated end-to-end on gfx90a)

Three pieces, because the container is read-only and can't see the host lib dir:

1. **Install the tool** (compute node, in the container) —
   `scripts/setup-rocprof-compute-beverin.sh` clones the `rocm-7.2.4` tag
   (matches the container's ROCm 7.2.x) and `uv pip install`s its deps into
   scratch. **It pins `pandas<3`** — this is the load-bearing fix:
   `requirements.txt` says `pandas>=1.4.3` so uv grabs pandas 3.x, but
   rocprof-compute 3.4.0 was written for pandas 2.x's `object` dtype, and pandas
   3.0's strict `str` dtype breaks BOTH the v3→v2 counter join
   (`Agent_Id` merge) and the analyze metric assignment. Without the pin, profile
   collects data but analyze dies.
2. **Stage runtime libs** (LOGIN node, not the container) —
   `scripts/stage-rocprof-compute-libs-beverin.sh` copies `libdw.so.1` + deps
   from the host `/usr/lib64` (which the container can't see) onto shared
   scratch. `rocprofv3`'s python bindings `dlopen` `libdw.so.1`, which the Ubuntu
   container lacks.
3. The profile driver mirrors those libs into `/opt/rocm/lib` per run (see below).

Layout on scratch:
```
/capstor/scratch/cscs/xyao/rocprof-compute-src     # source tree (tag rocm-7.2.4)
/capstor/scratch/cscs/xyao/rocprof-compute-pylibs  # deps, pandas<3
/capstor/scratch/cscs/xyao/rocprof-compute-libs    # libdw.so.1, libelf.so.1, ...
```
Reinstall: `scripts/run-on-beverin.sh srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 --gpus-per-node=1 --time=00:25:00 bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/setup-rocprof-compute-beverin.sh'` then `rcc run -- bash scripts/stage-rocprof-compute-libs-beverin.sh`.

## Activate (the driver does this; shown for manual use)

There is no `omniperf` on `$PATH` — the launcher is the in-tree `src/rocprof-compute`
script. And `rocprof-compute` **resets the profiler subprocess's `LD_LIBRARY_PATH`
to the ROCm lib dir only** (`profiler_rocprofiler_sdk.py` ~L73), so the staged
`libdw.so.1` must be mirrored into `/opt/rocm/lib` for the loader to find it:
```bash
export ROCPC_SRC=/capstor/scratch/cscs/xyao/rocprof-compute-src
export ROCPC_PYLIBS=/capstor/scratch/cscs/xyao/rocprof-compute-pylibs
export ROCPC_LIBS=/capstor/scratch/cscs/xyao/rocprof-compute-libs
cp -f "$ROCPC_LIBS"/*.so* /opt/rocm/lib/   # per-container-instance; redo every run
export PYTHONPATH="$ROCPC_PYLIBS:$ROCPC_SRC/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ROCPC_LIBS:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
RCP=(python3 "$ROCPC_SRC/src/rocprof-compute")
"${RCP[@]}" --version          # -> rocprofiler-compute version: 3.4.0
```
`scripts/profile-rocprof-compute-beverin.sh` does all of this for you.

## Procedure: profile → analyze one kernel

1. **Pick the question first** — each is a separate multi-pass `rocprofv3` run, so
   don't ask for everything at once:
   | Question | mode flag | block | feeds skill |
   |---|---|---|---|
   | compute- or memory-bound? how far from roofline? | `roof` (no `-b` flag = default metric set incl. roofline) | roofline + defaults | diagnose-memory-bound / route decision |
   | low occupancy? which stall dominates? | `sq` (`-b SQ`) | SQ (shader/scheduler) | diagnose-low-occupancy |
   | cache hit / achieved bandwidth? | `-b TCC TA TD LDS` | caches | diagnose-memory-bound |
   | matrix engine (MFMA) utilized? | `sq` + compute set | SQ+CPC | map-to-matrix-cores |

2. **Run it** via the driver (one command: stage-libs mirror + activate + profile + analyze):
   ```bash
   # default: roofline + full metric set on dual_rmsnorm, submitted via sbatch
   KERNEL=dual_rmsnorm MODE=roof scripts/bench-on-beverin.sh slurm/profile_omniperf_beverin.sbatch
   # interactive (only if mi300 has a free node; else it drops mid-queue)
   scripts/run-on-beverin.sh \
     srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 --gpus-per-node=1 --time=00:25:00 \
     bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/profile-rocprof-compute-beverin.sh dual_rmsnorm sq'
   ```
   Outputs land in `.omniperf-workloads/<kernel>_<mode>/` (raw `pmc_perf.csv`,
   `roofline.csv`, `empirRoof_*.pdf`) and `<name>.analyze.txt` (the metric
   tables). To profile a kernel not in `benchmarks/probe_omniperf.py`, add a
   builder there first (warm-up + fixed iteration count → steady-state dispatch).
   (Note: `profile -p` IS the output dir — `-n` is only a label — so the driver
   uses one dir per `(kernel,mode)`.)

3. **Read the analyze output into the fix skills' vocabulary.** The tables are
   numbered; here is the real map (verified on a `dual_rmsnorm` profile):
   - **`0.1 Top Kernels`** — which dispatch dominates (e.g.
     `dual_rmsnorm_kernel`, 88.8% of GPU time, mean 75.7 µs). Use `--list-stats`
     then `-k <id>`/`-d <id>` to isolate ONE dispatch when several are present.
   - **`1. System Info`** — `wave_size` (64 on AMD!), `max_waves_per_cu`, and
     `ip_blocks` (confirms what was collected, e.g.
     `SQ|LDS|SQC|TA|TD|TCP|TCC|SPI|CPC|CPF|roofline`).
   - **`2.1 ... Occupancy`** — the headline number. **`2.1.15 Wavefront
     Occupancy`** = achieved/peak wavefronts (the dual_rmsnorm profile read
     **50.37%**). `2.1.7 Active CUs`, `2.1.13 VALU Active Threads`. This is the
     `diagnose-low-occupancy` trigger.
   - **`5.x ... Stall`** sections (`CPF Stall`, `CPC Stall Rate`, `SQ` stalls) —
     the dominant stall reason maps to the `diagnose-low-occupancy` step-1
     vocabulary (instruction-cache / memory wait-cnt / VGPR-limit / scratch).
   - **`roofline`** (`roofline.csv` + `empirRoof_*.pdf`) — places the kernel's
     dot on the compute-vs-memory line. Memory region & far below →
     `diagnose-memory-bound`; at the compute ceiling with bad `ms` →
     `map-to-matrix-cores`.
   - ASCII "speed-of-light" bars (Wave Occupancy, LDS, Scalar L1D Cache, xGMI)
     give a one-glance saturation read.

4. **Close the loop with `record_measurement`.** Compute the two fields `verify()`
   stubs, then record them so the next task skips re-profiling (§6.2):
   ```python
   # tflops  = 2 * FLOPs_of_op(shape) / (ms / 1e3)        # FLOP model per op
   # bw_pct  = bytes_moved / (ms / 1e3) / peak_HBM_gfx942  # from roofline or TCC
   ```
   MI300A peaks: ~130.7 TF/s FP16/BF16 MFMA, ~819 GB/s HBM3 (use the measured
   device, not a datasheet). Hand the numbers to
   `record_measurement(impl_card_id, arch="amd_cdna3", shape, dtype, knobs, ms, tflops, achieved_bw_pct, source=<run_id>)`.

## Pitfalls

- **pandas 3.x breaks analyze (the load-bearing gotcha).** `requirements.txt`
  has `pandas>=1.4.3`; uv installs pandas 3, whose strict `str` dtype makes the
  v3→v2 counter join AND the metric assignment fail with
  `merge on str and int64 columns for key 'Agent_Id'` / `Invalid value ... for
  dtype 'str'`. The setup script pins `pandas<3` — if you reinstall by hand,
  keep that pin or analyze dies after a successful profile.
- **`libdw.so.1` is invisible to the profiler subprocess.** rocprofv3 `dlopen`s
  it; the Ubuntu container lacks it; the host `/usr/lib64` is not mounted in the
  container; AND `rocprof-compute` resets the profiler subprocess
  `LD_LIBRARY_PATH` to `/opt/rocm/lib` only. So: stage the libs from the LOGIN
  node (`stage-rocprof-compute-libs-beverin.sh`) and the driver mirrors them
  into `/opt/rocm/lib`. Skipping either → `OSError: libdw.so.1: cannot open…`.
- **Interactive `srun` drops mid-queue.** The `mi300` partition is usually full;
  your rcc connection times out while the job is `PD Priority`. Use the sbatch
  wrapper for real profiles and `tail` the `.out`. (Validating on an idle `mi200`
  node is fine for toolchain checks — same flow — but the numbers are gfx90a.)
- **`profile -p` is the output dir; `-n` is only a label.** The driver gives
  each `(kernel,mode)` its own dir; if you call `profile` by hand, don't expect a
  `-n` subdir or `analyze` will say `Invalid directory`.
- **Profile the wrong dispatch.** A Triton kernel compiles to several kernels
  (e.g. `tl.sprod` reductions); `torch.randn` shows up too. Use `--list-stats`
  then `-k <id>`/`-d <id>` to isolate, else you average over unrelated dispatches.
- **The bf16-GEMM container pathology.** This container's bf16 GEMM misses the
  MFMA/hipBLASLt path (~470× slower than fp16, see `bench_ffn`). Profile GEMM
  kernels in fp16 unless you are *diagnosing* that exact pathology — otherwise
  the profile shows a container bug, not your kernel.
- **Don't assume the number alone; read the dominant stall.** Low occupancy with
  a saturated compute engine is fine; low occupancy with an idle MFMA engine is
  `map-to-matrix-cores`, not an occupancy fix. Route by the *combination*.
- **Re-verify correctness after any fix.** This skill only diagnoses; the fix
  skills re-run `verify` + `verify_parity`. A profiler-guided change that breaks
  parity changed the math, not just the scheduling.
