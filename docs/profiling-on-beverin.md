# Profiling on beverin (MI300A) with ROCm Compute Profiler

beverin runs the **open `amdgpu` driver** on headless, containerized MI300A
nodes. That rules out **RGP** (gpuopen.com/rgp), whose full per-wave capture needs
the proprietary `amdgpu-pro`/PAL driver in Radeon Developer Mode. The supported
headless profiler for AMD Instinct is the **ROCm Compute Profiler** (formerly
**Omniperf**) — it gives the same occupancy / stall / cache / roofline data on
the open driver. The `.agents/skills/use-rocprof-compute` skill is the procedural
guide; this page is the setup + invocation reference.

## Why not `pip install omniperf`

AMD never published Omniperf / `rocprof-compute` to PyPI (both 404), and the
GitHub release ships source only. The app is pure Python and shells out to the
system `rocprofv3` (ROCm 7.2, at `/usr/bin/rocprofv3` in the
`tokenspeed-rocm-aiter-myofi` container), so the install is a source clone into
**scratch** (the container image is read-only; scratch persists).

Two non-obvious fixes the scripts bake in:
- **Pin `pandas<3`.** `requirements.txt` has no upper bound, so uv grabs pandas
  3, whose strict `str` dtype breaks the v3→v2 counter join and the analyze
  metric assignment. Without the pin, profile works but analyze dies.
- **Stage `libdw.so.1` from the login node.** `rocprofv3` `dlopen`s it; the
  container lacks it; the host `/usr/lib64` isn't mounted in the container; and
  `rocprof-compute` resets the profiler subprocess `LD_LIBRARY_PATH` to
  `/opt/rocm/lib` only — so the driver mirrors the staged lib into `/opt/rocm/lib`.

## One-time install

```bash
# 1. install the tool (compute node, in the container)
scripts/run-on-beverin.sh \
  srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
       --gpus-per-node=1 --time=00:25:00 \
  bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/setup-rocprof-compute-beverin.sh'
# 2. stage the runtime libs (LOGIN node — it sees the host /usr/lib64)
rcc run -- bash scripts/stage-rocprof-compute-libs-beverin.sh
```

Layout on scratch:
```
/capstor/scratch/cscs/xyao/rocprof-compute-src      # source tree (tag rocm-7.2.4)
/capstor/scratch/cscs/xyao/rocprof-compute-pylibs   # deps (pandas, scipy, dash, textual, ...)
/capstor/scratch/cscs/xyao/rocprof-compute-libs     # libdw.so.1, libelf.so.1, ...
```

## Profile + analyze a kernel

The driver mirrors the runtime libs, activates the env, profiles, then analyzes
— one command. The `mi300` partition is usually full, so submit via sbatch
(interactive `srun` drops your connection while queued):

```bash
# default: dual_rmsnorm, roofline + full metric set
KERNEL=dual_rmsnorm MODE=roof scripts/bench-on-beverin.sh slurm/profile_omniperf_beverin.sbatch
# SQ scheduler block (occupancy + stall reasons)
KERNEL=moe_sum_reduce MODE=sq scripts/bench-on-beverin.sh slurm/profile_omniperf_beverin.sbatch
rcc run -- tail -f omniperf-<jobid>.out
```

Output: `.omniperf-workloads/<kernel>_<mode>/` (`pmc_perf.csv`, `roofline.csv`,
`empirRoof_*.pdf`) and `<name>.analyze.txt` (the metric tables).

## Modes

| Mode | collects | answers |
|---|---|---|
| `roof` | roofline + full default metric set | compute- or memory-bound + occupancy/stalls (start here) |
| `sq` | SQ scheduler block only | active occupancy %, dominant stall reason, VGPR-limit |
| `full` | every block explicitly | everything (slow — many rocprof passes) |

To profile a kernel not yet in `benchmarks/probe_omniperf.py`, add a builder
there (warm-up + fixed iteration count so the dispatch is steady-state).

## Computing the fields `verify()` stubs

`verify(..., measure_perf=True)` returns only `ms`; `tflops` and
`achieved_bw_pct` are `None` (open question §11). Derive them from a profile and
hand them to `record_measurement`:

```python
# tflops = 2 * FLOPs_of_op(shape) / (ms / 1e3)
# bw_pct = bytes_moved / (ms / 1e3) / peak_HBM_gfx942
```

MI300A peaks (measure on-device to confirm): ~130.7 TF/s FP16/BF16 MFMA,
~819 GB/s HBM3.

## See also

- `.agents/skills/use-rocprof-compute/SKILL.md` — full procedure + how to route a
  profile to the `diagnose-*` / `map-to-matrix-cores` fix skills.
- `docs/benchmarking-on-beverin.md` — the rcc / sbatch / srun mechanics.
