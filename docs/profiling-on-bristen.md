# Profiling on bristen (A100 / sm_80) with Nsight Compute & Systems

bristen is the NVIDIA sibling of beverin: CSCS Cray Shasta nodes with **4×
NVIDIA A100-SXM4-80GB (Ampere, compute capability 8.0)**, AMD EPYC 7713 hosts,
driver 550.54.15. This page is the NVIDIA counterpart of
[`profiling-on-beverin.md`](profiling-on-beverin.md) — on AMD you use the ROCm
Compute Profiler; on bristen you use **Nsight Compute (`ncu`)** for per-kernel
occupancy/stall/roofline metrics and **Nsight Systems (`nsys`)** for the system
timeline.

## The two non-obvious fixes the scripts bake in

These took real digging — they are why profiling "just works" through the
sbatch wrapper rather than from a naive `srun ... ncu`:

1. **There is no module system and no CUDA toolkit on the base image.** Every
   job runs inside the **NGC PyTorch container** (`nvcr.io/nvidia/pytorch:24.10-py3`)
   via the pyxis `srun --container-image=` flag. That one image bundles
   torch 2.5.0a0 + triton 3.0.0 (to run the kernels) **and** Nsight Compute
   2024.3.2 + Nsight Systems (to profile them) — the all-in-one mirror of
   beverin's `tokenspeed-rocm-aiter-myofi` container.

2. **DCGM holds the GPU performance counters, so `ncu` fails with
   "driver resource unavailable" unless you pause it first.** The node monitor
   daemon (`/usr/bin/dcgmi`) continuously collects profiling fields; `ncu`'s
   kernel-replay needs the same counters and is refused. The fix is
   `dcgmi profile --pause` before `ncu` and `--resume` after. This **must** be
   done from the **host** (the sbatch script runs there), not the container —
   `slurm/profile_ncu_bristen.sbatch` does it and traps `--resume` on exit.
   `nsys` uses passive CUPTI activity and does **not** need the pause.

A third, smaller quirk: this ncu build's option parser rejects a bare `--`
before the target, and `-k` needs the `regex:` prefix to substring-match the
Triton kernel name. `profile-ncu-bristen.sh` handles both.

## nvprof is not supported here

You asked for `ncu`/`nvprof`: on A100 (sm_80, Volta-and-later) **nvprof is
non-functional** — NVIDIA removed profiling support for these architectures
from nvprof; use `ncu` (per-kernel) and `nsys` (system) instead. nvprof's
binary is present in the container/HPC SDK but will not produce data.

## Profile a kernel

`ncu` profiles go through the sbatch (it does the host-side DCGM pause), like
beverin's contended `mi300` partition — interactive `srun` would also drop your
connection while queued:

```bash
# default: dual_rmsnorm, roofline + full metric set
KERNEL=dual_rmsnorm scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
# SQ scheduler block (occupancy + stall reasons)
KERNEL=moe_sum_reduce MODE=sq scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
rcc --profile bristen run -- tail -f ncu-<jobid>.out
```

`nsys` (system timeline + kernel-duration/memory stats, no DCGM pause needed)
can run interactively or via sbatch:

```bash
scripts/run-on-bristen.sh bash scripts/profile-nsys-bristen.sh dual_rmsnorm
# or
KERNEL=dual_rmsnorm scripts/bench-on-bristen.sh slurm/profile_nsys_bristen.sbatch
```

### Output layout

```
.ncu-workloads/<kernel>_<mode>/
    <name>.ncu-rep      importable in the Nsight Compute GUI
    <name>.report.txt   human-readable section tables (roofline/occupancy/stalls)
    <name>.run.log      the ncu run log (passes, warnings)
    <name>.sol.csv      peak-utilization rows (for record_measurement)
.nsys-workloads/<kernel>/
    <kernel>.nsys-rep   importable in the Nsight Systems GUI
    <kernel>.sqlite     queryable (one row per event)
    <kernel>.stats.txt  auto-printed CUDA API/kernel/memory stats
```

## Modes (ncu)

| Mode | collects | answers |
|---|---|---|
| `roof` | SpeedOfLight roofline + compute/memory workload + occupancy + launch stats (DEFAULT) | compute- vs memory-bound + how far from the line + occupancy |
| `sq` | occupancy + warp stall reasons + scheduler stats | active occupancy %, dominant stall reason (mirror of rocprof's SQ block) |
| `full` | every section | everything (slow — one replay per section group) |

## How the profile routes to a fix skill

A bristen `ncu` profile fills the same diagnosis slots a beverin
`rocprof-compute` profile does, so the same skills consume it — just swap the
profiler:

- Memory Throughput % ≫ Compute % → `diagnose-memory-bound` (coalescing, vector loads, staging).
- Low active occupancy / a dominant stall reason → `diagnose-low-occupancy`.
- Compute-bound but low tensor-pipe utilization → `map-to-matrix-cores` (tensor cores).

## Computing the fields `verify()` stubs

As on beverin, `verify(..., measure_perf=True)` returns only `ms`; derive
`tflops` and `achieved_bw_pct` from the profile and feed them to
`record_measurement`. ncu's `report.txt` gives Duration (`gpu__time_duration`)
and DRAM throughput directly; cross-check Duration against `nsys` and against
`verify()`'s `ms`:

```python
# tflops = 2 * FLOPs_of_op(shape) / (duration_s)
# bw_pct = dram_throughput_%   (ncu reports this directly as "DRAM Throughput")
```

A100-80GB SXM4 peaks (measure on-device to confirm): ~9.7 TF/s FP16/BF16 tensor
(without 2:4 sparsity), ~1.55 TF/s FP32, ~1.94 TB/s HBM2e.

## Interactive ncu (power users)

Because the DCGM pause is host-side, interactive `ncu` needs a host step around
the container step. Easiest is a one-line salloc session:

```bash
rcc --profile bristen run -- bash -lc '
  srun -A a-infra02 --partition=normal --nodes=1 --gpus-per-node=1 --time=00:15:00 bash -c "
    /usr/bin/dcgmi profile --pause
    srun --container-image=docker://nvcr.io/nvidia/pytorch:24.10-py3 \
         --container-mounts=/capstor:/capstor \
         bash -c \"cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/profile-ncu-bristen.sh dual_rmsnorm roof\"
    /usr/bin/dcgmi profile --resume
  "'
```

For everything else prefer the sbatch path above.

## See also

- [`benchmarking-on-bristen.md`](benchmarking-on-bristen.md) — the rcc / sbatch / container mechanics.
- [`profiling-on-beverin.md`](profiling-on-beverin.md) — the AMD (MI300A / ROCm Compute Profiler) counterpart.
- `.agents/skills/diagnose-low-occupancy/SKILL.md`, `.agents/skills/diagnose-memory-bound/SKILL.md`,
  `.agents/skills/map-to-matrix-cores/SKILL.md` — the fix skills an ncu profile routes to.
