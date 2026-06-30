# Clusters — benchmarking & profiling on beverin (MI300A) and bristen (A100)

This is the unified runbook for running xkernels benchmarks and profilers on the
two CSCS clusters. Both are reached through the same [`rcc`](https://github.com/ResearchComputer/remote-cluster-controller)
toolkit and the shared `/capstor` filesystem, so the mechanics are stated once
and the per-cluster specifics follow.

| Cluster | GPU | Arch id | Software stack | Profiler |
|---|---|---|---|---|
| **beverin** | AMD Instinct MI300A | `gfx942` (CDNA3) | `tokenspeed-rocm-aiter-myofi` uenv (torch 2.11 + rocm 7.2) | ROCm Compute Profiler (rocprof-compute, ex-Omniperf) |
| **bristen** | NVIDIA A100-SXM4-80GB | `sm_80` (Ampere) | NGC PyTorch `24.10-py3` container (torch 2.5.0a0 + **triton 3.0.0**) | Nsight Compute (`ncu`) + Nsight Systems (`nsys`) |

`/capstor/scratch/cscs/xyao/xkernels` is **shared** between the two clusters, so
one `rcc push` syncs the tree to both; select the cluster with
`rcc --profile {beverin,bristen}` (default is `beverin`).

The GB10 (ds5) CUTE-DSL testbed is a separate, single-node runbook:
[`ds5-testbed.md`](ds5-testbed.md).

---

## Shared mechanics (both clusters)

`rcc` is already installed locally and initialized in `.rcc/` (profiles:
`beverin`, `bristen`, `ds5`; remote working copy
`/capstor/scratch/cscs/xyao/xkernels`). The remote path is on scratch so it does
not clobber the existing `/capstor/scratch/cscs/xyao/kernels` checkout. SLURM
scripts accept a `REPO` override, so they work with the rcc-synced tree.

All commands go through the driver `scripts/cluster.sh`, which has two verbs:

- **`scripts/cluster.sh run --host <h> <cmd>`** — pushes with `rcc push`, then
  runs the command interactively on a compute node (good for quick tests; lands on
  a compute node because the login nodes have no GPU). Env overrides:
  `BRISTEN_TIME=…`, `BRISTEN_GPU=…`.
- **`scripts/cluster.sh submit --host <h> [sbatch]`** — pushes and submits a
  SLURM job (use this for anything that queues: beverin `mi300` is usually
  saturated; profiling jobs especially). Prints the job id + a `tail -f` hint.

`.rcc/rccignore` excludes local-only files (`.git/`, `.rcc/`, `.claude/`, caches,
build artifacts) so syncs stay small.

### Sync only / inspect remote state

```bash
rcc push                          # default (beverin) profile
rcc --profile bristen push
rcc status
rcc shell
rcc run -- squeue -u xyao
```

### The consolidated benchmark table

```bash
scripts/cluster.sh submit --host beverin                              # bench_all_beverin.sbatch (9 ops)
scripts/cluster.sh submit --host bristen                              # bench_all_bristen.sbatch
# or interactively:
scripts/cluster.sh run --host bristen python3 -u meta/benchmarks/bench_all.py
```

Each cell is **median of Triton `do_bench`** (`xkernels.utils.benchmarking.benchmark`),
bf16 unless noted (FFN is fp16 — see the bf16-GEMM cliff in `kernels/gemm.md`).
Shapes mirror the README "Performance" table (Kimi-K2.6 / V4 serving regime).
**On bristen, run each kernel in its own process** — see the SIGSEGV gotcha
(`meta/wiki/04-gotchas.md` §1); `bench_all_bristen_isolated.sbatch` +
`scripts/bench_kernel_loop_bristen.sh` do this (`set +e` so the loop survives a
per-kernel crash).

---

## beverin (AMD MI300A / gfx942)

The `tokenspeed-rocm-aiter-myofi` container sets its own working directory, so
wrap the python command in `bash -c 'cd REPO && ...'`:

```bash
scripts/cluster.sh run --host beverin \
  srun --environment=tokenspeed-rocm-aiter-myofi \
       --partition=mi300 --gpus-per-node=1 --time=00:10:00 \
  bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && python3 -u meta/benchmarks/bench_all.py'
```

- **Partition:** `mi300` (the head node has MI250X — always use `--partition=mi300`
  for gfx942 results).
- **Container:** `tokenspeed-rocm-aiter-myofi` provides torch 2.11.0+rocm7.2 and
  the Triton build the kernels use.

### Profiling — ROCm Compute Profiler (ex-Omniperf)

beverin runs the **open `amdgpu` driver** headless, which rules out RGP. The
supported profiler is the **ROCm Compute Profiler** — same occupancy/stall/cache/
roofline data on the open driver. The `.agents/skills/use-rocprof-compute` skill
is the procedural guide; this section is the setup + invocation reference.

**Why not `pip install omniperf`.** AMD never published it to PyPI (both 404); the
GitHub release ships source only. The app is pure Python shelling out to the
system `rocprofv3` (ROCm 7.2, at `/usr/bin/rocprofv3` in the container), so the
install is a source clone into **scratch** (the container is read-only). Two
non-obvious fixes the setup scripts bake in (silent killers — see
`meta/wiki/04-gotchas.md` §5):
- **Pin `pandas<3`.** Unbounded `requirements.txt` → uv grabs pandas 3, whose
  strict `str` dtype breaks the v3→v2 counter join and the analyze metric
  assignment.
- **Stage `libdw.so.1` from the login node.** `rocprofv3` `dlopen`s it; the
  container lacks it; the host `/usr/lib64` isn't mounted; and rocprof-compute
  resets the subprocess `LD_LIBRARY_PATH` to `/opt/rocm/lib` only — so the driver
  mirrors the staged lib into `/opt/rocm/lib` (writable but per-container-instance
  → redo every run, which the profile script does).

One-time install (compute node for the tool, login node for the staged libs):
```bash
scripts/cluster.sh run --host beverin \
  srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
       --gpus-per-node=1 --time=00:25:00 \
  bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && bash scripts/setup-rocprof-compute-beverin.sh'
rcc run -- bash scripts/stage-rocprof-compute-libs-beverin.sh    # LOGIN node
```

Profile + analyze a kernel (the driver mirrors the libs, activates the env,
profiles, then analyzes — one command; submit via sbatch because `mi300` is
usually full):
```bash
KERNEL=dual_rmsnorm MODE=roof scripts/cluster.sh submit --host beverin scripts/slurm/profile_omniperf_beverin.sbatch
KERNEL=moe_sum_reduce MODE=sq   scripts/cluster.sh submit --host beverin scripts/slurm/profile_omniperf_beverin.sbatch
rcc run -- tail -f omniperf-<jobid>.out
```
Output: `.omniperf-workloads/<kernel>_<mode>/` (`pmc_perf.csv`, `roofline.csv`)
+ `<name>.analyze.txt` (the metric tables).

---

## bristen (NVIDIA A100 / sm_80)

Unlike beverin, bristen has **no module system and no CUDA toolkit on the base
image**. Every command runs inside `nvcr.io/nvidia/pytorch:24.10-py3` via pyxis
`srun --container-image=`. That one image bundles torch 2.5.0a0 + triton 3.0.0
(to run the kernels) **and** Nsight Compute 2024.3.2 + Nsight Systems (to profile
them). `scripts/cluster.sh run/submit --host bristen` and the
`scripts/slurm/*_bristen.sbatch` scripts wire this up (image overridable via
`$BRISTEN_IMAGE` / `$IMAGE`). `/capstor` is bind-mounted in, so the synced tree is
visible and writable.

```bash
scripts/cluster.sh run --host bristen python3 -u meta/benchmarks/bench_all.py
scripts/cluster.sh run --host bristen python3 -u tests/test_mhc_pre_post.py
BRISTEN_TIME=00:20:00 BRISTEN_GPU=1 scripts/cluster.sh run --host bristen python3 -u ...
```

- **Partition / account:** `--partition=normal`, `-A a-infra02` (bristen's `srun`
  rejects jobs without `-A`).

### Profiling — Nsight Compute (`ncu`) + Nsight Systems (`nsys`)

Two non-obvious fixes the scripts bake in (these took real digging —
`meta/wiki/04-gotchas.md` §4/§7):

1. **DCGM holds the GPU performance counters**, so `ncu` fails with *"driver
   resource unavailable"* unless you pause it first. Fix: `dcgmi profile --pause`
   before `ncu`, `--resume` after — **from the host** (the sbatch script runs
   there), not the container. `scripts/slurm/profile_ncu_bristen.sbatch` traps
   `--resume` on exit. `nsys` uses passive CUPTI activity and needs **no** pause.
2. **ncu option-parser quirks:** this build rejects a bare `--` before the target;
   `-k` needs the **`regex:`** prefix to substring-match the Triton kernel name;
   `--export` suppresses the stdout section text so the script captures the run
   log then regenerates the human report via `--import`.

```bash
KERNEL=dual_rmsnorm           scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
KERNEL=dual_rmsnorm MODE=sq   scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
KERNEL=dual_rmsnorm           scripts/cluster.sh submit --host bristen scripts/slurm/profile_nsys_bristen.sbatch
# nsys interactively (no DCGM pause needed):
scripts/cluster.sh run --host bristen bash scripts/profile-nsys-bristen.sh dual_rmsnorm
```

Output layout:
```
.ncu-workloads/<kernel>_<mode>/  {<name>.ncu-rep, .report.txt, .run.log, .sol.csv}
.nsys-workloads/<kernel>/        {<kernel>.nsys-rep, .sqlite, .stats.txt}
```

### nvprof is dead here

On A100 (sm_80, Volta-and-later) **nvprof is non-functional** — NVIDIA removed
profiling support for these architectures from nvprof. The binary is present but
produces nothing. Use `ncu` (per-kernel) and `nsys` (system timeline) only.

---

## Profiling modes (both profilers, same slots)

| Mode | collects | answers |
|---|---|---|
| `roof` | roofline + full default metric set | compute- vs memory-bound + how far from the line + occupancy (**start here**) |
| `sq` | scheduler block only | active occupancy %, dominant stall reason |
| `full` | every section/block explicitly | everything (slow — many replay passes) |

To profile a kernel not yet in `meta/benchmarks/probe_omniperf.py` /
`probe_ncu.py`, add a builder there (the twins share identical seeded shapes so a
beverin profile is directly comparable to a bristen one; each warms up 5× then
runs 10 steady-state iterations so the profiled dispatch is clean).

## How a profile routes to a fix skill

A profile fills the same diagnosis slots on either arch, so the same skills
consume it — just swap the profiler:
- Memory Throughput % ≫ Compute % → `diagnose-memory-bound`.
- Low active occupancy / a dominant stall reason → `diagnose-low-occupancy`.
- Compute-bound but low tensor/MFMA utilization → `map-to-matrix-cores`.

Read the matching profiler skill **before** running (`use-rocprof-compute` on AMD,
`use-nsight-compute` on NVIDIA) — they hold the question→mode table and exact
metric names. The 2026-06-26 full-library campaign that exercised all of this is
in `meta/wiki/` (methodology, benchmarks, profiling, gotchas).

## Computing the fields `verify()` stubs

`verify(..., measure_perf=True)` returns only `ms`; `tflops` and `achieved_bw_pct`
are `None` (open question, `library.md` §11). Derive them from the profile / shape
and feed them to `record_measurement`:
```python
tflops = 2 * FLOPs_of_op(shape) / (ms / 1e3)
bw_pct  = bytes_moved / (ms / 1e3) / peak_HBM
```
Device peaks (measure on-device to confirm): MI300A ≈ 130.7 TF/s FP16/BF16 MFMA
(fp8 ≈ 2×), ≈ 819 GB/s HBM3; A100-80GB SXM4 ≈ 9.7 TF/s FP16/BF16 tensor, ≈ 1.94
TB/s HBM2e. On bristen `achieved_bw_pct` is direct (ncu "DRAM Throughput %"); on
beverin derive it from the byte model (the rocprof per-kernel FLOPs column is
broken — see `meta/wiki/03-profiling.md`).

## See also

- [`ds5-testbed.md`](ds5-testbed.md) — the GB10 (sm_121) CUTE-DSL single-node testbed.
- `.agents/skills/use-rocprof-compute/`, `.agents/skills/use-nsight-compute/` —
  the procedural profiler skills (read before running).
- `meta/wiki/01-methodology.md` — the benchmark/profile campaign harness + field
  definitions; `meta/wiki/04-gotchas.md` — the host-side gotchas.
