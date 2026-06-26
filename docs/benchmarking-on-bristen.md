# Benchmarking xkernels on bristen (NVIDIA A100)

bristen is the NVIDIA cluster paired with beverin (AMD MI300A): CSCS Cray
Shasta, **4× A100-SXM4-80GB (sm_80)** per node, AMD EPYC 7713 hosts. `/capstor`
is shared between the two clusters, so the same synced tree serves both — the
`bristen` rcc profile just points the `srun`/sbatch at the NVIDIA nodes.

This mirrors [`benchmarking-on-beverin.md`](benchmarking-on-beverin.md); read
that for rcc basics. Only the bristen-specific mechanics are below.

## Setup

`.rcc/config.toml` has a `bristen` profile (host `bristen`, same remote
`/capstor/scratch/cscs/xyao/xkernels`). Select it explicitly — the default
profile is still `beverin`:

```bash
rcc --profile bristen status
rcc --profile bristen push
rcc --profile bristen run -- squeue -u xyao
```

## Why everything runs in the NGC PyTorch container

Unlike beverin (which has the `tokenspeed-rocm-aiter-myofi` uenv), bristen has
**no module system and no CUDA toolkit on the base image**. Every command runs
inside `nvcr.io/nvidia/pytorch:24.10-py3` via pyxis `srun --container-image=`.
That image provides torch 2.5.0a0 + triton 3.0.0 + Nsight Compute + Nsight
Systems. `scripts/run-on-bristen.sh` and the `slurm/*_bristen.sbatch` scripts
wire this up for you (image overridable via `$BRISTEN_IMAGE` / `$IMAGE`).

## Common commands

### Run a benchmark / test interactively on an A100

```bash
scripts/run-on-bristen.sh python3 -u benchmarks/bench_all.py
scripts/run-on-bristen.sh python3 -u tests/test_mhc_pre_post.py
```

`run-on-bristen.sh` pushes, then `srun`s onto a compute node inside the
container with `PYTHONPATH` set. Env overrides:

```bash
BRISTEN_TIME=00:20:00 BRISTEN_GPU=1 scripts/run-on-bristen.sh python3 -u ...
```

### Submit a SLURM job

```bash
# consolidated benchmark table
scripts/bench-on-bristen.sh                              # slurm/bench_all_bristen.sbatch
scripts/bench-on-bristen.sh slurm/bench_all_bristen.sbatch
```

The script prints the job id and a `tail` hint.

### Profile a kernel

```bash
KERNEL=dual_rmsnorm              scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
KERNEL=dual_rmsnorm MODE=sq      scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
KERNEL=dual_rmsnorm              scripts/bench-on-bristen.sh slurm/profile_nsys_bristen.sbatch
```

See [`profiling-on-bristen.md`](profiling-on-bristen.md) for the DCGM-pause
detail and the per-mode metrics.

## How it works

- **Partition / account:** `--partition=normal`, `-A a-infra02` (bristen's
  `srun` rejects jobs without `-A`). `normal` nodes are 128-core, 4×A100.
- **`scripts/run-on-bristen.sh`** does `rcc --profile bristen push`, then
  `rcc --profile bristen run -- srun -A a-infra02 --partition=normal
  --gpus-per-node=1 --container-image=docker://$IMAGE --container-mounts=/capstor:/capstor,...
  bash -lc 'cd $REPO && PYTHONPATH=... && "$@"'`. The login node has no GPU, so
  every interactive command must land on a compute node.
- **`scripts/bench-on-bristen.sh`** pushes and submits a sbatch with
  `REPO=/capstor/scratch/cscs/xyao/xkernels`; the sbatch itself handles the
  container image and GPU allocation.
- **`/capstor` is bind-mounted into the container** (`--container-mounts`), so
  the synced tree at `/capstor/scratch/cscs/xyao/xkernels` is visible and
  writable inside.

## See also

- [`profiling-on-bristen.md`](profiling-on-bristen.md) — Nsight Compute / Systems setup + the DCGM-pause gotcha.
- [`benchmarking-on-beverin.md`](benchmarking-on-beverin.md) — the AMD MI300A counterpart (uenv-based).
