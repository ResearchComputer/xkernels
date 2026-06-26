# Benchmarking xkernels on beverin (MI300A)

This repo is configured to use [`rcc`](https://github.com/ResearchComputer/remote-cluster-controller) to sync the local tree to beverin and run benchmarks/tests there.

## Setup

`rcc` is already installed locally and initialized in `.rcc/`:

- **host:** `beverin` (ssh alias)
- **remote working copy:** `/capstor/scratch/cscs/xyao/xkernels`

The remote path is on scratch so it does not clobber the existing `/capstor/scratch/cscs/xyao/kernels` checkout. Existing SLURM scripts accept a `REPO` override, so they work with the rcc-synced tree automatically.

## Common commands

### Sync only

```bash
rcc push
```

### Run a benchmark interactively on a MI300A node

The `tokenspeed-rocm-aiter-myofi` container sets its own working directory, so wrap the actual python command in `bash -c 'cd REPO && ...'`:

```bash
scripts/run-on-beverin.sh \
  srun --environment=tokenspeed-rocm-aiter-myofi \
       --partition=mi300 --gpus-per-node=1 --time=00:10:00 \
       bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && python3 -u benchmarks/bench_all.py'
```

### Run a single test interactively

```bash
scripts/run-on-beverin.sh \
  srun --environment=tokenspeed-rocm-aiter-myofi \
       --partition=mi300 --gpus-per-node=1 --time=00:10:00 \
       bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && python3 -u tests/test_mhc_pre_post.py'
```

### Submit a SLURM benchmark job

```bash
# consolidated benchmark table
scripts/bench-on-beverin.sh slurm/bench_all_beverin.sbatch

# specific kernel benchmark/test
scripts/bench-on-beverin.sh slurm/test_mhc_pre_post_beverin.sbatch
scripts/bench-on-beverin.sh slurm/bench_moe_combine_beverin.sbatch
```

The script prints the job id and a `tail -f` command to follow the output.

### Inspect remote state

```bash
rcc status
rcc shell
rcc run -- squeue -u xyao
```

## How it works

- `scripts/run-on-beverin.sh` pushes with `rcc push`, then invokes the command through `bash -lc` so it runs from `/capstor/scratch/cscs/xyao/xkernels` with `PYTHONPATH` set to `.../xkernels/src`.
- `scripts/bench-on-beverin.sh` pushes and submits a SLURM script with `REPO=/capstor/scratch/cscs/xyao/xkernels`; the script itself handles the container environment (`tokenspeed-rocm-aiter-myofi`) and GPU allocation.
- `.rcc/rccignore` excludes local-only files (`.git/`, `.rcc/`, `.claude/`, caches, build artifacts) so syncs stay small.

### Why the `bash -c 'cd REPO && ...'` wrapper for `srun`?

The `tokenspeed-rocm-aiter-myofi` container image has its own default working directory (`.../tokenspeed-amd`). Passing `--chdir` to `srun` is not enough because the container changes directory after startup. The SLURM scripts avoid this by running `cd "$REPO"` inside their own `bash -c` wrapper. For interactive `srun` commands, use the same pattern.

## Notes

- The head node has MI250X GPUs; for MI300A (gfx942) results, always use `--partition=mi300` or submit the provided `slurm/*_beverin.sbatch` scripts.
- The container environment `tokenspeed-rocm-aiter-myofi` provides torch 2.11.0+rocm7.2 and the Triton build used by the kernels.

## See also

- [`docs/profiling-on-beverin.md`](profiling-on-beverin.md) — setting up + running AMD's ROCm Compute Profiler (Omniperf) for wave-level occupancy/stall/roofline diagnosis.
- [`.agents/skills/use-rocprof-compute/SKILL.md`](../.agents/skills/use-rocprof-compute/SKILL.md) — the procedural skill that routes a profile to the `diagnose-*` / `map-to-matrix-cores` fix skills.
