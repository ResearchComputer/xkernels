# scripts/ — remote-execution toolkit + job library

Everything here drives work on the HPC clusters via
[`rcc`](https://github.com/ResearchComputer/remote-cluster-controller) (sync the
local tree → run/submit on a remote host). Hosts are defined in `.rcc/config.toml`:

| profile   | host     | GPU                  | role                                  |
|-----------|----------|----------------------|---------------------------------------|
| `beverin` | beverin  | AMD MI300A (gfx942)  | AMD benchmark/profile target          |
| `bristen` | bristen  | NVIDIA A100 (sm_80)  | NVIDIA benchmark/profile target       |
| `ds5`     | ds5      | NVIDIA GB10 (sm_121) | CUTE-DSL test bed (no SLURM)          |

`rcc push` syncs (`.rcc/rccignore` drops `.git/`, `.venv/`, caches). The toolkit
below adds the per-host mechanics (container image, `srun`, DCGM pause, `PYTHONPATH`,
venv/CUDA_HOME) that a bare `rcc run` does not.

## Layout

```
scripts/
  cluster.sh                 THE entry point: push + run/submit on any host
  vkl_artifacts.py           emit/check registry JSON materialized from VKL sources
  profile-*.sh               per-profiler wrappers (ncu/nsys on bristen, rocprof-compute on beverin)
  setup-/stage-rocprof-*     one-time profiler install + lib staging on beverin
  bench_kernel_loop_bristen.sh   per-kernel isolation harness (see meta/wiki/04-gotchas.md)
  slurm/                     supported SLURM jobs (Tier A)
  archive/                   one-shots & exploration scratch, kept for reproducibility
    ds5-probes/              the ds5 GB10 CUTE-DSL investigation corpus
    campaigns/               dated campaign drivers (record-measurement write-back, PR bench)
    issues/                  per-issue SLURM jobs (one per closed meta/docs/kernels/*.md)
```

## The entry point — `cluster.sh`

Replaces the four per-host shims (`run-on-{beverin,bristen}.sh` +
`bench-on-{beverin,bristen}.sh`) with one host-agnostic command, and adds `ds5`
coverage. `--host` defaults to `beverin`.

```bash
# run a command interactively on a compute node
scripts/cluster.sh run --host beverin -- python3 -u meta/benchmarks/bench_all.py
scripts/cluster.sh run --host bristen -- python3 -u meta/benchmarks/bench_all.py
scripts/cluster.sh run --host ds5     -- python -m xkernels.ops._cute_backend.smoke_vecadd

# submit a SLURM job (beverin / bristen only — ds5 has no SLURM)
scripts/cluster.sh submit --host beverin scripts/slurm/bench_all_beverin.sbatch
scripts/cluster.sh submit --host bristen                         # default: bench_all_bristen.sbatch
KERNEL=dual_rmsnorm MODE=sq scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
```

For `run`, `--` separates `cluster.sh`'s flags from your command (optional unless
the command's first token starts with `-`). bristen `run` honors the
`BRISTEN_IMAGE` / `BRISTEN_PARTITION` / `BRISTEN_ACCOUNT` / `BRISTEN_GPU` /
`BRISTEN_TIME` env overrides.

## VKL registry artifacts

VKL-authored kernels still materialize registry JSON for the card-driven
substrate. Treat those files as generated artifacts:

```bash
scripts/vkl_artifacts.py check     # fail if checked-in JSON drifted from VKL
scripts/vkl_artifacts.py write     # regenerate Op Specs + reference/backend cards
scripts/vkl_artifacts.py list      # list VKL-managed op short names
```

The checker preserves mutable card state (`perf.measured`, tuning traces, and
creation timestamps), so autotune/write-back data remains owned by the card.

## Profilers (Tier A)

| wrapper                      | does                                                     |
|------------------------------|----------------------------------------------------------|
| `profile-ncu-bristen.sh`     | Nsight Compute per-kernel profile (roof/sq/full)         |
| `profile-nsys-bristen.sh`    | Nsight Systems timeline                                  |
| `profile-rocprof-compute-beverin.sh` | ROCm Compute Profiler (ex-Omniperf) profile+analyze |
| `setup-rocprof-compute-beverin.sh`   | one-time profiler install into scratch            |
| `stage-rocprof-compute-libs-beverin.sh` | one-time staging of rocprofv3 runtime `.so`s    |

These are driven through the supported SLURM jobs below (which do the host-side
DCGM pause etc.). Referenced by `meta/docs/usage/clusters.md` and the
`.agents/skills/use-*-compute/` skills.

## Supported SLURM jobs (`scripts/slurm/`)

`bench_all_{beverin,bristen,bristen_isolated}.sbatch`,
`profile_{ncu,nsys}_bristen.sbatch`, `profile_omniperf_beverin.sbatch`. Submit
any of them via `scripts/cluster.sh submit --host <h> scripts/slurm/<job>.sbatch`.

## Archive (`scripts/archive/`)

One-shots and exploration scratch, kept as the reproducibility trail — not
maintained, not on any supported path. See [`archive/README.md`](archive/README.md).

## Lifecycle policy

- A **new benchmark/profiler/job** meant to be reused → Tier A: a `slurm/` job +
  `cluster.sh submit` (and/or a `profile-*` wrapper), linked from the relevant
  `meta/docs/usage/clusters.md`.
- A **one-shot for a specific issue or campaign** → write it next to its work,
  then move it under `scripts/archive/` when the issue closes. Cite the script
  path in the card/issue doc so the measurement is reproducible.
- An **exploratory probe** (API discovery, "does X work?") → expect it to become
  archive material. Put its *conclusion* in a doc/card, not just the script.
