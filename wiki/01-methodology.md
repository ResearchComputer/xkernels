# Benchmark & profile campaign — methodology

How every kernel in `registry/impls/*.triton.card.json` was benchmarked and
profiled across both vendor clusters, and what the numbers in the sibling pages
mean. Reproduce any cell by following the commands here.

## The two clusters

| Cluster | GPU | Arch id | Software stack | Profiler |
|---|---|---|---|---|
| **beverin** | AMD Instinct MI300A | `gfx942` (CDNA3) | `tokenspeed-rocm-aiter-myofi` uenv (torch 2.11 + rocm 7.2) | ROCm Compute Profiler (rocprof-compute, ex-Omniperf) |
| **bristen** | NVIDIA A100-SXM4-80GB | `sm_80` (Ampere) | NGC PyTorch `24.10-py3` container (torch 2.5.0a0 + **triton 3.0.0**) | Nsight Compute (`ncu`) + Nsight Systems (`nsys`) |

`/capstor/scratch/cscs/xyao/xkernels` is **shared** between the two clusters, so
one `rcc push` syncs the tree to both; select the cluster with
`rcc --profile {beverin,bristen}` (default is `beverin`).

## The 10 ops

All cards are portable Triton (`arch.family = any`), so they run on both arches
— **except `mm_fp8_blockscale`**, which uses the gfx942-only `e4m3fnuz` fp8 MFMA
path (sm_80 has no hardware fp8). Regime is taken from each card's
`perf.roofline`:

- **memory-bound** (7): `dual_rmsnorm`, `hc_prenorm_gemm`, `mha_merge_state`,
  `mhc_pre`, `moe_align_block_size`, `moe_sum_reduce`, `sparse_mla_attention`.
- **compute-bound** (3): `fused_ffn`, `mm_fp8_blockscale`, `moe_int4_w4a16`.

## Benchmark harness

- **beverin**: `slurm/bench_all_beverin.sbatch` → `benchmarks/bench_all.py`
  (9 ops) + `slurm/bench_fp8_blockscale_beverin.sbatch` →
  `benchmarks/bench_fp8_blockscale_gemm.py` (the 10th, gfx942-only).
- **bristen**: `slurm/bench_all_bristen_isolated.sbatch` → a shell loop that
  runs `benchmarks/bench_one.py <kernel>` **once per kernel in its own process**
  (process isolation — see `gotchas.md#Triton-3.0.0-OptimizeThreadLocality-SIGSEGV`).

Each cell is **median of Triton `do_bench`** (`xkernels.utils.benchmarking.benchmark`),
bf16 unless noted (FFN is fp16 — see gotchas), optimized backend vs the naive
PyTorch a practitioner would write. Shapes mirror the README "Performance"
table (Kimi-K2.6 / V4 serving regime).

## Profiling harness

One builder per op lives in `benchmarks/probe_omniperf.py` (AMD) and
`benchmarks/probe_ncu.py` (NVIDIA) — the twins share identical seeded shapes so a
beverin profile is directly comparable to a bristen one. Each builder warms up
5× then runs 10 steady-state iterations so the profiled dispatch is clean.

- **beverin** (`slurm/profile_omniperf_beverin.sbatch` →
  `scripts/profile-rocprof-compute-beverin.sh`): `MODE=roof` =
  roofline + default metric set. Output `.omniperf-workloads/<kernel>_roof/`
  + `<kernel>_roof.analyze.txt`.
- **bristen** (`slurm/profile_ncu_bristen.sbatch` →
  `scripts/profile-ncu-bristen.sh`): `MODE=roof` = SpeedOfLight +
  Compute/Memory Workload + Occupancy + LaunchStats. Output
  `.ncu-workloads/<kernel>_roof/` (`.report.txt`, `.sol.csv`, `.ncu-rep`).

**Profiling subset note.** The native probes cover the 4 representative kernels
(dual_rmsnorm, moe_sum_reduce, fused_ffn, mha_merge_state) on both arches plus
the rest on beverin; on bristen the set is further limited by which kernels the
container's Triton 3.0.0 can compile (see gotchas). The 4 representative kernels
span **both regimes** (memory-bound reduce/merge + compute-bound GEMM), so the
roofline vocabulary they yield generalizes to the regime-mates.

## Field definitions (what `verify()` stubs vs what we compute)

`verify(..., measure_perf=True)` returns only `ms`; `tflops` and
`achieved_bw_pct` are `None` (open question, library §11). We derive them from
the profile / the bench shape:

```
tflops   = 2 * FLOPs_of_op(shape) / (ms / 1e3)
bw_pct   = bytes_moved / (ms / 1e3) / peak_HBM
```

Device peaks (measure on-device to confirm): MI300A ≈ 130.7 TF/s FP16/BF16 MFMA
(fp8 ≈ 2×), ≈ 819 GB/s HBM3; A100-80GB SXM4 ≈ 9.7 TF/s FP16/BF16 tensor, ≈ 1.55
TF/s FP32, ≈ 1.94 TB/s HBM2e.

## Reproduce cheatsheet

```bash
# bench (beverin)
scripts/bench-on-beverin.sh                              # 9-op table
scripts/bench-on-beverin.sh slurm/bench_fp8_blockscale_beverin.sbatch
# bench (bristen, isolated)
rcc --profile bristen run -- env REPO=/capstor/scratch/cscs/xyao/xkernels \
    sbatch slurm/bench_all_bristen_isolated.sbatch
# profile (beverin)
KERNEL=dual_rmsnorm scripts/bench-on-beverin.sh slurm/profile_omniperf_beverin.sbatch
# profile (bristen)
KERNEL=dual_rmsnorm scripts/bench-on-bristen.sh slurm/profile_ncu_bristen.sbatch
```
