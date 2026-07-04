# xkernels wiki — the shared knowledge base

The project's knowledge base of **facts that cost real debugging time** —
numbers, roofline diagnoses, reverse-engineered API surfaces, and host-side
gotchas, recorded so the next agent (or human) doesn't re-derive them.

This is the **shared layer** that cuts across all the per-kernel docs in
[`../docs/kernels/`](../docs/kernels/): the per-op *narrative* (what a kernel
computes + its measured numbers + why it won/lost) lives there, while the
*cross-cutting method* (how the numbers were produced, what the profiler
revealed, which host traps bit us) lives here. Read a `kernels/*.md` for the op,
then come here for the method behind its numbers.

It grew out of the **benchmark & profile campaign (2026-06-26)** — a full pass
over every kernel in `registry/impls/*.triton.card.json` (10 ops × 2 cards) on
both vendor clusters, **beverin (AMD MI300A, gfx942)** and **bristen (NVIDIA
A100, sm_80)** — and now also holds the authoring reference for the CUTE DSL
`cuda` backend (page 5).

> **TL;DR.** All 10 ops benchmark on MI300A (6–204× over naive PyTorch; fp8 MFMA
> hits 363 TFLOP/s). On A100, **6/9 portable ops run**; 3 are blocked by a
> Triton-3.0.0 compiler/`waves_per_eu` portability gap (see
> [`04-gotchas.md`](04-gotchas.md) §1/§1b) — *not* kernel bugs. The one clear
> performance-tune target is `moe_int4_w4a16` (register-pressure-capped
> occupancy); most memory-bound kernels are already at the HBM roofline.

## Pages

**The benchmark & profile campaign (2026-06-26):**

1. **[01-methodology.md](01-methodology.md)** — clusters, harness, field
   definitions, reproduce cheatsheet.
2. **[02-benchmarks.md](02-benchmarks.md)** — full speedup tables, both arches +
   the fp8 sweep, and a cross-arch read.
3. **[03-profiling.md](03-profiling.md)** — roofline / regime validation per
   kernel, occupancy, and which fix skill each profile routes to.
4. **[04-gotchas.md](04-gotchas.md)** — the experiences (Triton SIGSEGV,
   `waves_per_eu`, bf16 GEMM pathology, fp8 `fnuz`, DCGM pause, rocprof install).

**Authoring reference (2026-06-29, CUTE DSL `cuda` backend on ds5/GB10):**

5. **[05-cutedsl-authoring.md](05-cutedsl-authoring.md)** — how to write a
   `cutlass.cute` (CUTE DSL) kernel: the three-function structure, the
   `cute.compile` compile-cache (the 119× launch-overhead fix), the math/
   reduction primitive calling conventions, the bf16-native-read perf lever,
   and the negative results. The API surface was reverse-engineered by grep
   + GPU probe (no tutorial docs exist) — this page is the map.

**vkl Phase D (2026-07-04, native HIP codegen + the H1/H2 count, issue #75):**

6. **[06-vkl-phased.md](06-vkl-phased.md)** — the native HIP `load_inline`
   spellings (`__hip_bfloat16`, `at::cuda::getCurrentCUDAStream()`, the
   `PYTORCH_ROCM_ARCH=gfx942` pin) that ship `lower/hip.py` at the FMA
   mechanism-validation bar; the reverse-engineered MFMA codegen surface on
   gfx942 (only `__builtin_amdgcn_mfma_f32_32x32x4bf16` exists; archdb's
   `{m:32,k:16}` matches no instruction — the map for the MFMA follow-up); the
   H1/H2 named-edit-frequency methodology (compute→BLAS vs bandwidth→DRAM-
   roofline) and the empirical finding that H1 is needed only for compute-bound
   ops; plus the drift-gate / no-sm_90-host / rcc-quote-stripping gotchas.

## Headline numbers

| Kernel | MI300A opt (ms) | MI300A speedup | A100 opt (ms) | A100 speedup | Regime (measured) |
|---|---:|---:|---:|---:|---|
| `mha_merge_state` | 0.784 | 3.1× | 1.046 | 4.9× | balanced |
| `sparse_mla_attention` | 0.111 | 27.0× | ✗ | ✗ | compute (high AI) |
| `mhc_prenorm_gemm` | 0.013 | ≫100×⚠ | ✗ | ✗ | launch-bound (T=8) |
| `mhc_pre` (+post) | 0.080 | 34.5× | ✗ | ✗ | memory |
| `dual_rmsnorm` | 0.054 | 4.4× | 0.053 | 9.8× | memory (at roofline) |
| `moe_sum_reduce` | 0.373 | 8.4× | 0.651 | 10.3× | strongly memory |
| `moe_align_block_size` | 1.644 | 33.7× | 0.883 | 75.4× | dispatch-bound |
| `fused_ffn` (fp16) | 5.285 | 1.0× | 4.288 | 1.1× | compute (torch.matmul GEMMs; Triton kernel = SwiGLU only) |
| `moe_int4_w4a16` | 1.364 | 23.4× | 2.225 | 25.8× | register-pressure-capped |
| `mm_fp8_blockscale` | 0.331¹ | 5.9×¹ | N/A (no sm_80 fp8) | — | compute (363 TFLOP/s) |

¹ largest V4 shape (M=4096, N=7168, K=2048); see [`02-benchmarks.md`](02-benchmarks.md)
for the full fp8 sweep. ⚠ launch-overhead-dominated; treat as ≫100×, not precise.
✗ = blocked on A100 by Triton-3.0.0 portability gaps, not a correctness bug.

## What changed in the repo for this campaign

- `meta/benchmarks/probe_{omniperf,ncu}.py` — extended from 4 → **all 10 ops** as
  profilable single-kernel workloads (identical seeded shapes across arches).
- `meta/benchmarks/bench_one.py` + `scripts/slurm/bench_all_bristen_isolated.sbatch` +
  `scripts/bench_kernel_loop_bristen.sh` — per-kernel **process isolation** so a
  native Triton SIGSEGV only loses one row.
- `scripts/archive/issues/bench_fp8_blockscale_beverin.sbatch` — standalone gfx942 fp8 bench.
- `scripts/profile-ncu-bristen.sh` — kernel→fragment map extended to all 10 ops.
- `scripts/archive/campaigns/record_campaign_measurements.py` — writes the 17 campaign points into
  the cards' `perf.measured` (re-runnable; dedups by point).
- **10 `registry/impls/*.triton.card.json`** — `perf.measured` now populated
  (was `[]` for all).

## Follow-ups this campaign surfaced

- **Gate the AMD-only Triton kwargs** (`waves_per_eu`, `matrix_instr_nonkdim`,
  `kpack`) behind arch detection so `sparse_mla`, `mhc_prenorm_gemm`, `mm_fp8`
  run on NVIDIA Triton too (currently `arch.family=any` but AMD-hardcoded).
- **Bump the bristen container** to a ≥25.x NGC image (Triton ≥3.1) to clear the
  `OptimizeThreadLocality` SIGSEGV on `mhc_pre`.
- **DONE — `perf.measured` written back.** The 17 bench/profile points from this
  campaign are now recorded on the cards (9 beverin + 2 beverin fp8 + 6 bristen),
  via `scripts/archive/campaigns/record_campaign_measurements.py` (the sanctioned
  `record_measurement` path). Each entry cites a reproducible SLURM `source` +
  `arch`; the 3 A100-blocked ops have no `nvidia_sm80` entry (correctly absent).
  All 20 cards still schema-validate; 36 registry tests pass.
- **Profiler-skill depth gaps** (from diffing against
  `use-rocprof-compute` / `use-nsight-compute` — see `03-profiling.md` § "Relation
  to the profiler skills"): (a) run **`sq` mode** on both arches to get the
  **dominant stall reason** this pass lacks (routing is currently ratio+occupancy
  only); (b) derive a **per-op `bytes_moved` model** so the MI300A
  `achieved_bw_pct` card field can be filled via the skill's analytical formula
  (currently `null`); (c) use **dispatch isolation** (`-d`/`-k`) on the beverin
  rocprof runs so tiny-T kernels aren't polluted by `torch.randn` setup dispatches.
