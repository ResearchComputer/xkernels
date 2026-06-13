# Issue #39 — V4 sparse-MLA (#33) + MHC prenorm GEMM (#37) perf pass (gfx942)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2,
`tokenspeed_triton` AMD backend. **Bench/test:**
`benchmarks/tune_mhc_prenorm_gemm.py`, `benchmarks/tune_sparse_mla.py`
(`slurm/tune_v4_perf_beverin.sbatch`), job **384616**. Metric: median ms, Triton
`do_bench`.

## TL;DR

The two V4 kernels landed correctness-first (#33, #37). This perf pass turns their
launch parameters (block sizes + CDNA3 lowering knobs `waves_per_eu` /
`matrix_instr_nonkdim` / `kpack`) into a small, env-overridable config and
characterizes the candidate space on real gfx942.

- **MHC prenorm GEMM: a clean, uniform win.** `BLOCK_M=32, BLOCK_K=128,
  waves_per_eu=4` is the fastest config at *every* decode batch size (T=1/8/64) —
  **1.48–1.63× faster than the #36 baseline** (BLOCK_M=BLOCK_K=64), correctness
  preserved (rel_err ~5e-4). **Promoted to the default.**
- **Sparse-MLA: no static winner.** The best `BLOCK_N` depends on the query-token
  count. At single-token decode (Tq=1) `BLOCK_N=128, num_warps=8` is **1.13–1.24×**
  faster; at Tq>1 the #33 default `BLOCK_N=64` is already best and larger BLOCK_N
  *regresses* (up to ~4.3× slower at Tq=8 topk=1024 — the `[D]=512` fp32
  accumulator dominates VGPR/LDS). So `BLOCK_N=64` **stays the default** (no
  multi-token regression) and the Tq=1 win ships **opt-in, off by default**
  (`DECODE_SPARSE_MLA_CONFIG`), mirroring the #20/#12 precedent.

Correctness is invariant to these knobs by construction (the flash reduction is
exact for any chunk size; the GEMM split-K only sums over splits), pinned by
`tests/test_issue39_perf_pass.py`.

## MHC prenorm GEMM (V4-Flash shape K=16384, N=24, splits=16)

| T | baseline #36 (ms) | best #39 (ms) | speedup | best cfg | rel_err |
|--:|------------------:|--------------:|--------:|----------|--------:|
| 1  | 0.0184 | **0.0117** | **1.57×** | BM=32 BK=128 wpe=4 | 5.0e-04 |
| 8  | 0.0191 | **0.0129** | **1.48×** | BM=32 BK=128 wpe=4 | 3.9e-04 |
| 64 | 0.0213 | **0.0130** | **1.63×** | BM=32 BK=128 (BK=256 ties) | 4.0e-04 |

Why the winner wins: the problem is memory-bound (stream A once over K=16384,
tiny N=24). The smaller `BLOCK_M=32` packs the tiny-T rows tighter and frees
VGPRs, letting `waves_per_eu=4` raise occupancy to hide the K-stream global-load
latency; `BLOCK_K=128` doubles the per-load A/fn read width. The #36 launch left
all of this on the table (default occupancy, no AMD knobs, BLOCK_K=64).

**LDS limit found on-device.** `BLOCK_K=256` fp32 at `num_stages=2` needs 96 KB and
raises `OutOfResources(98304, 65536)` — CDNA3 has 64 KB LDS/CU. The 256-wide
candidates are pinned to `num_stages=1` (48 KB), and the sweep/tests treat
`OutOfResources` as "config infeasible here", not a failure. `BLOCK_M=128,BLOCK_K=128`
also OOMs and is skipped by the sweep.

(Both the kernel time and the ~120–230× vs `F.linear+sqsum` reflect that the
naive fp32 `F.linear` hits the slow dense-GEMM path on this stack — the #17
cliff — so a practitioner's torch replacement would indeed pay it.)

## Sparse-MLA (V4 geometry H=128, D=512, d_v=448, MQA)

`BLOCK_N` is a pure perf knob (the flash reduction is exact for any chunk size).
Best config and speedup vs the #33 default (`BLOCK_N=64`):

| Tq | topk | base #33 (ms) | best (ms) | speedup | best cfg |
|---:|-----:|--------------:|----------:|--------:|----------|
| 1 | 256  | 0.0206 | **0.0182** | **1.13×** | BN=128 w8 wpe=1 |
| 1 | 512  | 0.0325 | **0.0272** | **1.20×** | BN=128 w8 wpe=1 |
| 1 | 1024 | 0.0570 | **0.0459** | **1.24×** | BN=128 w8 wpe=1 |
| 8 | 256  | 0.0596 | 0.0591 | 1.01× | BN=64 (default) |
| 8 | 512  | 0.1107 | 0.1091 | 1.01× | BN=64 (default) |
| 8 | 1024 | 0.2110 | 0.2094 | 1.01× | BN=64 (default) |

At Tq>1 the larger-BLOCK_N configs are *worse* — e.g. Tq=8 topk=1024 `BLOCK_N=256`
is 0.90 ms (≈4.3× slower than 0.21 ms). The wide chunk inflates the per-program
score/value tiles and the `[D]=512` fp32 accumulator, cutting occupancy. Only the
Tq=1 regime (grid is just H=128 programs) benefits from a wider chunk + more warps.

## What ships

- `ops/mhc/triton/configs.py` — `DEFAULT_MHC_GEMM_CONFIG` promoted to the measured
  winner (`BLOCK_M=32, BLOCK_K=128, waves_per_eu=4`); `BASELINE_MHC_GEMM_CONFIG`
  retains the #36 launch for A/B. The wrapper threads the AMD lowering knobs.
- `ops/attention/triton/sparse_mla_config.py` — `DEFAULT_SPARSE_MLA_CONFIG`
  unchanged from #33 (`BLOCK_N=64`); `DECODE_SPARSE_MLA_CONFIG` is the opt-in
  Tq=1 win.
- Both kernels read `XKERNELS_MHC_GEMM_CONFIG` / `XKERNELS_SPARSE_MLA_CONFIG`
  (JSON dict, partial override allowed) so a deployment can pin a regime-specific
  config without code changes. The AMD knobs are ignored by non-AMD Triton and
  under `TRITON_INTERPRET=1`, so everything stays portable.
- `benchmarks/tune_mhc_prenorm_gemm.py`, `benchmarks/tune_sparse_mla.py`,
  `slurm/tune_v4_perf_beverin.sbatch` reproduce the sweep.

## Validation

- Offline (`TRITON_INTERPRET=1` CPU + GPU): `tests/test_issue39_perf_pass.py`
  pins config resolution and the numerical invariance of the result under the
  perf knobs (BLOCK_N for sparse-MLA, BLOCK_K for the GEMM), plus the unchanged
  #33/#37 suites.
- On-device (beverin / MI300A, job 384616): **41 passed** (kernel suites + the
  #39 invariance tests, real Triton compile at bf16) and the full config sweep
  above.

## Not done / future

- An MFMA-tiled score/value path for sparse-MLA (the issue's other candidate)
  would change the kernel structure (currently `tl.sum`, not `tl.dot`); the
  `matrix_instr_nonkdim`/`kpack` knobs are already threaded for it but inert
  today. Split-KV + `mha_merge_state` for very long top-k is also deferred.
- A regime-aware auto-select (Tq==1 → decode config) could fold the sparse-MLA
  opt-in into the wrapper, but is left explicit/opt-in here to avoid a hidden
  multi-token regression.
