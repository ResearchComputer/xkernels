# Issue #17 — Characterize the bf16 dense-GEMM MFMA/hipBLASLt cliff on gfx942

**Date:** 2026-06-11
**Status:** Approved (Phase 1: characterization)
**Issue:** ResearchComputer/kernels#17
**Target hardware:** AMD Instinct MI300A (gfx942, CDNA3), torch 2.11+rocm7.2

## Purpose

The README perf note records a stack cliff: on this torch 2.11+rocm7.2 build the
**bf16** dense GEMM misses the MFMA/hipBLASLt fast path and runs **~470× slower
than fp16** (0.8 vs 358 TFLOP/s) for the `fused_ffn` probe shape. tokenspeed
serves Kimi-K2.6 in bf16 and has several large non-quantized GEMMs that go
through `torch.matmul` / `F.linear` (MLA projections, the first dense layer's MLP
and every layer's shared-expert MLP, `lm_head`). If they hit the same slow path
they are large hidden decode/prefill bottlenecks.

**This is a two-phase issue and the fix is data-dependent.** A 470× cliff to
0.8 TFLOP/s is not "suboptimal tile selection" — it looks like bf16 falling off
the MFMA tensor path into a scalar fallback, which is frequently a routing/env
problem (an env var or `torch.backends` toggle), not a missing kernel. Phase 1
(this spec) **characterizes** which bf16 dense-Linear shapes fall off the fast
path and whether any blas-routing knob restores it, plus measures a Triton
`tl.dot` bf16 baseline as the achievable ceiling. The actual fix — document an
env/routing remedy **or** ship an MFMA-safe Triton bf16 GEMM — is **Phase 2**,
decided from the Phase-1 data.

## Scope (Phase 1 only)

In scope: extend `benchmarks/probe_ffn.py` with a dense-Linear sweep; add blas
mode switching; add a Triton bf16 GEMM baseline; a SLURM job that runs all modes;
run on beverin; write a findings doc with a recommendation.

Out of scope (Phase 2, follow-up): registering a production bf16 GEMM backend,
wiring it into `fused_ffn` / tokenspeed, editing the README perf note, or
changing default torch backend settings repo-wide.

## Shapes swept

Kimi-K2.6 per-rank dense / MLA / shared-expert / head shapes the issue names,
expressed as `(K → N)` (`F.linear` weight is `[N, K]`):

| Tag | K | N | Source |
|-----|---|---|--------|
| `q_a_proj`      | 7168 | 1536  | MLA q down-proj |
| `kv_a_proj`     | 7168 | 576   | MLA kv down-proj |
| `shexp_gate_up` | 7168 | 2048  | shared-expert gate/up |
| `shexp_down`    | 2048 | 7168  | shared-expert down |
| `lm_head`       | 7168 | 32768 | output proj (representative large vocab tile) |
| `ffn_gate_up`   | 4096 | 11008 | existing probe FFN (continuity) |

M (rows) swept: decode `{1, 2, 4, 8, 16, 32}` and prefill `{512, 2048, 4096}`.

For each `(M, K, N) × {fp16, bf16}` the probe reports latency, TFLOP/s, and the
**bf16/fp16 TFLOP/s ratio**, flagging shapes whose ratio is far below 1 (the
fast-path miss). A small ratio threshold (e.g. < 0.5) marks a "MISS".

## Blas modes

The probe takes `--mode`; the SLURM job runs it once per mode (several knobs are
read at torch import, so per-process isolation is the correct mechanism):

| Mode | What it sets | Hypothesis |
|------|--------------|-----------|
| `default`      | nothing (as the engine runs today) | reproduce the 470× cliff |
| `hipblaslt`    | `TORCH_BLAS_PREFER_HIPBLASLT=1` + `torch.backends.cuda.preferred_blas_library("hipblaslt")` | hipBLASLt has a bf16 MFMA solution rocBLAS lacks |
| `no-hipblaslt` | `TORCH_BLAS_PREFER_HIPBLASLT=0` | isolate whether hipBLASLt is the one regressing |
| `tunableop`    | `PYTORCH_TUNABLEOP_ENABLED=1` (+ tuning enabled) | TunableOp tunes the bf16 GEMM onto MFMA |

The probe applies runtime-settable knobs itself and reads the import-time env the
job set; it prints the active mode and the resolved
`torch.backends.cuda.preferred_blas_library()` so the log is self-describing.

## Triton bf16 baseline

A minimal, single-config `@triton.jit` `tl.dot` bf16 GEMM (fp32 accumulate),
timed on the same shapes. This is **measurement only** for Phase 1 — it tells us
the MFMA ceiling a Triton drop-in could reach. If torch bf16 is 0.8 TFLOP/s but
Triton bf16 reaches ~hundreds, a kernel is a viable fix; if a blas mode already
restores torch bf16, no kernel is needed. (If Phase 2 chooses the kernel, this
graduates into a proper registered backend; it is not registered here.)

## Architecture / components

1. **`benchmarks/probe_ffn.py`** (extend): keep the existing fp16/bf16 matmul +
   FFN probe; add
   - `_apply_blas_mode(mode)` — set runtime backend knobs, read env, print state;
   - `_bench_gemm(M, K, N, dtype)` → (ms, TFLOP/s) via CUDA events;
   - `_triton_bf16_gemm(...)` + its launcher (single fixed tile, fp32 acc);
   - `sweep(mode, Ms)` — loop the shape table × M × {fp16, bf16, triton-bf16},
     print one markdown table with the bf16/fp16 ratio and a MISS flag;
   - `--mode` / `--shapes` CLI; default runs the dense sweep.
2. **`slurm/probe_dense_bf16_beverin.sbatch`** — runs the probe for each mode in
   sequence (env set per invocation), one combined log.
3. **`docs/issue-17-bf16-dense-gemm.md`** — the findings: per-mode tables, which
   shapes miss, whether any mode restores MFMA, the Triton ceiling, and a
   **recommendation** for Phase 2 (env-doc vs Triton kernel).

This is a benchmark/diagnostic tool, so it lives entirely under `benchmarks/` +
`slurm/` + `docs/`; it imports nothing new into `src/` and registers no backend.

## Data flow

`mode → env/runtime knobs applied → for each (shape, M, dtype): time GEMM → TFLOP/s
table + ratio`. The Triton path runs the same shapes for a ceiling column. The
job aggregates modes; the findings doc compares them.

## Error handling / edge cases

- No GPU → print a clear message and exit 0 (matches existing probe behavior);
  the kernel sweep needs a real device.
- A blas knob unavailable on this torch (e.g. `preferred_blas_library` missing) →
  warn and continue in whatever mode torch defaults to, printing the resolved
  state so the log stays honest.
- `lm_head` N=32768 prefill M=4096 is a large alloc (~bf16 0.5–2 GB transient) —
  fine on MI300A's 128 GB; guard with a try/except that records OOM as a skipped
  cell rather than aborting the sweep.
- Triton bf16 GEMM is correctness-sanity-checked against `torch.matmul` (fp32
  ref) once at a small shape before timing, so a wrong kernel can't masquerade as
  a fast one.

## Testing

- **Local (`TRITON_INTERPRET=1`, no GPU):** a unit test that the Triton bf16 GEMM
  matches `torch.matmul` on a small shape (fp32 accumulate path), and that
  `_apply_blas_mode` runs without error for every mode and reports a state. The
  TFLOP/s sweep itself is GPU-only (skipped locally).
- **On device (beverin, gfx942):** run the SLURM job across all modes; confirm
  the `default` mode reproduces the cliff and capture whether any mode restores
  the fast path; record the Triton ceiling.

## Deliverable acceptance (Phase 1)

- `probe_ffn.py` sweeps the dense shapes across M and the four blas modes, with a
  Triton bf16 ceiling column, and flags fast-path misses.
- SLURM job produces one comparison log on MI300A.
- `docs/issue-17-bf16-dense-gemm.md` states, with numbers: which bf16 dense
  shapes miss MFMA, whether an env/routing mode fixes it, the Triton ceiling, and
  a concrete Phase-2 recommendation.
- Findings reported on issue #17. (Phase 2 fix is a separate PR.)
