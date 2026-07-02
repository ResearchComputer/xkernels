# Experiences & gotchas (the facts that cost real debugging time)

Recorded from the 2026-06-26 full-library benchmark + profile campaign across
beverin (MI300A) and bristen (A100). Each entry is a concrete, reproducible
fact — not advice.

## 1. Triton 3.0.0 `OptimizeThreadLocality` SIGSEGV on sm_80 (bristen) — THE big one

**Symptom.** `meta/benchmarks/bench_all.py` on bristen dies with `Caught signal 11
(Segmentation fault)` *before printing a single result row*. The backtrace is
entirely inside Triton's MLIR compiler:

```
mlir::triton::gpu::TritonGPUOptimizeThreadLocalityPass::runOnOperation()
  .../OptimizeThreadLocality.cpp:124   (processing a triton::ReduceOp)
```

**Cause.** The NGC `pytorch:24.10-py3` container ships **Triton 3.0.0**
(torch 2.5.0a0). Its `OptimizeThreadLocality` pass has a bug that segfaults when
rewriting a `ReduceOp`'s loads on sm_80. The crash happens at **JIT-compile
time** of the first reduction-bearing kernel — so it is a *native* SIGSEGV, **not
a Python exception**: `bench_all.py`'s per-kernel `try/except` cannot catch it,
and the whole process dies, losing every subsequent row.

**Mitigation that works.** Run each kernel's bench in its **own process** so one
speedbump only loses that one row. `meta/benchmarks/bench_one.py` wraps a single
`bench_all` function; `scripts/slurm/bench_all_bristen_isolated.sbatch` (+
`scripts/bench_kernel_loop_bristen.sh`) loops over kernels calling it (`set +e`
so the loop survives a per-kernel SIGSEGV). This recovered 6/9 rows and
pinpointed the failures (see next two entries).

**Per-kernel bristen outcome (9-op `bench_all`, Triton 3.0.0 / sm_80):**

| kernel | result | failure |
|---|---|---|
| mha_merge_state, dual_rmsnorm, moe_sum_reduce, moe_align_block_size, fused_ffn, moe_int4_w4a16 | ✓ OK | — |
| mhc_pre_post | ✗ rc=139 | **OptimizeThreadLocality SIGSEGV** (this entry) |
| sparse_mla, mhc_prenorm_gemm | ✗ rc=1 | **`waves_per_eu` KeyError** (next entry) |

So the original whole-process death was `mhc_pre_post` (4th in loop order):
`merge_state` ran, `sparse_mla` + `mhc_prenorm_gemm` raised *catchable* KeyErrors
(recorded but never printed — the table prints only at the end), then
`mhc_pre_post`'s native SIGSEGV killed the process before the table flushed.

**Open.** A newer NGC image (≥ 25.x, Triton ≥ 3.1) likely fixes the pass; not
tested in this campaign because the isolation loop already recovers the data.

## 1b. `waves_per_eu` (AMD-only Triton kwarg) → `KeyError` on NVIDIA Triton

**Symptom.** `sparse_mla` and `mhc_prenorm_gemm` abort with
`KeyError: 'Keyword argument waves_per_eu was specified but unrecognised`.

**Cause.** `waves_per_eu` (with its siblings `matrix_instr_nonkdim` and
`kpack`) is an **AMD-CDNA-specific** Triton autotune/launch kwarg. It is threaded
into the launch meta by the AMD-tuned configs in
`ops/attention/triton/sparse_mla_*`, `ops/mhc/triton/configs.py`, and
`ops/gemm/triton/configs.py` (the `mm_fp8` MFMA kernel even declares it as a
`tl.constexpr`). NVIDIA's Triton 3.0.0 (the 24.10 container) does not know the
kwarg and rejects it at launch. The **portable** kernels (moe_int4, fused_ffn,
rmsnorm, merge_state, sum_reduce, align) do not pass it and run fine.

**This is a real portability gap, not a profiler artefact.** The contract says
portability lives in the card, not the source — and the cards are
`arch.family = any` — yet these three op families hardcode an AMD-only kwarg into
the launch path. A correct fix is to gate the AMD kwargs behind arch detection
(only emit `waves_per_eu`/`matrix_instr_nonkdim`/`kpack` when the build
recognizes them, mirroring how `moe_int4_w4a16` already stays portable). Out of
scope for a benchmark/profile pass, but flagged for a follow-up.

## 2. bf16 GEMM misses the MFMA/hipBLASLt path on this torch+rocm build (beverin)

`bench_all.py` runs `fused_ffn` in **fp16, not bf16**, with a precise reason
documented in-line: on the `tokenspeed-rocm-aiter-myofi` build (torch 2.11 +
rocm 7.2) the **bf16** GEMM misses the MFMA/hipBLASLt fast path and runs ~470×
slower than fp16 (~0.8 vs ~358 TFLOP/s at the FFN shape). FFN is the only
GEMM-bound op in `bench_all.py`, so a bf16 number there would be a pathology, not
a representative figure. Consequence for the table: `fused_ffn` shows only a
**~1.0×** speedup over unfused torch (torch's fp16 path is already optimal
here), which is the honest result. See `meta/benchmarks/probe_ffn.py` for the probe.

## 3. fp8 needs `float8_e4m3fnuz`, not `float8_e4m3fn`, on gfx942

`mm_fp8_blockscale`'s native fp8 MFMA path emits `v_mfma_*_fp8` **only** on
`float8_e4m3fnuz` operands (the AMD CDNA3 fp8 encoding). `float8_e4m3fn` silently
falls back to an f16 MFMA (~30 TFLOP/s instead of ~360+). The bench and the
`mm_fp8_blockscale` probe both quantize to `fnuz`. This is also why
`mm_fp8_blockscale` is **bristen-N/A**: sm_80 has no fp8 tensor cores at all.

## 4. ncu needs a host-side DCGM pause; rocprof-compute does not

On bristen, `ncu`'s kernel-replay grabs the GPU performance counters, which the
node monitor (`/usr/bin/dcgmi` DCGM) holds continuously → `ncu` fails with
*"driver resource unavailable"*. Fix: `dcgmi profile --pause` before `ncu`,
`--resume` after — **from the host** (the sbatch script), not the container
(`scripts/slurm/profile_ncu_bristen.sbatch` traps `--resume` on exit). `nsys` uses
passive CUPTI activity and needs no pause. On beverin the open `amdgpu` driver
has no equivalent contention; rocprof-compute just runs.

## 5. rocprof-compute ("Omniperf") is a source clone + two non-obvious pins

AMD never published it to PyPI; it's a source clone into scratch (read-only
container). Two fixes the setup script bakes in, both silent killers:
- **Pin `pandas<3`.** `requirements.txt` is unbounded → uv grabs pandas 3, whose
  strict `str` dtype breaks the v3→v2 counter join and the analyze metric
  assignment. Profile works; analyze dies.
- **Stage `libdw.so.1` (+deps) from the login node.** `rocprofv3` `dlopen`s it;
  the container lacks it; the host `/usr/lib64` isn't mounted; and
  rocprof-compute resets the profiler subprocess `LD_LIBRARY_PATH` to
  `/opt/rocm/lib` only — so `setup` mirrors the staged libs into `/opt/rocm/lib`
  (writable but per-container-instance → redo every run, which
  `profile-rocprof-compute-beverin.sh` does).

## 6. nvprof is dead on A100; use ncu/nsys

sm_80 (Volta-and-later) has no nvprof profiling support. The binary is present in
the container/HPC SDK but produces nothing. `ncu` (per-kernel) and `nsys`
(system timeline) are the only working NVIDIA profilers here.

## 7. ncu/ncu-script quirks worth pinning

- This ncu build's option parser **rejects a bare `--`** before the target —
  `profile-ncu-bristen.sh` omits it.
- `-k` needs the **`regex:` prefix** to substring-match the Triton kernel name.
- `--export` suppresses the stdout section text, so the script captures the run
  log then regenerates the human report via `--import`.
- For multi-kernel dispatches (`mhc_pre`, `moe_align_block_size`) `-c 1` samples
  the first matching kernel — a representative roofline, not the whole op.

## 8. Queue reality on the contended `mi300` partition

beverin `mi300` is usually saturated (this run: 112 nodes `alloc`, ~5 `idle`).
Benchmarks land fast (seconds to allocate); **rocprof roof profiles queue and
drain slowly** (~15–25 min each, multiple rocprof passes). Submit the 10 profile
jobs as independent sbatches so they parallelize across the ~5 free nodes
instead of serializing. bristen `normal` had 25 idle nodes — ncu jobs run
concurrently with near-zero queue wait.

## 9. The bench reproduces the README within run-to-run noise

Fresh beverin `bench_all` reproduced the checked-in README "Performance" table
to within ~3% on every row (e.g. `moe_int4_w4a16` 23.44× vs README 23.2×;
`dual_rmsnorm` 4.40× vs 4.2×; `moe_align_block_size` 33.73× vs 33.8×). The
`mhc_prenorm_gemm` row is the noisiest (0.013 ms opt → 123–205× swing) because
it is launch-overhead-dominated at T=8; treat its speedup as "≫100×", not a
precise figure.

## 10. `bench_all_beverin.sbatch` had a stale default `REPO`

The sbatch's `REPO` default was `/capstor/scratch/cscs/xyao/kernels` (missing the
`x`); the driver `scripts/cluster.sh submit --host beverin` overrides it to the correct
`.../xkernels`, so the documented path works — but submitting the sbatch
directly without `REPO=` would point at a stale/missing tree. Always submit via
the driver, or pass `REPO=/capstor/scratch/cscs/xyao/xkernels`.

---

The entries below are from a **later** pass (issue #50, MoE device-side
routing) — distinct campaign, same page, because this file's purpose is "facts
that cost real debugging time," not "one campaign's log."

## 11. The autotune wrapper corrupts certain dispatches under concurrency (the GPU debugging trichotomy)

**Provenance.** Issue #50 (MoE sync-free device-side routing), GPU-validation
pass on bristen A100 (sm_80), 2026-06-26. Produced the
[`diagnose-wrong-results`](../.agents/skills/diagnose-wrong-results/SKILL.md)
skill — the "kernel crashes / fails verify on GPU" peer of the perf-diagnose
skills (which all assume `verify().correctness.passed == true`).

**Symptom.** The fused INT4 MoE GEMM raises *illegal-memory-access* on the A100
when launched through its `@triton.autotune` wrapper under the new ghost-expert
EP routing dispatch — but ONLY when run after smaller decode buckets in the same
process (in isolation it is fine). The MXFP4 GEMM, with the *identical* routing,
is A100-clean at every scale.

**The three-way signature that fingerprinted it** (this is the reusable part):

| condition | result | implication |
|---|---|---|
| `CUDA_LAUNCH_BLOCKING=1` | **PASSES** | not a deterministic OOB in the hot loop |
| `compute-sanitizer` | **0 errors, PASSES** | not a memory-safety violation the sanitizer can reach |
| bypass autotune (resolved-config direct call) | **ALL buckets PASS** | the kernel is correct; the **autotune wrapper** is the corrupter |

No single tool shows this — only the *combination*. (`compute-sanitizer`
serializes enough that the trial-time corruption vanishes under it; blocking
removes the cross-launch overlap the wrapper needs.) The PASSES/clean/PASSES
signature is the fingerprint of autotune-wrapper corruption.

**Cause.** Triton's `@triton.autotune` *trials* configs whose `BLOCK_SIZE_M`
exceeds the dispatch, and the INT4 GEMM's token-id gather
`tl.load(sorted_token_ids_ptr + offs_token_id)` was **unmasked** — so during a
trial with a large `BLOCK_SIZE_M` the last block read past `EM`. The cached
winner was fine, but the trial corrupted the output buffer / crashed under
concurrency. The interpreter never saw it (it bounds-checks every load and runs
no trials).

**The fix, in two parts.**
1. **Mask every gather by the true extent** (defensive, always — the canonical
   lesson): `tl.load(..., mask=offs_token_id < EM, other=num_valid_tokens)`. The
   `other=` value is the pad id, which the existing `token_mask` already drops,
   so this is strictly result-preserving. Applied to BOTH the INT4 and MXFP4
   GEMMs (`src/xkernels/ops/moe/triton/{moe_int4,moe_mxfp4}_kernel.py`). An
   unmasked gather that "works" is a latent bug — it passes until a config trial
   or a new shape reads past the end.
2. **Pin/bypass the wrapper for the still-corrupting dispatch.** The mask alone
   did not clear the INT4 EP path (the wrapper's interaction is deeper than the
   one gather); INT4 EP keeps the reference align with a launcher comment
   documenting why, and the device-side Triton routing is A100-verified for INT4
   non-EP and both MXFP4 paths. The wrapper bug itself is GPU-gated follow-up.

**The methodological lesson (encoded as a skill).** This cost hours because the
repo had a `profile → diagnose → fix` pipeline that assumes the kernel already
passes correctness. The thing that actually happens first on a GPU — a crash or
wrong result — had no playbook. The ladder that cracked it:
1. **Reproduce standalone**, not in pytest (the harness's autotune-pinning +
   parametrize-ordering invent *and* hide bugs — see #12).
2. **Isolation ladder**: in-isolation → in-sequence → on-main → on-stash. (It
   also proved a *second* apparent failure — M=128 EP numerical drift — was a
   harness config-pin artifact, `BLOCK_SIZE_M=16` vs `align_block_m(128)=64`,
   not a kernel bug. The kernel was correct.)
3. **The trichotomy** above. Read the three-way signature.

If a kernel crashes or fails `verify` on GPU (rather than merely being slow),
route to [`diagnose-wrong-results`](../.agents/skills/diagnose-wrong-results/SKILL.md)
BEFORE touching a profiler.

## 12. Autotune config-pinning in tests is shape-coupled (`align_block_m`)

**Symptom.** A `test_ep_partials_sum_to_full[M=128-...]` reported ~94% numerical
mismatch (max abs err ~2.3) on the A100 — but the *same* M=128 EP dispatch was
correct in a standalone script (err 0.0080), and main passed the pytest case too.

**Cause.** The test calls `_pin_single_config()`, which forces a single autotune
config with `BLOCK_SIZE_M=16`. That pin is valid only for shapes where the
launcher's `align_block_m(M)` equals 16 — i.e. `M <= 32`. At `M=128`,
`align_block_m` returns 64, so the pinned 16-wide dispatch misroutes against a
64-wide block sort → wrong token/expert mapping → numerical garbage. The
standalone script (no pin) and main (no M=128 bucket) were both correct.

**Fix.** When extending a parametrize list on a test that pins a config, check
`align_block_m(M)` against the pin FIRST — it is a one-line CPU check. The M=128
prefill bucket was dropped from the INT4 EP test; decode buckets (M<=16,
`align_block_m=16`) are pin-compatible. The unpinned prefill path is validated
separately in `meta/benchmarks/bench_moe_e2e_routing.py`.
