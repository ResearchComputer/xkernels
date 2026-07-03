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

## 13. Backends register by import side-effect; DSL ops must be wired into `ops/<x>/__init__.py` (the `rmsnorm` lesson)

**Symptom.** A card with a real Triton kernel + measured `perf.measured` (e.g.
`rmsnorm.triton@1.0.0`, which had 5 GB10 entries recorded) raised
`KeyError: "backend 'TRITON' not registered for kernel 'rmsnorm'; have
['REFERENCE']"` from `verify("rmsnorm.triton@1.0.0")` — even though the same card
PASSED when the standalone `scripts/ds5_rmsnorm_gpu_gate.py` ran it. The op was
also unreachable as `xkernels.rmsnorm(...)` (no such export).

**Cause.** Dispatch backends register by **import side-effect**, not by scanning
the registry. `import xkernels` runs each `ops/<family>/__init__.py`, which is
where each backend module is imported under a `backend_registration_guard`. The
`rmsnorm` card landed via the DSL emit path (commit `fc3834e`) and its perf was
recorded by a one-shot script that called `register_dsl(spec_of(rmsnorm),
"triton")` itself — but **nobody added the `ops/norm` import wiring** that makes
`register_dsl` fire on a plain `import xkernels`. So the card was real and
verified-in-a-script yet invisible to the package surface (issue #66 was closed
against an unreachable op). The `silu_and_mul`/`gelu_and_mul` ops (#67) did NOT
hit this because their landing commit (`de1f6bc`) wired `ops/activation` end to
end — that is the template.

**Fix (and the reusable rule).** A DSL-emitted op is not done at "card emitted";
it is done when **three** wiring pieces exist, mirroring `ops/activation`:
1. `ops/<family>/triton/<op>_kernel.py` — a one-liner module that calls
   `register_dsl(spec_of(<body>), backend="triton")` (the generated kernel is
   lazy, so this is GPU- and triton-import-safe at import time);
2. that module imported under a `backend_registration_guard(..., TRITON, ...)` in
   `ops/<family>/__init__.py` (DSL backends skip the `triton_import_ctx` the
   hand-written kernels need);
3. a `dispatch(...)` interface function in `ops/<family>/interface.py` + the name
   re-exported from `xkernels/__init__.py`.
The closure check is a one-liner on ds5 — `import xkernels; verify("<op>.triton@
<ver>", arch="nvidia_sm121")` must PASS **with no manual `register_dsl`** (see
`meta/docs/usage/ds5-testbed.md` for the rcc+docker recipe). The same gap
currently lurks for `apply_rope` (#68) and the fp8 quant helpers (#57) — their
DSL cards are emitted but `measured=[]` and unwired, which is why those issues
stayed open.

**Fix shipped (2026-07-02).** `ops/norm/triton/rmsnorm_kernel.py` +
`ops/norm/interface.rmsnorm` + the `__init__` exports; `verify` passes through
plain `import` (compiled=True, max_rel 3.1e-7), `xkernels.rmsnorm(x, w)` is
bit-exact vs the reference, parity agrees, 134 tests pass. The #57 fp8 quant
helpers (`per_token_group_quant_fp8` / `per_block_quant_fp8`) wired the same
way (`ops/gemm/triton/quant_kernel.py`): both triton cards PASS `verify` +
`verify_parity` on GB10 (bit-exact, max_rel 0.0), and the grouped-view triton
dispatch is bit-exact with the `[M,K]` reference. **#68 `apply_rope` also wired
(`ops/attention/triton/rope_kernel.py`) — but needed §14's codegen fix first**
(its device kernel had a modulo-sign OOB).

## 14. RESOLVED — `apply_rope`'s generated device kernel OOB was a modulo-sign bug (#68, fixed)

> This entry was originally written as an unfixed blocker (the prior session
> diagnosed a "multi-dim address decomposition codegen bug" and shipped a
> reference-only interim). That diagnosis was **wrong**: the codegen's address
> math is in-bounds; the real bug was a Python-vs-C **modulo sign** mismatch.
> Rewritten as a resolved gotcha because the lesson generalizes to every future
> DSL data-addressing op.

**Symptom.** `verify("apply_rope.triton@1.0.0", arch="nvidia_sm121")` on ds5
(GB10) raised `RuntimeError: Triton Error [CUDA]: an illegal memory access was
encountered` (`compiled=False`). `compute-sanitizer --tool memcheck` reported a
stream of `Access to 0x... is out of bounds` (true OOB — trichotomy `FAILS/
blocking + sanitizer ERROR`). The CPU oracle and the **reference** card were
bit-exact; only the **generated triton device kernel** crashed.

**Real cause (not what the first diagnosis said).** The first diagnosis guessed
an off-by-one in the multi-axis offset decomposition; it was **wrong**, and
chasing it wasted a session. The generated kernel's offsets are all in-bounds by
construction — every per-axis index is `% shape`, every load is masked. The bug
is a **modulo-sign mismatch**: `_TritonGenMultiDim._offset` emitted `(coord) %
shape` assuming **Python** semantics (floored: `-1 % 64 == 63`), but **CUDA /
Triton `%` follows C sign** (truncated: `-1 % 64 == -1`). Where does a negative
coord come from? The **`Concat` b-branch** shifts its output coord by
`-len_a` (`b_coord = c{ax} - len_a`), so for output lanes `c{ax} < len_a` the
b-coord is *negative*. That negative coord then feeds a downstream load's
per-axis index (e.g. `apply_rope`'s `g13 = cache[pos*D + (c2-32) % 64]`); a
C-sign `%` keeps it negative → the offset is `pos*D - k` → for `pos=0` it reads
*before* the buffer → illegal memory access. The masked-`where` discards those
lanes' *values* but the *load already executed* OOB.

**Fix.** Add a floored-modulo helper `_floored_mod(expr, n)` =
`((expr) % n + n) % n` (non-negative for all inputs) and use it at the three
emit sites that compute a load's per-axis broadcast index (`_offset` + the
Gather leading/trailing axes). The output coord decomposition (`c0/c1/c2` from
`offs >= 0`) is left on plain `%` (its dividend is always non-negative). For
non-negative coords `((x%n)+n)%n == x%n`, so the fix is **result-preserving**
for every existing (non-`Concat-b`) load — it only changes the (discarded,
value-irrelevant) b-branch lanes, wrapping them in-bounds. This is exactly the
"mask every gather by its true extent; result-preserving" rule the
`diagnose-wrong-results` skill states, generalized to the whole broadcast
family. One-line-after-the-helper change; bit-exact-within-bf16 verified.

**Verification (ds5/GB10, after the fix).** `verify("apply_rope.triton@1.0.0")`
compiled=True, passed=True (5/5, max_rel 7.5e-3 < bf16 rtol 1e-2);
`verify("apply_rope.reference@1.0.0")` bit-exact; `verify_parity(archs=[
"nvidia_sm121"])` agree=True; compute-sanitizer CLEAN. The triton backend is
wired (`ops/attention/triton/rope_kernel.py` + the §13 import); the 3
device-gate tests in `test_vkl_rope.py` RUN and PASS (the skip markers are now
no-ops).

**The reusable lesson (why this is a gotcha, not just a fix).**
1. **Python `%` ≠ CUDA `%` for negative dividends.** Any DSL lowering that
   emits `coord % shape` for a broadcast index MUST floor it — a `Concat`
   b-branch (and any future op that subtracts from a coord) makes the dividend
   negative. Assume Python semantics only when the dividend is provably `>= 0`.
2. **An OOB read is committed even if `tl.where` discards the value.** The
   `where` selects between two *already-loaded* values; both loads run. So the
   b-branch's address must be in-bounds even though its result is thrown away —
   hence floor the modulo rather than relying on the `where`.
3. **"Sanitizer reports OOB" names the *symptom*, not the *root cause*.** The
   first session read the sanitizer's "out of bounds" as "the multi-axis
   decomposition is wrong" and spent a session on a wrong fix. The standalone
   repro + offset-algebra (read the *generated source*) pinpointed the negative
   modulo in 15 minutes. **Always dump + read the generated kernel source
   before theorizing about codegen** — the `diagnose-wrong-results` skill's
   step-1 standalone-repro includes exactly this.
4. **`verify_parity` defaults to CPU** (`device` derived from `archs`); pass
   `archs=["nvidia_sm121"]` (or `device="cuda"`) or the TRITON backend is
   "not runnable" and parity is inconclusive (`agree=None`), not a pass.
5. **`verify_parity` uses the COMBINED criterion** `|a-b| <= atol + rtol*|b|`
   (the op's per-dtype `atol` + `cross_backend_rtol`), NOT pure rel-only.
   Rel-only (`|a-b|/|b|` with `|b|` clamped at only `1e-8`) is
   **mathematically the wrong metric for outputs that span orders of
   magnitude** — attention/softmax produce legitimate near-zero elements whose
   relative error explodes even when both backends agree to machine precision.
   Proven case (`paged_attention_prefill` #71): a fp32 reference-vs-triton pair
   with max ABS gap `8.3e-7` (= fp32 machine epsilon) reported max REL `2.75`
   from a single output element of magnitude ~`3e-7` that is genuinely ~0 in
   BOTH backends. The combined criterion's `atol` floor (0.1 for attention)
   absorbs that near-zero regime exactly, so parity passes — consistent with
   single-card `verify`, which already uses the combined form (`_within_tolerance`).
   Exact ops (`atol=0`) reduce to rel-only, so their behavior is unchanged. This
   is provably a superset of rel-only (no op can go PASS→FAIL; it only converts
   near-zero false-FAILs to honest PASSes). See `_within_tolerance`'s docstring,
   which flags the rel-only parity inconsistency this resolves.

## 15. Workspaces enable CUDA graph capture (the real #52 win, not allocation savings)

**The eager-mode allocation savings from preallocated workspaces are MARGINAL
(0–6 µs).** torch's caching allocator makes `torch.empty` nearly free on size
reuse, so "avoid allocation" is NOT the win. The load-bearing value of the
`*Workspace` dataclasses (`xkernels.ops.attention.workspace`) is enabling **CUDA
/HIP graph capture**: a graph requires the SAME memory addresses across captures,
which per-call allocation makes impossible. With a workspace, the whole decode
step captures once and replays as one graph launch.

Measured on ds5 (GB10), `paged_attention` decode (Qwen3-4B GQA), graph-replay
vs eager:
```
   B  seq   eager_ms  graph_ms   speedup
   1  128    0.0175    0.0062    2.83x   <- launch-overhead-dominated (the win)
   1  512    0.0370    0.0369    1.00x   <- kernel-bound (no win)
  16  512    0.0920    0.0932    0.99x   <- kernel-bound
```
The speedup is exactly where issue #52 hypothesized ("allocation overhead
dominates SMALL-BATCH latency"): single/few-request decode, where Python
dispatch + kernel-launch overhead is a large fraction of total time. In a real
serving stack the win COMPOUNDS — every layer's attention + rope + the rest all
capture into one graph replay, collapsing the whole per-token Python overhead.

**Recipe** (the `workspace.py` module docstring has the full version):
```python
ws = PagedAttentionWorkspace.allocate(B_max, H_q, D, device="cuda", dtype=bf16)
for _ in range(3): paged_attention(q, ..., workspace=ws)   # warmup
with torch.cuda.graph(g):
    out = paged_attention(q, ..., workspace=ws)            # capture (stable addrs)
# replay each decode step: mutate inputs in place, g.replay()
```

**Stale-data safety**: these workspaces are safe to reuse for SMALLER buckets
(`B < B_max`) because the kernels FULLY overwrite every output element every
call (flash softmax always produces a value). The caller reads the valid `[:B]`
slice. Do NOT extend this pattern to outputs that need SELECTIVE zeroing (MoE
combine outputs with atomic-add into skipped-expert slots) without a per-call
zero — those are a separate follow-up.

## 16. `@triton.autotune` + atomic-add combine = silently wrong (the int4 fused_combine finding)

**Root cause found while wiring the MoE workspaces (#52):**
`fused_moe_int4_w4a16(fused_combine=True)` is **silently wrong** whenever the
GEMM launches through `@triton.autotune` (i.e. when `get_moe_int4_config(...)`
returns `None` — no tuned config for the shape). Verified on GB10 for M=8:
the fused-combine output was **312** vs the correct reference **0.65** (~480×
too big), while `fused_combine=False` (the scratch path) matched the reference
at 0.002.

**Why:** the fused-combine path atomic-accumulates the top-k weighted expert
results into a single `[M, N]` fp32 output buffer. `@triton.autotune`'s
benchmarking runs EVERY candidate config into that SAME buffer — each run's
atomic-adds accumulate, so after autotune the buffer holds roughly
`N_configs ×` the correct value. The scratch path is unaffected because it
WRITES (each token-slot to a unique row), not atomic-adds — the last autotune
trial's write simply wins, and it's correct.

**The mxfp4 kernel avoids this** by always calling `get_default_config(M)`
(always returns a config) and launching the resolved-config path (single run,
no autotune). The int4 kernel does not — `get_moe_int4_config` returns `None`
for untuned shapes and falls through to the autotune entry point.

**This breaks the ALLOC path equally** (it is not a workspace bug). The workspace
falls back to allocate-each-call under `config is None` (the SOUNDNESS GUARD in
`_moe_int4_wa16_triton`) so it never claims correctness for an unsound launch —
but the alloc path is still wrong. **Fix (follow-up):** make int4 resolve a
config before the combine launch (mirror mxfp4's `get_default_config`), or gate
`fused_combine=True` on `config is not None`. Until then, `fused_combine=True`
without a tuned config is broken on EVERY arch, not just GB10.
