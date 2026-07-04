# vkl Phase D — native HIP codegen surface + the H1/H2 named-edit count

Recorded from the 2026-07-04 work on **issue #75** (vkl Phase D: the
cross-vendor codegen + edit-frequency track). Three PRs landed — #77
(schedule-spine round-trip tests), #78 (`src/xkernels/vkl/lower/hip.py` + the H1/H2
GEMM harness), #79 (the bandwidth-bound H1/H2 harness) — all GPU-verified on
**beverin (MI300A / gfx942, ROCm 7.2)** and **ds5 (GB10 / sm_121, CTK 13.1)**.

Two threads, each producing knowledge this page preserves:

1. **Native HIP codegen.** The first non-Triton lowering that reaches AMD
   silicon — `lower/hip.py` emits a real `load_inline` HIP kernel (not a
   `torch.matmul` wrapper). It ships at the **correctness-first wavefront-FMA
   bar** (mirroring `lower/cuda.py`'s FMA twin), NOT the MFMA ceiling. The MFMA
   ceiling is the documented follow-up, and §1 below is the reverse-engineered
   MFMA instruction/builtin surface that de-risks it — exactly the knowledge the
   `map-to-matrix-cores` skill flags as *"no native HIP source exists yet."*
2. **The H1/H2 count.** The empirical scope of the agent-native (named-edit)
   claim — across 3 ops × 2 archs. The methodology (§3) and the headline finding
   (§4) are the durable parts; the numbers themselves are in the cards / the
   issue-#75 comments.

> **TL;DR.** (1) A native HIP kernel via `load_inline` needs `__hip_bfloat16`,
> `at::cuda::getCurrentCUDAStream()` from `<ATen/cuda/CUDAContext.h>` (ROCm's
> torch build exposes the `at::cuda` stream API — **do not** spell `at::hip::`),
> triple-chevron launch, and `PYTORCH_ROCM_ARCH=gfx942` pinned around the
> compile. (2) The MFMA ceiling is **not** a hipify: only
> `__builtin_amdgcn_mfma_f32_32x32x4bf16` (K=4) exists as a clang builtin;
> `archdb.native_shape("amd_cdna3","mfma")={m:32,k:16}` matches **no** available
> instruction — the archdb/ISA mismatch is the first thing to reconcile. (3) H1
> (freehand native override) is needed **only for compute-bound ops** (GEMM: H2
> caps at 45–54% of BLAS because matrix-engine targeting lives in Triton's
> autotune config, not the declared `specialization_knobs`); for bandwidth-bound
> ops (rmsnorm, quant) H1 is categorically unnecessary.

## 1. The MFMA codegen surface on gfx942 (the expensive recon)

`map-to-matrix-cores` says *"no native HIP/CUDA kernel source exists yet"* — the
raw-`v_mfma` operand layout is the unproven frontier. This section is the
empirical map, produced on beverin (ROCm 7.2 / AMD clang 22) by
`__has_builtin` probes + deliberate-typo error-message mining + header grep.
**Do not write `lower/mfma.py` from memory** — read this first.

### 1.1 What clang actually exposes (the only bf16 MFMA builtin)

Enumerated via `__has_builtin` + call-and-read-the-error:

| candidate name | clang (gfx942) knows it? |
|---|---|
| `__builtin_amdgcn_mfma_f32_32x32x4bf16` | **✓ exists** — 6-arg: `a, b, c, cbsz, abid, blgp` |
| `__builtin_amdgcn_mfma_f32_32x16x16bf16` | ✗ undeclared |
| `__builtin_amdgcn_mfma_f32_16x16x16bf16` | ✗ undeclared |
| `__builtin_amdgcn_mfma_f32_4x4x4bf16` | ✗ undeclared (gfx9 dropped) |
| `__builtin_amdgcn_mfma_f32_32x32x8bf16` / `16x16x8bf16` | ✗ undeclared (f16 path, not bf16) |

So on this toolchain the **only** bf16 MFMA clang builtin is the **32×32×4**
(K-reduction = 4). Its 6-arg shape is `a, b, c, cbsz, abid, blgp` where the last
three are byte-selection / block-id / block-group modifiers that control the A/B
VGPR lane distribution — **the load-bearing hard part** (no in-repo template).

### 1.2 The archdb / ISA mismatch (resolve this FIRST)

```
archdb.native_shape("amd_cdna3", "mfma")  ==  {"m": 32, "k": 16}
archdb.instr_peak("amd_cdna3", "mfma")    ==  1300.0   # bf16
```

That `{m:32, k:16}` matches **no available bf16 MFMA instruction or builtin** on
gfx942 — the one exposed is 32×32×**4** (K=4), and the K=16 family isn't exposed
as a builtin at all. So an MFMA kernel effort starts by reconciling `archdb`
against the actual ISA: either the canonical shape should be `{m:32, k:4}` (the
32×32×4), or the K=16 path has to go via **inline asm** (next subsection), not
clang builtins. Do not write a `MapTo(instruction="mfma", instr_shape=(32,16))`
kernel against the current `archdb` and expect it to map to a real instruction —
it won't.

### 1.3 Three emit paths, in order of effort

1. **composable-kernels (CK) warp intrinsics** —
   `/opt/rocm/include/ck/utility/amd_xdlops.hpp` +
   `ck/tensor_operation/gpu/warp/xdlops_gemm.hpp` (`MfmaInstr`, `mfma_f32_32x32x4`)
   expose templated MFMA helpers that **hide the VGPR lane layout**. A `lower/hip.py`
   override could emit a CK-backed GEMM. **Caveat:** this is "call AMD's library,"
   which weakens the doc-09 codegen thesis (the point is the vkl lowering *emits*
   the MFMA instruction from the math IR, not that it delegates to a vendor lib).
   Include-path + compile-time plumbing into `load_inline` is non-trivial.
2. **`__builtin_amdgcn_mfma_f32_32x32x4bf16`** — the faithful codegen route, but
   you must nail the A/B VGPR lane distribution (`cbsz`/`abid`/`blgp`) yourself.
   No in-repo template exists; the ISA manual + an empirical
   load-known-values-and-read-back probe is the way.
3. **Inline asm `v_mfma_f32_32x32x4_bf16`** — the integrated assembler
   **rejects** the naive register-range spelling:
   ```
   v_mfma_f32_32x32x4_bf16 v[0:15], v[0:3], v[0:3], v[0:15] cbsz:1 abid:0 blgp:0
                                        ^ error: unexpected token in argument list
   ```
   Needs the correct SDWA-style operand spelling (modifier order / register-range
   syntax for the dst + src operands). Same layout problem as (2) once it parses.

### 1.4 Recon recipe (so the next agent doesn't redo it blind)

```bash
# beverin, in the tokenspeed-rocm-aiter-myofi env
# 1. which builtins exist?  __has_builtin is the source of truth (NOT header grep —
#    MFMA builtins are clang compiler builtins, declared in no .h):
for name in __builtin_amdgcn_mfma_f32_32x32x4bf16 \
            __builtin_amdgcn_mfma_f32_32x16x16bf16 ; do
  hipcc --offload-arch=gfx942 -c -x hip <(echo "#include <hip/hip_runtime.h>
  __global__ void k(){ constexpr bool b = __has_builtin($name); }") -o /tmp/p.o \
    2>&1 | grep -i error
done
# 2. the CK library path (grep, not memory):
grep -rnoE "struct [A-Za-z]*MfmaInstr|mfma_f32_32x32x4" /opt/rocm/include/ck
# 3. read the EXACT builtin arg types from a deliberate-type-mismatch error
```
The deliberate-error-mining trick (call the builtin with wrong-typed args, read
the diagnostic's "expected ..." note) is how the 6-arg signature was pinned.

## 2. The HIP `load_inline` spellings (the working `lower/hip.py` incantations)

The twin of page 05's CUTE-DSL authoring map, for the AMD side. Every spelling
below was found by a `load_inline` recon probe on beverin (the committed
`lower/hip.py` is the working template — copy it). torch on ROCm is
2.11.0+rocm7.2.

### 2.1 The canonical HIP C++ source block

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>                       // __hip_bfloat16
#include <ATen/cuda/CUDAContext.h>              // at::cuda::getCurrentCUDAStream

__global__ void gemm_kernel(const __hip_bfloat16* A, const __hip_bfloat16* B,
                            float* C, int M, int N, int K) {
    // ... tiled wavefront-FMA, fp32 accumulate ...
}

void launch(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    auto stream = at::cuda::getCurrentCUDAStream();
    gemm_kernel<<<grid, block, 0, stream.stream()>>>(
        reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(b.data_ptr()),
        c.data_ptr<float>(), M, N, K);
}
```

### 2.2 The four traps (each cost a compile cycle)

| trap | wrong spelling | right spelling | why |
|---|---|---|---|
| bf16 type | `__nv_bfloat16` (CUDA spelling) | **`__hip_bfloat16`** | clang will suggest the rename in its diagnostic; `<hip/hip_bf16.h>` |
| stream API | `at::hip::getCurrentCUDAStream()` | **`at::cuda::getCurrentCUDAStream()`** | the ROCm torch build exposes the `at::cuda` surface via `<ATen/cuda/CUDAContext.h>`; `at::hip::` is **not** in the symbol surface |
| launch syntax | `hipLaunchKernelGGL(...)` | **triple-chevron `<<<>>>`** | the HIP macro works on ROCm; it's what the CUDA twin uses, so the same source string serves both |
| arch pin | (none — autodetect) | **`PYTORCH_ROCM_ARCH=gfx942`** | pin around the `load_inline` call or hipcc targets the wrong/default arch and the build silently degrades |

The `lower/hip.py` `_compile_override` wraps the compile with `PYTORCH_ROCM_ARCH`
in the env and feeds the C++ string to `torch.utils.cpp_extension.load_inline`
with `cuda_sources=[...]` (HIP reuses the `cuda_sources` slot — there is no
separate `hip_sources`). The override is then registered with
`register_dsl_hip(spec, override)` as backend `HIP`.

### 2.3 The maturity framing that ships honestly

`lower/hip.py` is **wavefront-FMA**, not MFMA — exactly mirroring `lower/cuda.py`
(which is also CUDA-core FMA, reaching the tensor-core ceiling for neither).
The card must say so: `arch.requires == []`, `uses_primitives == []` (NOT
`["matrix_cores","mfma"]`). See §6.1 for the drift gate that enforces this.

## 3. The H1/H2 named-edit-frequency methodology

The criterion-#3 deliverable was *"what fraction of perf-to-ceiling pushes are
achievable via H2 named edits vs requiring an H1 freehand override?"* The
non-obvious methodological point — the one that took a wrong turn to learn — is
that **the ceiling depends on the op's regime**:

| regime | op examples | the honest ceiling | why not the obvious choice |
|---|---|---|---|
| **compute-bound** | GEMM | **vendor BLAS TFLOP/s** (`torch.matmul` = cuBLAS / hipBLASLt) | the matrix engine is the limiter |
| **bandwidth-bound** | rmsnorm, per_block_quant_fp8 | **DRAM roofline GB/s** (`arch_peaks(arch)["dram_bw_gbs"]`) | NOT torch-eager — eager is *unfused*, so it's not a ceiling at all (it's a floor) |

The verdict threshold is **70% of the ceiling** for H2-ACHIEVABLE; below that,
the gap is either H2 tiling work (bandwidth-bound) or H1 territory
(compute-bound, see §4). `achieved_bw = bytes_moved / ms`, where `bytes_moved`
is a hand byte-model per op (e.g. rmsnorm ≈ 4·T·d bytes bf16: read x + write out
+ read fp32 w; the fp32 mean-square reduction is on-chip).

The two harnesses (`scripts/h1h2_count_gemm.py`, `scripts/h1h2_count_bw.py`) are
self-locating on either vendor (they sniff `torch.version.hip` → ARCH) and print
the three numbers + verdict. They are the templates for the next op.

## 4. The empirical H1/H2 finding (the deliverable)

3 ops × 2 archs, GPU-measured 2026-07-04:

| op (regime) | arch | H2 (Triton + declared knobs) | % of ceiling | H1 verdict |
|---|---|---:|---:|---|
| GEMM bf16 (compute) | amd_cdna3 | 173.0 TF | **54%** of BLAS (321.7 TF) | **H1 NEEDED** |
| GEMM bf16 (compute) | nvidia_sm121 | 39.5 TF | **45%** of BLAS (88.2 TF) | **H1 NEEDED** |
| rmsnorm (bw) | amd_cdna3 | 1606 GB/s | 30% of DRAM (5300) | H1 not needed* |
| rmsnorm (bw) | nvidia_sm121 | 230 GB/s | **95%** of DRAM (243) | H1 not needed |
| per_block_quant_fp8 (bw) | amd_cdna3 | 1408 GB/s | 27% of DRAM (5300) | H1 not needed* |
| per_block_quant_fp8 (bw) | nvidia_sm121 | 240 GB/s | **99%** of DRAM (243) | H1 not needed |

\* MI300A rmsnorm/quant sit at ~28% of its very wide 5300 GB/s HBM3 roofline —
but this is **H2-SHORT, not H1 territory**: the gap is H2 tiling/coalescing/
occupancy, and a native matrix-core override (H1) is irrelevant to a
bandwidth-bound op. See §5 for the saturation reality.

**The scoping claim (this IS the agent-native pitch, honestly bounded):** H1 —
the freehand native override body, the not-reliable regime — is needed **only
for compute-bound ops where matrix-engine targeting escapes the declared
`specialization_knobs`.** On Triton that escape is concrete: the matrix knobs
(`matrix_instr_nonkdim`/`kpack` on AMD; tensor-core shapes on NVIDIA) live
**inside** the `@triton.autotune` config space, not the entry signature, so they
are not closed-enum `specialization_knobs` an agent can `SetKnob`. For
bandwidth-bound ops H1 is categorically unnecessary — the win is coalesced
vectorized loads, which Triton already emits.

Closing the compute-bound gap needs **either** (a) extending the declared H2
knob space to surface the matrix-engine shapes (still H2, but a config-space
change), **or** (b) a native matrix-core override body (H1 — the MFMA follow-up
to `lower/hip.py`'s FMA kernel). That boundary is exactly what this count was
meant to surface.

## 5. MI300A HBM3 is genuinely hard to saturate for small kernels

The 27–30%-of-roofline numbers for rmsnorm/quant on MI300A are **honest data,
not a harness artifact.** MI300A's HBM3 is very wide (5300 GB/s peak); published
MI300 kernels hit 60–80% only for **large streaming GEMMs**. For small / irregular
/ reduction-bearing kernels (rmsnorm T=8192 d=4096 = 134 MB; quant G=32768 B=4096
= 403 MB) the achieved fraction is much lower — the launch + reduction overhead
dominates and the wave can't keep the memory pipe full. Contrast GB10 (243 GB/s
roofline, bandwidth-starved): the *same* kernels hit 95–99% — a narrow pipe is
easy to fill.

**Implication for diagnosis:** do NOT route a "MI300A bandwidth-bound op at 30%
of roofline" straight to `diagnose-memory-bound` as if it were a coalescing bug.
First check whether the shape is large enough that any kernel *could* saturate
(roofline math: `bytes / roofline_bw` vs the launch floor). For genuinely small
kernels the "gap" is the roofline model misfitting, not the kernel — and H1 won't
help (it's bandwidth-bound).

## 6. Gotchas this engagement bit

### 6.1 The drift gate (`test_vkl_artifacts`) enforces card honesty

`tests/test_vkl_artifacts.py::test_vkl_managed_artifacts_do_not_drift` regenerates
every VKL-managed card from `emit_override_card` and diffs against the checked-in
JSON. **Hand-editing a managed card breaks CI** — even (especially) when the edit
is "honest" (e.g. setting `requires: []` for an FMA override by hand). The fix is
always upstream: change what `emit_override_card` produces, then regenerate.

Concretely: `_ARCH_REQUIRES` in `src/xkernels/vkl/override.py` declares the
native features a `(backend, arch)` override body uses. It had speculative
`("hip","amd_cdna3"): ["matrix_cores","mfma"]` entries that forced the hip card
to **claim** MFMA while the kernel was FMA — actively misleading. The fix was to
**drop** the hip entries (an FMA override uses no matrix features → honestly
absent from the table, exactly like the live cuda/sm_121 FMA twin), regenerate,
and update the test. The rule: only **full matrix-engine** overrides belong in
`_ARCH_REQUIRES`; an entry is added back when a real `v_mfma`/`wgmma` body ships.

### 6.2 No sm_90 host; sm_121 stands in for the NVIDIA column

The cluster exposes only **beverin (gfx942), bristen (sm_80), ds5 (sm_121)** —
there is **no `sgs-gpu07`** (the sm_90 node) configured. Issue #75's body said
"NVIDIA node (bristen) for the comparison half" but bristen is sm_80, not sm_90;
the actual sm_90 node isn't reachable. **sm_121 (ds5/GB10) stands in for the
NVIDIA column** in the H1/H2 table — documented in both harnesses. If sm_90
access is wired up, the harnesses are parameterized on ARCH and rerun trivially.

### 6.3 rcc / ssh strips quotes — use self-locating scripts

`rcc run -s '...'` and `srun bash -c '...'` **strip shell quotes**, so an inline
`python -c "..."` with a quoted string literal dies with `SyntaxError:
unexpected character after line continuation character` (the `\"` escapes survive
the strip as literal backslashes). The robust pattern is a **self-locating bash
script** pushed to the remote and invoked by absolute path:
```bash
rcc push scripts/my_probe.sh
scripts/cluster.sh run --host beverin srun <env> bash /capstor/.../scripts/my_probe.sh
```
This also dodges the related trap that `cluster.sh run`'s auto-push may skip
untracked files — an explicit `rcc push <path>` first is the reliable path.

### 6.4 `verify._as_device` misfires "AMD-target-but-NVIDIA-device" on ROCm

`src/xkernels/verify.py`'s `_as_device` assumes `cuda_available` ⇒ NVIDIA, so on
a ROCm box (where `torch.cuda.is_available()` is also true) it warns
`arch 'amd_cdna3' is an AMD target but the available CUDA device is NVIDIA;
running anyway`. It's **cosmetic** (everything runs as torch either way; the arch
label reflects the *requested* target), but noisy. Fix is a one-line
`torch.version.hip` check in `_as_device` — left separate because it's a
`verify.py` change (AGENTS.md hard rule: no production kernel/verify change
without the full `verify`/`verify_parity` gate).

## Reproduce

```bash
# beverin (MI300A / gfx942)
scripts/cluster.sh run --host beverin \
  srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
  --gpus-per-node=1 --time=00:15:00 \
  bash -c 'cd $REPO && PYTHONPATH=src python3 scripts/h1h2_count_gemm.py'
scripts/cluster.sh run --host beverin \
  srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
  --gpus-per-node=1 --time=00:15:00 \
  bash -c 'cd $REPO && PYTHONPATH=src python3 scripts/h1h2_count_bw.py'
# ds5 (GB10 / sm_121)
rcc --profile ds5 run --docker -s \
  'cd /workspace && PYTHONPATH=src python scripts/h1h2_count_gemm.py'
rcc --profile ds5 run --docker -s \
  'cd /workspace && PYTHONPATH=src python scripts/h1h2_count_bw.py'
# the HIP override correctness gate
scripts/cluster.sh run --host beverin \
  srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
  --gpus-per-node=1 --time=00:15:00 \
  bash -c 'cd $REPO && PYTHONPATH=src python3 -m pytest -q tests/test_vkl_override_hip.py'
```

## Open follow-ups this page surfaces

- **Reconcile `archdb.native_shape("amd_cdna3","mfma")`** against the actual
  gfx942 ISA (§1.2) — the canonical `{m:32,k:16}` matches no available builtin.
- **Write `lower/mfma.py`** (the literal criterion-#1 ceiling): pick the raw-
  builtin path (§1.3.2, faithful codegen) vs CK (§1.3.1, library delegation),
  nail the `cbsz`/`abid`/`blgp` lane layout with a load-known-values probe, then
  re-run the GEMM H1/H2 row — it should jump from 54% toward the 1300 TF ceiling.
- **Fix the cosmetic `_as_device` ROCm misfire** (§6.4) — a `verify.py` change,
  so gated on the full `verify`/`verify_parity` sweep.
