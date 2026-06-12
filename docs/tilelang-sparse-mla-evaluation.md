# TileLang sparse-MLA backend — evaluation + productionization status

Evaluation of TileLang as a faster backend for the DeepSeek-V4 sparse-MLA
attention compute (issue #32) on AMD MI300A (gfx942), and the in-progress
productionization into xkernels as `Backend.TILELANG`.

## Why: the Triton kernel regresses at large top-k

The shipped Triton `sparse_mla_attention` kernel runs one program per
`(token, head)` and walks the top-k **serially**. Measured on MI300A
(`bench_sparse_mla.py`, T=8, H=128, D=512, d_v=448, vs naive torch
gather+softmax):

| top-k | Triton | naive torch | Triton speedup |
|---:|---:|---:|---:|
| 256 | 0.063 ms | 0.155 ms | 2.47× |
| 512 | 0.109 ms | 0.152 ms | 1.39× |
| **1024** | 0.210 ms | 0.154 ms | **0.73×** (slower!) |
| 2048 | 0.417 ms | 0.251 ms | 0.60× |

Triton time grows ~linearly with top-k and falls **below** naive torch at the
V4-Pro top-k=1024 — the serial reduction loses to torch's batched rocBLAS GEMM.

## Result (CORRECTED): TileLang is *not* faster at V4's top-k

An early head-to-head suggested TileLang was 1.76–6.22× faster. **That was a
measurement artifact** — it timed Triton on an unfavorable flattened per-token
`kv_flat[T·topk, D]` workspace, not the real shared-pool `sparse_mla_attention`
path. Measured consistently (same `sparse_mla_attention` op, shared pool
`kv[Kv,D]` + indices; T=8, H=128, D=512, d_v=448, Kv=8192; all validated vs the
torch oracle, out rel ~1.3e-2):

| top-k | naive torch | **Triton** | TileLang (full) | TileLang (kernel-only) |
|---:|---:|---:|---:|---:|
| 512 | 0.155 ms | **0.108 ms** | 0.389 ms | 0.270 ms |
| 1024 | 0.154 ms | **0.207 ms** | 0.398 ms | 0.273 ms |
| 2048 | 0.251 ms | 0.410 ms | **0.405 ms** | 0.272 ms |

(Triton here matches `bench_sparse_mla.py` exactly.) The TileLang kernel *is*
flat with top-k (~0.27 ms, split-KV working as intended), but:

- At V4's top-k (512 Flash, 1024 Pro) **Triton is faster** end-to-end, and even
  **naive torch** (batched gather + rocBLAS) beats TileLang. TileLang only reaches
  parity at top-k≈2048 and would win beyond that.
- TileLang's ~0.27 ms kernel floor (2 launches: split + combine) + ~0.12 ms
  wrapper overhead (sink rescale, lse, q-pad) is higher than Triton's single
  fused gather+attention kernel at these sizes.

**Conclusion:** the TileLang backend is correct and a good structure for
*very large* top-k, but it does **not** improve DeepSeek-V4 serving at the actual
top-k (512/1024). Keep Triton as the default; ship TileLang as an opt-in backend
only, and do not promote it into "auto". The honest win here is the rigorous
measurement, not a speedup.

## Building TileLang on ROCm (the gating prerequisite)

TileLang has **no ROCm pip wheel** (the PyPI wheel is CUDA-only and pulls a CUDA
torch). It must be built from source against ROCm. Confirmed working on beverin
(gfx942 / ROCm 7.2): tilelang 0.1.11, HIP GEMM with MFMA ran correctly. The full
recipe (LLVM 18 + a real `libtinfo.so.5`, login-node clone, `pip --user
--no-deps` with cython, `USE_ROCM=ON`) is in `slurm`-adjacent build scripts on
scratch and summarized below; bake it into the serving image via an enroot
overlay (`enroot create → start --rw + build → export`).

Build essentials:
- `CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=<llvm18>/bin/llvm-config"`
- `pip install -e . --no-build-isolation --no-deps` (avoids dragging CUDA wheels)
- build deps incl. **cython** (missing it is the classic `cython_wrapper.pyx` fail)
- compile target `"hip"`.

## Productionization status (this branch)

**Done + validated locally:**
- `Backend.TILELANG` added to the dispatch enum; **opt-in** (explicit
  `backend="tilelang"`), deliberately *not* in the AMD "auto" order yet.
- Guarded registration import in `ops/attention/__init__.py` (no-op where the
  from-source TileLang build is absent — i.e. everywhere except a gfx942 serving
  image). Full local test suite stays green.

**Done + validated on-device (`ops/attention/tilelang/sparse_mla_kernel.py`):**
- A split-KV flash-MLA backend matching the `sparse_mla_attention` signature
  (pre-gather via indices, nope/rope split, attention sink, lse). The kernel is
  the *unmodified* proven AMD split+combine; the sink is applied wrapper-side as
  an exact per-`(token,head)` rescale (`out *= sigmoid(lnZ_real - sink)`), and lse
  is derived from the per-split `glse` — keeping the kernel un-customized avoids
  the LayoutInference failure the in-kernel sink fold caused.
- **gfx942 validation (MI300A, job 383135):** `backends=[…,'TILELANG']`; out
  rel **1.3e-2** (no-sink and with-sink), lse |err| **1.8e-2**; the length-mask
  path correctly raises `NotImplementedError`. Numerically correct.
- **Two gotchas resolved:** (a) the value dim must be padded to a multiple of 128
  (V4's 448 → 512) — TileLang's FullRow-gemm fragment layout is invalid at 448
  but fine at 512 (zeros add nothing, sliced off); (b) `import tilelang` at module
  load hangs under `tokenspeed_triton`, so tilelang is imported lazily (inside
  `_build`) and registration is gated by `find_spec` — off the `import xkernels`
  critical path.

- The KV gather is **fused into the kernel** (per-row index-load from the pool,
  zero-padding nope 448→512 on load) — no torch pre-gather or pad `cat`. This was
  done to try to realize the win; it kept the kernel flat (~0.27 ms) but did *not*
  beat Triton at V4 top-k (see the corrected table — the ~0.27 ms 2-launch kernel
  floor is simply higher than Triton's single fused kernel there).

**Bottom line:** a working, numerically-correct TileLang sparse-MLA backend
exists and is opt-in, but it is **not** a perf win for DeepSeek-V4 at top-k
512/1024. It is kept as an optional backend (large-top-k regime) — not promoted
to "auto", not a blocker for V4 serving, and Triton remains the recommendation.

## If revisited (would-be next steps)

Only worth pursuing if V4's selected top-k grows well past ~2048, or under
HIP-graph capture where per-call Python/launch overhead disappears (which could
shift the crossover — untested):

1. Reduce the kernel floor: a single-pass (non-split) variant for small top-k, or
   fusing split+combine to one launch; tune block sizes for dim=448 natively
   (avoid the 512 pad).
2. Add the per-token length mask (TileLang varlen) for padded/variable top-k.
3. fp8_ds_mla fused gather + the full `flash_mla_with_kvcache` decode path.
4. Bake the from-source TileLang build into the serving image (enroot overlay).
