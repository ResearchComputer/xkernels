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

## Result: TileLang split-KV is categorically better

Head-to-head on MI300A (same shape; TileLang's AMD-tuned `deepseek_mla` reference
flash-MLA with split-KV, both validated against the same torch reference,
rel err ~1-2e-3):

| top-k | n_split | TileLang | Triton | **TileLang speedup** |
|---:|---:|---:|---:|---:|
| 512 | 2 | 0.187 ms | 0.329 ms | **1.76×** |
| 1024 | 4 | 0.198 ms | 0.644 ms | **3.26×** |
| 2048 | 8 | 0.205 ms | 1.276 ms | **6.22×** |

TileLang stays ~flat as top-k grows (split-KV + MFMA tiling parallelize the
reduction across the GPU); the win grows with top-k. This is a structurally
better kernel for the problem — worth productionizing as an optional backend.

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

**Known limitation — end-to-end perf (the next thing to fix):** dispatched
head-to-head on MI300A, the backend is currently *slower* than Triton
(0.27× / 0.48× / 0.93× at top-k 512 / 1024 / 2048) **despite the kernel being
1.76–6.22× faster** (table above). The wrapper does the gather + the 448→512
padding `cat` + casts in torch on every call (~constant ~0.4 ms), which dominates
and negates the kernel win. The Triton kernel gathers *inside* the kernel, so it
has no such overhead.

## Next steps (to realize the win)

1. **Fuse the gather into the TileLang kernel** (index-load KV like the Triton
   kernel does) and drop the torch pre-gather + padding `cat` — this is what
   makes the kernel speedup show up end-to-end. (Alternatively, handle dim=448
   natively so no padding `cat` is needed.)
2. Add the **per-token length mask** for padded/variable top-k → then promote
   `TILELANG` into the AMD "auto" order.
3. fp8_ds_mla fused gather + the full `flash_mla_with_kvcache` decode path.
4. Bake the from-source TileLang ROCm build into the tokenspeed serving image
   (enroot overlay) so the backend is present at serve time.
