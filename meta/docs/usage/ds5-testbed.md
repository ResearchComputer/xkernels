# ds5 — CUTE DSL test bed (Grace+Blackwell GB10)

`ds5` (`ssh ds5`, dgx-spark-05.inf.ethz.ch) is an aarch64 **Grace+Blackwell DGX
Spark** with a single **NVIDIA GB10 GPU = compute capability 12.1 (`sm_121`)**,
CUDA 13.0 toolkit at `/usr/local/cuda-13.0`. It is the NVIDIA test bed for
**CUTE-DSL** (`cutlass.cute`) cards — a *portability layer for one vendor*
(NVIDIA), **not** a cross-vendor backend. It slots in as `backend: "cuda"` Impl
Cards under the existing Op Specs; the Triton + reference backends stay the
portable path.

This is the **environment + reproduce runbook**. The *authoring knowledge* for
writing a CUTE DSL kernel (the three-function structure, the `cute.compile`
compile-cache, the math/reduction primitives, the bf16-native-read perf lever, the
negative results) lives in [`meta/wiki/05-cutedsl-authoring.md`](../../wiki/05-cutedsl-authoring.md)
— read it before authoring a card.

## Hardware facts (load-bearing)

- **arch = `sm_121`** (`torch.cuda.get_device_capability(0) == (12, 1)`). The
  xkernels `arch.family` enum admits `nvidia_sm121` (added in this work); the
  CUTE DSL targets sm_121 automatically via the device-default arch.
- Driver-only install by default, but the **CUDA 13.0 toolkit + nvcc are present**
  (CUTE DSL JIT needs nvcc). No conda, no system torch.
- Unified memory (Grace+Blackwell); 3.4 TB free disk on `/local/home/xiayao`.
- **The matrix engine is gated on this stack** (be honest about it): on sm_121 /
  CTK-13.0, `MmaFP8Op` (fp8 m16n8k32→fp32) needs CTK ≥ 13.1; `MmaSM120BlockScaledOp`
  is MX microscaling (e8m0/e4m3), not DeepSeek's fp32 block=128 scales; bf16 MMA
  would fail the op's fp32 sweep point (rtol 1e-3). So the 5 cards ship the honest
  **fp32-FMA path** (the matrix-engine path is a documented follow-up gated on
  CTK 13.1). This means **all 5 cards are memory-bound** (AI ≤ 43 << 121).

## Package-name trap (cost real time — read this)

PyPI `cutlass` (0.5.0) is an **unrelated squatter** (`linear_model`/`metrics`/
`preprocessing`). `nvidia-cutlass` (4.2.0.0) only ships `pycute` (the pure-Python
CUTE *layout* algebra) + codegen helpers — **not** the DSL compiler. The real
CUTE DSL is **`nvidia-cutlass-dsl`** (installs the top-level `cutlass` module →
`cutlass.cute`), latest stable 4.5.2, with a `[cu13]` extra that matches the
CUDA-13 toolkit on ds5. The xkernels `cute` extra pins `nvidia-cutlass-dsl[cu13]`
(see `pyproject.toml`).

Always export `CUDA_HOME=/usr/local/cuda-13.0` (and put `$CUDA_HOME/bin` on PATH)
before running anything that JITs — the CUTE DSL invokes nvcc.

## rcc profile

`.rcc/config.toml` has a `ds5` profile (host `ds5`,
`remote_dir = /local/home/xiayao/xkernels`). Use it:
```bash
rcc --profile ds5 push                 # sync the project tree (ignores .venv/.git)
rcc --profile ds5 run -s '<snippet>'   # NOTE: use -s for shell snippets; plain `run --` execs argv literally
```

## Running verify / tests / benchmarks — rcc + docker (the canonical path)

`verify()`, `verify_parity()`, `verify(..., measure_perf=True)`, pytest, and the
`meta/benchmarks/*` scripts are **device calls** — they need the GB10. Run them
**inside the NGC container via rcc's `--docker` target** (this is the canonical
path for verify/test/bench; the `.venv` recipe further down is the fallback for
CUTE-DSL stand-up). The container config lives in `.rcc/config.toml`
(`[profiles.ds5.docker]`: image `nvcr.io/nvidia/pytorch:26.01-py3`, mounts the
remote tree at `/workspace`, sets `PYTHONPATH=/workspace/src` + a persistent
`TRITON_CACHE_DIR`, `--privileged` for GB10 CUDA init):

```bash
rcc --profile ds5 push                      # sync the tree → /local/home/xiayao/xkernels
rcc --profile ds5 run --docker -s 'python - <<PY
from xkernels import verify, verify_parity
r = verify("dual_rmsnorm.triton@1.0.0", arch="nvidia_sm121", measure_perf=True)
print("passed=", r["correctness"]["passed"], "max_rel=", r["correctness"]["max_rel_err"], "ms=", r["perf"]["ms"])
print("parity agree=", verify_parity("dual_rmsnorm@1.0.0", archs=["nvidia_sm121"])["agree"])
PY'
```

- **`--docker`** runs inside the profile's container; **`-s`** passes a shell
  snippet (heredocs / pipes / quotes ok) — plain `--` execs a literal argv. The
  container auto-sets `PYTHONPATH=/workspace/src`, so `import xkernels` resolves
  to the just-pushed tree with **no venv activate** needed.
- **arch string is `nvidia_sm121`** (GB10). ds5 is **NVIDIA only** — the
  AMD/CDNA3 (gfx942) ceiling is a separate GPU-gated follow-up on **beverin**
  (`scripts/cluster.sh run --host beverin -- ...`). The portable Triton kernel is
  arch-agnostic, so a ds5 PASS is the same code that will run on gfx942.
- **Backend registration is by import side-effect.** `import xkernels` auto-wires
  the package ops (e.g. `dual_rmsnorm.triton`). **DSL-emitted ops not yet imported
  by `ops/<x>/__init__.py` need an explicit `register_dsl(spec_of(<vkl_body>),
  "triton")`** before `verify`, or it raises `KeyError: backend 'TRITON' not
  registered` (the standalone `rmsnorm` is the current example — copy
  `scripts/ds5_rmsnorm_gpu_gate.py`).
- **`perf.tflops` / `achieved_bw_pct` are `None`** from `verify` (only `ms` is
  measured). Compute them from the profiler (`use-nsight-compute`, bristen) and
  feed `record_measurement(...)`.
- **Tests / benchmarks** take the same path:
  `rcc --profile ds5 run --docker -s 'python -m pytest tests/test_<x>.py -q'`
  and `rcc --profile ds5 run --docker -s 'python -u meta/benchmarks/bench_all.py'`.
- **Detached / long runs:** add `--detach` (survives disconnects; manage with
  `rcc --profile ds5 bg ps|logs|attach|wait`).

## Stand-up (one-time, reproducible)

```bash
rcc --profile ds5 run -s 'curl -LsSf https://astral.sh/uv/install.sh | sh'   # uv -> ~/.local/bin (no sudo)
rcc --profile ds5 push
rcc --profile ds5 run -s '\
  export PATH="$HOME/.local/bin:$PATH"; \
  export CUDA_HOME=/usr/local/cuda-13.0; \
  cd /local/home/xiayao/xkernels && . .venv/bin/activate && \
  uv venv --python 3.12 .venv && \
  uv pip install torch && \
  uv pip install -e ".[dev,cute]" numpy'
```

Result on ds5 (2026-06-29): torch 2.12.1+cu130 (aarch64, drives the GB10),
nvidia-cutlass-dsl[cu13] 4.5.2, triton 3.7.1, numpy 2.5.0, xkernels editable
(registry loads: 10 specs / 10 cards / 15 kernels).

## The 5 verified cards

All pass `verify` + `verify_parity` on ds5. Copy the closest one for the next
card; the three-function structure + compile-cache + device guard are identical
across all of them.

| card | file | pattern it demonstrates | ms (bf16) |
|---|---|---|---:|
| `mm_fp8_blockscale.cuda@1.0.0` | `ops/gemm/cute/mm_fp8_blockscale_kernel.py` | GEMM (tiled, Kahan K-reduce, B-transpose coalescing, host dequant) | 0.080 |
| `dual_rmsnorm.cuda@1.0.0` | `ops/norm/cute/rmsnorm_kernel.py` | Block-wide reduction (warp_reduce + SMEM + sync + rsqrt) | 0.061 |
| `moe_sum_reduce.cuda@1.0.0` | `ops/moe/cute/sum_reduce_kernel.py` | Per-thread small reduction + bf16-native-read | 0.252 |
| `mha_merge_state.cuda@1.0.0` | `ops/attention/cute/merge_state_kernel.py` | Online-softmax (exp/log/max) + bf16-native-read | 0.042 |
| `hc_prenorm_gemm.cuda@1.0.0` | `ops/mhc/cute/prenorm_gemm_kernel.py` | Fused epilogue (GEMM + squared-sum share the K-axis) | 0.040 |

## The perf levers that landed here

These are summarized here because they shape the runbook; the full analysis is in
[`meta/wiki/05-cutedsl-authoring.md`](../../wiki/05-cutedsl-authoring.md).

- **The compile-cache (119×).** `@cute.jit.__call__` rebuilds the MLIR engine every
  call (~9 ms) — ncu shows the GPU dispatch is ~90 µs, so the GPU sits idle ~99%
  of the time. The fix: `cute.compile()` once into a handle cached per
  Constexpr tuple, launched **tensors-only** (re-passing the Constexpr segfaults —
  uncatchable SIGSEGV). End-to-end **9.47 ms → 0.0795 ms (119×)**. Duplicated
  verbatim in all 5 cards' launch functions.
- **bf16-native-read (2.0×).** Host-upcasting a bf16 input to fp32 both adds an
  upcast launch *and* doubles read traffic. The DSL promotes bf16→fp32 on load
  losslessly, so read bf16 natively and accumulate fp32. Applied to
  `mha_merge_state` (0.084→0.042 ms, 28%→56% peak BW) and `moe_sum_reduce`
  (134→108 µs dispatch). **Not** applied to `dual_rmsnorm`/`hc_prenorm_gemm`
  (launch-bound at sweep sizes) or the GEMM (its fp32 inputs come from host fp8→fp32 dequant).
- **The tolerance fix (unblocks all bf16/fp32 cards).** xkernels' `verify` used
  the non-standard AND criterion; torch/numpy use the combined `|a−e| ≤ atol +
  rtol·|e|`. The AND form false-failed any non-bit-identical backend at moderate
  magnitude (1 bf16-ULP at |out|=2 is 0.0156 > any sane atol). FIXED in `verify.py`
  (`_within_tolerance`); GEMM card re-verified 5/5, 88 local tests pass.

## Reachability findings

- The CUTE DSL **framework** acknowledges `sm_120` (22 refs) and `sm_121` (15
  refs) → GB10 is a targetable arch.
- **torch 2.12 already vendors CUTE DSL kernels** under
  `torch/_inductor/kernel/vendored_templates/cutedsl/`, including
  `dense_blockscaled_gemm_persistent.py` (= our `mm_fp8_blockscale`) and
  `cutedsl_grouped_gemm.py` (= MoE). These are authoritative, tested DSL kernels
  to model cards on — **but** the blockscaled one is hard-gated to
  `cc ∈ {100,101,103}` and does not cover `sm_121`. A CUTE card on ds5 needs a
  kernel authored for sm_121, not the vendored sm_100 kernel.

## Reproduce

```bash
rcc --profile ds5 push
rcc --profile ds5 run -s '\
  export PATH="$HOME/.local/bin:$PATH"; \
  export CUDA_HOME=/usr/local/cuda-13.0; export PATH=$CUDA_HOME/bin:$PATH; \
  cd /local/home/xiayao/xkernels && . .venv/bin/activate && \
  python -m xkernels.ops._cute_backend.smoke_vecadd'          # the hello-world self-check
# per-card verify + perf:
python scripts/archive/ds5-probes/ds5_verify_card.py mm_fp8_blockscale.cuda@1.0.0 mm_fp8_blockscale@1.0.0
python scripts/archive/ds5-probes/ds5_cute_perf.py
# the API probes (re-derive any convention that moved in a future DSL release):
python scripts/archive/ds5-probes/ds5_dsl_math_probe2.py        # math intrinsics
python scripts/archive/ds5-probes/ds5_dsl_rowsum_probe.py       # reduction primitives
python scripts/archive/ds5-probes/ds5_bf16_load_probe.py        # bf16-native-read
python scripts/archive/ds5-probes/ds5_roofline_survey.py        # which regime each card is in
```

## See also

- [`meta/wiki/05-cutedsl-authoring.md`](../../wiki/05-cutedsl-authoring.md) — the
  CUTE DSL authoring reference (API surface, compile-cache, primitives, negative
  results). Read before authoring a card.
- [`clusters.md`](clusters.md) — the beverin/bristen multi-node cluster runbooks.
