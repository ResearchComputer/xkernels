# xkernels

Customized compute kernels across hardware vendors (NVIDIA, AMD, …) and kernel
types (FFN, MoE, comm, …), with a uniform PyTorch API, automatic backend
dispatch, and a correctness + benchmark harness.

## Install

```bash
pip install -e ".[dev]"          # pure-Python (reference + triton if present)
XKERNELS_FORCE_BUILD=1 pip install -e .   # also build CUDA/HIP extensions
```

Triton/CUDA backends are optional; the package runs on the pure-torch reference
path anywhere.

## Usage

```python
import torch
from xkernels import fused_ffn

y = fused_ffn(x, w_gate, w_up, w_down)            # backend="auto"
y = fused_ffn(x, w_gate, w_up, w_down, backend="triton")  # force a backend
```

```python
from xkernels import fused_moe_int4_w4a16  # INT4 W4A16 grouped fused-MoE GEMM

out = fused_moe_int4_w4a16(A, packed, scale, topk_ids, topk_w, group_size=32)
```

Override globally with `XKERNELS_BACKEND=reference|triton|cuda|hip`.

## Performance

Speedup of each kernel's optimized backend over the naive PyTorch a practitioner
would write without it. All numbers are median of Triton `do_bench`, bf16
unless noted. Reproduce with `python benchmarks/bench_all.py` (single GPU) or
`sbatch slurm/bench_all_beverin.sbatch`.

| Kernel | Shape | Naive PyTorch | Optimized | Speedup MI300A | Speedup MI250X |
|--------|-------|--------------:|----------:|---------------:|---------------:|
| `moe_int4_w4a16` | M=64, E=48, N=4096, K=7168, top_k=8 | 31.66 ms (dequant+matmul) | 1.36 ms | **23.2×** | **19.3×** |
| `moe_mxfp4` | E=256, hidden=4096, ispp=512, top_k=6, M=48 | 149.2 ms (torch loop) | 9.41 ms | **15.9×** | **14.0×** |
| `moe_align_block_size` | M=16384, top_k=8, E=48, block=16 | 55.47 ms (torch argsort+pad) | 1.64 ms | **33.8×** | **24.5×** |
| `moe_sum_reduce` | M=8192, top_k=8, H=7168 | 3.17 ms (torch reduce) | 0.40 ms | **8.0×** | **7.8×** |
| `mhc_prenorm_gemm` | T=8, K=16384, N=24, splits=16 | 2.65 ms (F.linear+sqsum) | 0.02 ms | **123.3×** | **1.0×** |
| `mhc_pre_post` | T=8, hc_mult=4, hidden=4096 | 2.77 ms (torch pre+post) | 0.08 ms | **35.5×** | **3.9×** |
| `sparse_mla` | T=8, H=128, D=512, topk=512 | 3.00 ms (torch gather+softmax) | 0.11 ms | **26.8×** | **12.8×** |
| `dual_rmsnorm` | T=8192, d=(1536,512) | 0.24 ms (2× sequential RMSNorm) | 0.06 ms | **4.2×** | **6.3×** |
| `mha_merge_state` | T=8192, H=128, D=128 | 2.45 ms (torch merge) | 0.79 ms | **3.1×** | **3.2×** |
| `fused_ffn` | M=4096, 4096→11008 (fp16) | 5.42 ms (unfused torch, fp16) | 5.36 ms | **1.0×** | **1.0×** |

Naive baselines: `moe_int4_w4a16` (tuned grouped GEMM, block-align excluded —
see its own row) vs per-expert dequant(int4→bf16)+matmul; `moe_sum_reduce` /
`mha_merge_state` vs their torch oracles; `dual_rmsnorm` vs two sequential
RMSNorm launches; `moe_align_block_size` vs the torch argsort + per-expert
padding reference; `fused_ffn` vs the unfused `reference` backend;
`sparse_mla` vs a torch gather+softmax reference; `fused_ffn` vs the unfused
`reference` backend.

Notes:

- **`moe_int4_w4a16` 23.2× / 19.3×** — the grouped INT4 W4A16 fused-MoE GEMM
  launches from a **checked-in tuned config** (issue #16) instead of runtime
  autotune. Winners are swept on MI300A for the Kimi-K2.6 per-rank shapes
  `w13` (N=4096, K=7168) and `w2` (N=7168, K=2048) across decode buckets
  M∈{1,2,4,8,16,32,…} and prefill, stored in
  `src/xkernels/ops/moe/triton/tuned_configs/` keyed by `(E,N,K,device,dtype)`;
  untuned shapes fall back to the autotuner. Per-shape `do_bench` on MI300A
  (bf16): gate_up M=1 → 0.34 ms, M=8 → 1.15 ms, M=16 → 1.36 ms; down M=1 →
  0.12 ms, M=16 → 0.67 ms. This row isolates the GEMM — the block-align it
  consumes is the `moe_align_block_size` row (not timed here). Re-tune with
  `sbatch slurm/tune_moe_int4_beverin.sbatch`.
- **`moe_mxfp4` 15.9× / 14.0×** — V4-Flash MXFP4 MoE (E=256, issue #43) vs a
  per-expert torch loop that dequantizes and matmuls. Speedup grows with batch
  size (≈1× at M=1, ≈16–19× at M≥256) because the kernel amortizes expert
  scheduling and weight unpack across tokens; see
  `docs/issue-43-mxfp4-moe-gemm.md`.
- **`moe_align_block_size` 33.8× / 24.5×** — the Triton perf backend
  (vLLM/SGLang-style 4-stage histogram + padded prefix-sum + scatter, issue #4)
  is validated bit-for-bit against the reference. The win is large because the
  reference pays a full `argsort` plus a 48-iteration per-expert Python padding
  loop with per-step host syncs; the kernel collapses that into 5 launches. The
  speedup holds across token counts (≈14× at M=16, rising to ≈33× at M=16384 on
  MI300A) — swept in `bench_moe_align_block_size.py`.
- **`mhc_prenorm_gemm` 123.3× / 1.0×** — the DeepSeek-V4 MHC hidden-compression
  prenorm GEMM (issue #36), a portable gfx942 replacement for NVIDIA-only
  `deep_gemm.tf32_hc_prenorm_gemm`. A memory-bound tall-skinny GEMM (K=16384,
  N=24) fused with the RMS-prenorm squared-sum, split along K for decode
  occupancy. The MI300A win is large partly because the kernel reads `A` once
  and fuses both outputs, and partly because the naive
  `F.linear(a.float(), fn.float())` fp32 baseline pays the same dense-GEMM stack
  cliff as `fused_ffn` below. On MI250X the baseline is already fast, so the
  speedup is ~1×. On-device parity rel 3.8e-04; see
  `docs/issue-36-mhc-prenorm-gemm.md`.
- **`mhc_pre_post` 35.5× / 3.9×** — DeepSeek-V4 MHC full prenorm/postnorm fusion
  (`mhc_pre` + `mhc_post`, issue #44) vs the torch oracle. The large MI300A
  speedup comes from fusing the RMS prenorm squared-sum, sigmoid gating,
  sinkhorn combination, and the post residual combine into a few Triton kernels.
  On MI250X the kernel is still ~4× faster than the reference; see
  `docs/issue-44-mhc-pre-post.md`.
- **`sparse_mla` 26.8× / 12.8×** — sparse MLA decode attention top-k gather +
  softmax/dequant (issue #32) vs a torch reference. The optimized path avoids
  materializing a large `repeat_interleave` multiplier and removes unnecessary
  `contiguous()` calls; see `docs/issue-32-sparse-mla-attention.md`.
- **`fused_ffn` ≈ 1.0×** — the Triton backend fuses only the SwiGLU *activation*;
  the three projection GEMMs dominate and are torch matmuls in both paths, so
  there is little left to win. Measured in fp16 because on this torch
  2.11+rocm7.2 build the **bf16** `torch.matmul` (NN layout) path misses
  MFMA/hipBLASLt and runs ~470× slower than fp16 (0.8 vs 358 TFLOP/s;
  see `benchmarks/probe_ffn.py` and `docs/issue-17-bf16-dense-gemm.md`). The
  production `F.linear` (NT layout) bf16 path is fast; the slowdown is specific
  to the NN benchmark shape.
- **`hierarchical_all_reduce`** (distributed) does *not* beat a flat all-reduce on
  the 2-node / 4-NIC-per-node MI300A stack — RCCL's flat collective is already
  topology-aware. Full analysis in `docs/issue-12-hierarchical-all-reduce.md`.

## Layout

- `src/xkernels/ops/<type>/` — kernels by type; each has `reference.py`,
  `interface.py`, and per-backend subdirs (`triton/`, `cuda/`).
- `src/xkernels/_dispatch.py` — backend registry + selection.
- `tests/`, `benchmarks/`, `examples/` — harness and demos.

See `docs/adding-a-kernel.md` to extend. Design: `docs/superpowers/specs/`.
