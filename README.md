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
would write without it, on one **AMD Instinct MI300A (gfx942)**, bf16, median of
Triton `do_bench`. Reproduce with `python benchmarks/bench_all.py` (single GPU)
or `sbatch slurm/bench_all_beverin.sbatch`.

| Kernel | Shape | Naive PyTorch | Optimized | Speedup |
|--------|-------|--------------:|----------:|--------:|
| `moe_int4_w4a16` | M=64, E=48, N=4096, K=7168, top_k=8 | 32.0 ms (dequant+matmul) | 1.37 ms | **23.3×** |
| `moe_sum_reduce` | M=8192, top_k=8, H=7168 | 3.21 ms (torch reduce) | 0.38 ms | **8.4×** |
| `moe_align_block_size` | M=16384, top_k=8, E=48, block=16 | 54.4 ms (torch argsort+pad) | 1.65 ms | **32.9×** |
| `dual_rmsnorm` | T=8192, d=(1536,512) | 0.25 ms (2× sequential RMSNorm) | 0.05 ms | **5.0×** |
| `mha_merge_state` | T=8192, H=128, D=128 | 2.57 ms (torch merge) | 0.80 ms | **3.2×** |
| `fused_ffn` | M=4096, 4096→11008 (fp16) | 5.56 ms (unfused torch) | 5.49 ms | **1.0×** |

Naive baselines: `moe_int4_w4a16` (tuned grouped GEMM, block-align excluded —
see its own row) vs per-expert dequant(int4→bf16)+matmul; `moe_sum_reduce` /
`mha_merge_state` vs their torch oracles; `dual_rmsnorm` vs two sequential
RMSNorm launches; `moe_align_block_size` vs the torch argsort + per-expert
padding reference; `fused_ffn` vs the unfused `reference` backend.

Notes:

- **`moe_int4_w4a16` 23.3×** — the grouped INT4 W4A16 fused-MoE GEMM launches
  from a **checked-in tuned config** (issue #16) instead of runtime autotune, so
  the production path no longer hits Triton's *"Using default MoE kernel config"*
  warning. Winners are swept on MI300A for the two Kimi-K2.6 per-rank shapes —
  `w13` (N=4096, K=7168) and `w2` (N=7168, K=2048) — across decode buckets
  M∈{1,2,4,8,16,32,…} and prefill, stored in
  `src/xkernels/ops/moe/triton/tuned_configs/` keyed by `(E,N,K,device,dtype)`;
  untuned shapes fall back to the autotuner. Per-shape `do_bench` (bf16): gate_up
  M=1 → 0.34 ms, M=8 → 1.15 ms, M=16 → 1.36 ms; down M=1 → 0.12 ms, M=16 →
  0.67 ms. This row isolates the GEMM — the block-align it consumes is the
  `moe_align_block_size` row above (not timed here). Re-tune with
  `sbatch slurm/tune_moe_int4_beverin.sbatch`.
- **`fused_ffn` ≈ 1.0×** — the Triton backend fuses only the SwiGLU *activation*;
  the three projection GEMMs dominate and are torch matmuls in both paths, so
  there is little left to win. Measured in fp16 because on this torch
  2.11+rocm7.2 build the **bf16** GEMM misses the MFMA/hipBLASLt path and runs
  ~470× slower than fp16 (0.8 vs 358 TFLOP/s; see `benchmarks/probe_ffn.py`) —
  a stack issue, not a kernel one.
- **`moe_align_block_size` 32.9×** — the Triton perf backend (vLLM/SGLang-style
  4-stage histogram + padded prefix-sum + scatter, issue #4) is validated
  bit-for-bit against the reference. The win is large because the reference pays
  a full `argsort` plus a 48-iteration per-expert Python padding loop with
  per-step host syncs; the kernel collapses that into 5 launches. The speedup
  holds across token counts (≈14× at M=16, rising to ≈33× at M=16384) — swept in
  `bench_moe_align_block_size.py`.
- **`hierarchical_all_reduce`** (distributed) does *not* beat a flat all-reduce on
  the 2-node / 4-NIC-per-node MI300A stack — RCCL's flat collective is already
  topology-aware. Full analysis in `docs/issue-12-hierarchical-all-reduce.md`.

## Layout

- `src/xkernels/ops/<type>/` — kernels by type; each has `reference.py`,
  `interface.py`, and per-backend subdirs (`triton/`, `cuda/`).
- `src/xkernels/_dispatch.py` — backend registry + selection.
- `tests/`, `benchmarks/`, `examples/` — harness and demos.

See `docs/adding-a-kernel.md` to extend. Design: `docs/superpowers/specs/`.
