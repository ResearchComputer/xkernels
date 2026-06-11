# Issue #17 — bf16 dense-GEMM MFMA/hipBLASLt characterization (gfx942)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2.
**Probe:** `benchmarks/probe_ffn.py` (`slurm/probe_dense_bf16_beverin.sbatch`),
jobs 381251/381255. Metric: TFLOP/s (`2*M*K*N`), Triton `do_bench`.

## TL;DR

The README's "bf16 GEMM is ~470× slower than fp16" is **real but misleading**:
it is specific to the **`torch.matmul` NN layout** (`a[M,K] @ b[K,N]`). The
**production dense path uses `F.linear` (NT layout, weight `[N,K]`)**, and there
**bf16 is already on the MFMA fast path** — it is **fp16** that is slow in NT.
So the Kimi-K2.6 bf16 dense/MLA/shared-expert/`lm_head` projections are **not**
secretly slow on this stack.

The one knob that fixes *every* slow path at once is **`TORCH_BLAS_PREFER_HIPBLASLT=0`**
(route GEMMs through rocBLAS instead of hipBLASLt). It makes NN-bf16 and NT-fp16
fast *and* is ~1.5–2× faster than hipBLASLt even on the already-working NT-bf16
path. **No custom Triton GEMM is needed** — rocBLAS meets or beats a Triton
`tl.dot` ceiling on these shapes.

## What was measured

For each Kimi-K2.6 dense shape `(K→N)` and `M ∈ {1,2,4,8,16,32,512,2048,4096}`,
under four blas-routing modes, four TFLOP/s numbers:

- `nn_bf16` — `torch.matmul`, NN (`a@b`) — the README path.
- `nt_bf16` — `F.linear`, NT (`x @ W^T`, `W=[N,K]`) — the **production** path.
- `nt_fp16` — `F.linear`, NT, fp16.
- `trit_bf16` — a single-tile Triton `tl.dot` bf16 GEMM (NN) — the MFMA ceiling.

A path is flagged when it runs below 10% of the per-cell Triton bf16 ceiling.

## Result by mode

| Mode | `preferred_blas` | NN bf16 | NT bf16 (prod) | NT fp16 |
|------|------------------|---------|----------------|---------|
| `default`      | hipBLASLt (`Cublaslt`) | **SLOW** | fast | **SLOW** |
| `hipblaslt` (`TORCH_BLAS_PREFER_HIPBLASLT=1`) | hipBLASLt | **SLOW** | fast | **SLOW** |
| `no-hipblaslt` (`TORCH_BLAS_PREFER_HIPBLASLT=0`) | **rocBLAS** (`Cublas`) | **fast** | **fast** | **fast** |
| `tunableop` (`PYTORCH_TUNABLEOP_ENABLED=1`) | hipBLASLt + TunableOp | partial↑ | fast | still slow (M≥8) |

`default` and `hipblaslt` are identical — hipBLASLt is already the default
backend on this build, so forcing it changes nothing. `tunableop` (bounded tuning)
partially lifts NN-bf16 at tiny M but left NT-fp16 slow and did not finish the
sweep in the 30-min window; rocBLAS is the clean, complete remedy.

## Representative numbers (TFLOP/s)

`ffn_gate_up` (K=4096, N=11008):

| M | mode | nn_bf16 | nt_bf16 | nt_fp16 | trit_bf16 |
|--:|------|--------:|--------:|--------:|----------:|
| 4096 | default       | 0.8  | 213.0 | 0.7   | 220.2 |
| 4096 | no-hipblaslt  | 397.6 | **421.0** | 412.1 | 218.6 |
| 16   | default       | 0.0  | 29.4  | 0.0   | 16.4  |
| 16   | no-hipblaslt  | 27.7 | 35.4  | 34.9  | 17.9  |

`lm_head` (K=7168, N=32768) and `shexp_down` (K=2048, N=7168) at M=4096:

| shape | mode | nt_bf16 |
|-------|------|--------:|
| lm_head     | default      | 186.9 |
| lm_head     | no-hipblaslt | **376.5** |
| shexp_down  | default      | 203.3 |
| shexp_down  | no-hipblaslt | **397.4** |

Two things to read off these:

1. **Production (NT bf16) is never flagged in any mode** — the decode/prefill
   bf16 projections already hit MFMA. Small-M numbers look low (e.g. q_a M=1 ≈
   0.6 TFLOP/s) but that is launch/latency-bound: the Triton ceiling at the same
   cell is *lower* (0.2), confirming it is not a fast-path miss.
2. **The slow paths cost ~250–500×.** `ffn_gate_up` M=4096: NN-bf16 0.8 vs
   NT-bf16 213 (≈266×); NT-fp16 0.7. This is the cliff the README saw — it lives
   in the NN matmul and NT-fp16 routes, not production NT-bf16.
3. **rocBLAS is ~2× faster than hipBLASLt** on the large-M working bf16 path
   (lm_head 376 vs 187; shexp_down 397 vs 203), in addition to fixing the slow
   routes.

## Conclusion / Phase-2 recommendation

**Document the env remedy; do not build a kernel.**

- **Set `TORCH_BLAS_PREFER_HIPBLASLT=0`** in the serving environment (route dense
  GEMMs through rocBLAS). This (a) eliminates the NN-bf16 and NT-fp16 cliffs and
  (b) speeds up the already-working NT-bf16 path by ~1.5–2× at prefill sizes.
  Validate end-to-end serving throughput before flipping it globally, since it is
  a process-wide backend choice.
- **A Triton `tl.dot` bf16 GEMM is not worth shipping**: the simple ceiling
  kernel (~218 TFLOP/s at M=4096) is matched by torch NT-bf16 under hipBLASLt and
  **beaten** by rocBLAS (~420). A custom kernel would not beat the vendor BLAS on
  these shapes; the win is routing, not a new kernel.
- **Correct the README perf note** (issue #14): the "bf16 misses MFMA, ~470×
  slower than fp16" statement should be scoped to `torch.matmul` (NN); for the
  production `F.linear` (NT) path bf16 is fast and fp16 is the slow one. (Left to
  the Phase-2 PR to keep this characterization PR scoped.)

## Caveats

- `tunableop` was bounded (`PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=30`) and did
  not finish the full sweep; its partial data is suggestive, not complete. The
  `no-hipblaslt` result is complete and is the recommended remedy regardless.
- All numbers are single-GPU microbench TFLOP/s, not end-to-end tok/s. The
  recommendation to switch to rocBLAS should be confirmed against live serving.
