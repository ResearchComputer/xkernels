# Normalization, reductions & the FFN activation fusion

The lighter-weight kernels that did not get a dedicated issue write-up — their
optimization story lives in the cross-cutting benchmark/profile campaign
(`meta/wiki/`) and the GB10 CUTE testbed (`meta/docs/usage/ds5-testbed.md`). This
page rounds out the per-kernel coverage so every op in the performance table has a
home.

- **`dual_rmsnorm`** — fused dual-branch RMSNorm (Kimi-K2.6).
- **`fused_ffn`** — the SwiGLU activation fusion.
- **`moe_sum_reduce` / `mha_merge_state`** — covered in `kernels/moe.md` and
  `kernels/attention.md` respectively (they are the MoE combine and the split-KV
  merge).

> **One-line summary.** `dual_rmsnorm` (4.2–9.8×) and `fused_ffn` (~1.0×) are
> both **memory/launch-bound** and **already at the roofline** — there is no
> compute headroom to chase. The honest read on `fused_ffn` is that the Triton
> kernel only fuses the *activation*; the three projection GEMMs are
> `torch.matmul` in both paths, so there is little left to win.

---

## `dual_rmsnorm` — fused dual-branch RMSNorm

Computes RMSNorm over two concatenated branches `d=(1536,512)` in one launch
(`xkernels.ops.norm`). One CTA per row, a thread-stride Kahan sum-of-squares →
`warp_reduction_sum`/SMEM partials → `rsqrt` → scale·x·w.

| Arch | shape | naive (ms) | optimized (ms) | speedup |
|---|---|---:|---:|---:|
| MI300A (gfx942) | T=8192, d=(1536,512) | 0.238 | 0.054 | **4.4×** |
| A100 (sm_80)    | T=8192, d=(1536,512) | 0.517 | 0.053 | **9.8×** |
| GB10 (sm_121, CUTE) | sweep | — | 0.061 | — |

**Profile verdict: memory-bound, at the roofline.** MI300A AI 3.9 F/B; A100 68%
DRAM, 54% compute, 92% occupancy — i.e. **~1.3–1.6 TB/s of the ~1.94 TB/s peak**.
There is little left to tune on the bandwidth axis; the wins in the table are the
wins. (At the sweep sizes the CUTE card is also **launch-bound**, <0.5 MB — its
low BW% is launch overhead, not bandwidth.) The CUTE card also demonstrated the
**verify harness tolerance fix** live: a near-zero reference element had
max_rel=5.2 which the old AND criterion false-failed; the combined `|a−e| ≤ atol +
rtol·|e|` criterion (now in `verify.py`) passes it via atol.

---

## `fused_ffn` — the SwiGLU activation fusion

The Triton backend fuses only the **SwiGLU activation**; the three projection
GEMMs dominate and are `torch.matmul` in both the reference and triton paths.
Measured in **fp16** (not bf16) because on this torch 2.11+rocm7.2 build the
**bf16** `torch.matmul` (NN layout) path misses MFMA/hipBLASLt and runs ~470×
slower than fp16 (the `kernels/gemm.md` / issue #17 cliff — the production
`F.linear` NT layout bf16 path is fast; the slowdown is specific to the NN
benchmark shape).

| Arch | shape | naive fp16 (ms) | optimized (ms) | speedup |
|---|---|---:|---:|---:|
| MI300A (gfx942) | M=4096, 4096→11008 | 5.425 | 5.285 | **1.03×** |
| A100 (sm_80)    | M=4096, 4096→11008 | 4.744 | 4.288 | **1.11×** |

**Profile verdict: the speedup is a rounding error against three GEMMs.** Both
paths run the same `torch.matmul` GEMMs at ~205–210 TFLOP/s (MI300A MFMA regime);
the fused SwiGLU Triton kernel only saves one elementwise launch. **No fix skill
applies to the Triton kernel** — it is a correct, already-fast elementwise fusion.
(The card's `compute_bound` regime honestly describes the op the card is embedded
in, not the kernel the card ships. The earlier "15.9% of MFMA-F16 peak" reading
came from a broken rocprof per-kernel FLOPs column — see `meta/wiki/03-profiling.md`.)

---

## Reproduce

```bash
# the full table (dual_rmsnorm + fused_ffn both in bench_all)
scripts/cluster.sh submit --host beverin                              # bench_all_beverin.sbatch
scripts/cluster.sh run --host bristen python3 -u meta/benchmarks/bench_all.py
# the bf16-GEMM stack characterization (why fused_ffn is measured fp16)
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/probe_ffn.py
```
