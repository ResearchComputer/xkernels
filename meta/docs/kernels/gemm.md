# Dense GEMMs — bf16 characterization & fp8 block-scale (gfx942)

The dense-GEMM family: the **bf16 dense-GEMM stack characterization** (issue #17),
and the **fp8 block-scale dense GEMM** that powers DeepSeek-V4's MLA / gate /
shared-expert projections — first a correctness-first portable path (issue #38),
then the native fp8 MFMA fast path that finally beats tuned BLAS (issue #41).

> **One-line summary.** (a) On gfx942, bf16 dense GEMMs are *already* on the MFMA
> fast path via the production `F.linear` (NT) layout — the apparent "~470× slower"
> cliff lives only in `torch.matmul` (NN) and is fixed by a single env knob
> (`TORCH_BLAS_PREFER_HIPBLASLT=0`); **no kernel needed**. (b) The fp8 block-scale
> GEMM needs **`float8_e4m3fnuz`** operands to reach the native fp8 MFMA — at
> which point it hits **359 TFLOP/s** and is **3.4–9.1× faster** than the
> torch reference.

---

## bf16 dense GEMM — characterization, not a kernel (issue #17)

**Hardware:** MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2.
**Probe:** `meta/benchmarks/probe_ffn.py`.

### The finding

The README's old "bf16 GEMM is ~470× slower than fp16" note was **real but
misleading**: it is specific to the **`torch.matmul` NN layout** (`a[M,K] @
b[K,N]`). The **production dense path uses `F.linear` (NT layout, weight
`[N,K]`)**, and there **bf16 is already on the MFMA fast path** — it is **fp16**
that is slow in NT. So Kimi-K2.6's bf16 dense/MLA/shared-expert/`lm_head`
projections are **not** secretly slow on this stack.

The one knob that fixes *every* slow path at once is
**`TORCH_BLAS_PREFER_HIPBLASLT=0`** (route GEMMs through rocBLAS instead of
hipBLASLt). It makes NN-bf16 and NT-fp16 fast *and* is **~1.5–2× faster** than
hipBLASLt even on the already-working NT-bf16 path. **No custom Triton GEMM is
needed** — rocBLAS meets or beats a Triton `tl.dot` ceiling on these shapes.

### Modes measured

| Mode | `preferred_blas` | NN bf16 | NT bf16 (prod) | NT fp16 |
|------|------------------|---------|----------------|---------|
| `default`      | hipBLASLt (`Cublaslt`) | **SLOW** | fast | **SLOW** |
| `hipblaslt` (`=1`) | hipBLASLt | **SLOW** | fast | **SLOW** |
| `no-hipblaslt` (`=0`) | **rocBLAS** (`Cublas`) | **fast** | **fast** | **fast** |
| `tunableop` | hipBLASLt + TunableOp | partial↑ | fast | still slow (M≥8) |

`default` and `hipblaslt` are identical — hipBLASLt is already the default on
this build. `tunableop` partially lifts NN-bf16 at tiny M but left NT-fp16 slow;
rocBLAS is the clean, complete remedy.

### Representative numbers (`ffn_gate_up`, K=4096, N=11008)

| M | mode | nn_bf16 | nt_bf16 | nt_fp16 | trit_bf16 (ceiling) |
|--:|------|--------:|--------:|--------:|----------:|
| 4096 | default       | 0.8  | 213.0 | 0.7   | 220.2 |
| 4096 | no-hipblaslt  | 397.6 | **421.0** | 412.1 | 218.6 |
| 16   | default       | 0.0  | 29.4  | 0.0   | 16.4  |
| 16   | no-hipblaslt  | 27.7 | 35.4  | 34.9  | 17.9  |

Large-shape NT bf16 at M=4096: rocBLAS `421 TFLOP/s`, ~2× over hipBLASLt's 213.

### Conclusion

**Set `TORCH_BLAS_PREFER_HIPBLASLT=0`** in the serving environment. A Triton
`tl.dot` bf16 GEMM is **not worth shipping** — the simple ceiling kernel
(~218 TFLOP/s at M=4096) is matched by torch NT-bf16 under hipBLASLt and beaten
by rocBLAS (~420). The win is routing, not a new kernel. (Small-M decode numbers
look low but are launch/latency-bound: the Triton ceiling at the same cell is
*lower*, confirming it is not a fast-path miss.)

---

## fp8 block-scale dense GEMM — `mm_fp8_blockscale`

The V4 hot path: every MLA projection, prefill **and** decode. Upstream all
`tokenspeed_kernel.ops.gemm.mm` backends were NVIDIA-only (`deep_gemm`,
`flashinfer`, and a `vendors={"nvidia"}`-gated triton path), so on gfx942 the only
selectable kernel was the slow `torch_mm_fp8_blockscale` reference (full fp32
materialization + dense fp32 matmul, no MFMA).

### The math (DeepSeek block-scale, `block = 128`)

```
out[M, N] = A_deq @ B_deq.T
```
- `A [M, K]` fp8 e4m3, **per-token-group** scale `A_scales [M, ceil(K/128)]`.
- `B [N, K]` fp8 e4m3 (Linear orientation), **per-block** scale
  `B_scales [ceil(N/128), ceil(K/128)]`.

The scales are **constant within a 128-K quant block**, so they factor out of the
inner contraction (this is the key to the MFMA fast path, see below). The quant
helpers `per_token_group_quant_fp8` / `per_block_quant_fp8` produce these
operands; `fp8_dtype=` selects the encoding.

### Path 1 — portable dequant-then-`tl.dot` (issue #38, the always-correct fallback)

One Triton program per `(row-tile, col-tile)`. Compute tiles are decoupled from
the quant `block` but constrained so each tile lands inside one quant block on N
and K (`block % BLOCK_N == 0`, `block % BLOCK_K == 0`) — that makes each tile's
`B` block-scale a scalar and its `A` group-scale per-row, so dequant is a cheap
broadcast. The K loop streams `BLOCK_K` columns: load fp8 `A`/`B`, upcast to fp32,
multiply by the two block scales, `tl.dot`-accumulate in fp32. Dequant happens
per tile in registers — the full dequantized operands are **never** materialized
in DRAM.

`dot_bf16` (opt-in, default **False**): cast the block-scaled operands to bf16 so
`tl.dot` runs on the CDNA3 bf16 MFMA path. Faster in principle but only
~bf16-accurate and, on these shapes, not actually faster than the fp32 dot.

**Honest negative result.** This kernel is **slower than the torch reference on
every shape** (rel 7e-4, correct, but ~21–37 TFLOP/s vs the ~400 ceiling):

| M | N | K | portable_fp32 | portable_bf16 | torch_ref | bf16/ref |
|--:|--:|--:|------------:|------------:|----------:|---------:|
| 1    | 512  | 7168 | 0.363 ms | 0.447 ms | **0.187 ms** | 0.42× |
| 2048 | 512  | 7168 | 0.612 ms | 0.701 ms | **0.540 ms** | 0.77× |
| 4096 | 7168 | 2048 | 2.848 ms | 3.232 ms | **1.888 ms** | 0.58× |

Why: dequant-then-fp32/bf16-`tl.dot` does **not** use the native fp8 MFMA path,
and `torch_mm_fp8_blockscale` routes its dense matmul through a tuned BLAS that
beats this naive Triton tiling. The small 64³ tiles also leave the MFMA
underutilized. (LDS note: a 64×128 tile hit `OutOfResources: shared memory,
Required 73728 > 65536` — CDNA3 has 64 KB LDS/CU — which is why tiles stay 64³.)

### Path 2 — native fp8 MFMA fast path (issue #41, the winner)

The follow-up: `tl.dot` **directly on fp8 operands**, with the block scales
applied as a post-accumulation correction, plus an autotuned tile space. On the V4
shapes it runs **3.4–9.1× faster than `torch_mm_fp8_blockscale`** and reaches
**359 TFLOP/s** at prefill (near the gfx942 ceiling).

#### Two-level (block-promoted) accumulation

Because the scales are constant within a 128-K block, the inner sum is a **raw
fp8·fp8 partial GEMM** — the native fp8 MFMA:

```
out[m,n] = Σ_kb  a_s[m,kb]·b_s[n//128,kb] · ( Σ_{k ∈ block kb} A_fp8[m,k]·B_fp8[n,k] )
```

Per quant K-block `kb`: accumulate the raw partial in an fp32 block-accumulator
via a fp8 `tl.dot`, then promote into the main fp32 accumulator scaled by
`a_s[:,None]·b_s[None,:]`. `BLOCK_N` becomes a free tuning knob (generalizing the
portable path's "`BLOCK_N` divides 128 → scalar" constraint). This is the *same
real arithmetic* as the exact fp32-dequant path, so parity is tight (rel 2–4e-6).

#### The fp8 format that reaches native fp8 MFMA — fnuz, not fn

This is the load-bearing lesson. The quant helpers produce
`torch.float8_e4m3fn` (OCP, bias 7, max 448). CDNA3's native fp8 MFMA decodes the
**fnuz** encoding (bias 8, max 240). A standalone probe
(`meta/benchmarks/probe_fp8_mfma.py`, dumping the AMDGCN) settled it:

| operands | parity | MFMA instruction | throughput |
|---|---|---|---|
| `float8_e4m3fn`   | exact (2e-7) | `v_mfma_f32_32x32x8_**f16**` (upcast) | 29.7 TFLOP/s |
| `float8_e4m3fnuz` | exact (2e-7) | `v_mfma_f32_32x32x16_**fp8_fp8**` ✅ | **373.9 TFLOP/s** |

So `e4m3fn` **silently upcasts** to an f16 MFMA (no fp8 win); **`e4m3fnuz` lowers
to the native fp8 MFMA at ~374 TFLOP/s** — a 12.6× gap. Both are numerically
exact, so the discriminator is the MFMA mnemonic + throughput, not parity. The
kernel is fp8-format-agnostic (it dots whatever dtype it is given); the fast path
is **unlocked by feeding it fnuz operands** (`quant(..., fp8_dtype=torch.float8_e4m3fnuz)`).

#### Autotune — N is the dominant axis

`get_fp8_gemm_config` is a baked direct-launch table (no per-call runtime
autotune), tuned on beverin via `meta/benchmarks/tune_fp8_blockscale_gemm.py` over
the full CDNA3 config space. Key finding: **N, not M, drives the choice.** The
N=512 MLA projections starve a 304-CU GPU with big 128×256 tiles (only ~32
workgroups → 78 TFLOP/s); tiny 64×64 / 16×16-MFMA tiles give many more
workgroups → **250 TFLOP/s (3.2×)**.

| regime | tile | MFMA |
|---|---|---|
| `N ≤ 1024` (e.g. N=512) | `BM64 BN64 BK128`, warps 4, stages 2 | 16×16×32 fp8 |
| `N > 1024`, decode `M ≤ 16` | `BM16 BN128 BK128`, warps 4, stages 2 | 16×16×32 fp8 |
| `N > 1024`, prefill | `BM128 BN128 BK128`, warps 8, stages 2 | 32×32×16 fp8 |

`BLOCK_K = 128` (one full quant block) + 2 stages fits the 64 KB CDNA3 LDS
because fp8 operands are half the bytes of the portable path's fp32 tiles —
directly resolving the `OutOfResources` that forced the portable path down to 64³.

#### Performance (the honest positive result)

`do_bench`, bf16 out, native fp8 MFMA (fnuz operands):

| M | N | K | mfma (fnuz) | TFLOP/s | portable (#38) | torch_ref | mfma/ref | mfma/portable |
|--:|--:|--:|------------:|--------:|-------------:|----------:|---------:|--------------:|
| 1    | 512  | 7168 | **0.053 ms** | — | 0.363 ms | 0.185 ms | **3.48×** | 6.8× |
| 2048 | 512  | 7168 | **0.060 ms** | 250 | 0.619 ms | 0.550 ms | **9.13×** | 10.3× |
| 4096 | 7168 | 2048 | **0.335 ms** | 359 | 2.884 ms | 1.913 ms | **5.72×** | 8.6× |

### What ships / how to select

`path ∈ {auto, mfma, portable}` on the `Backend.TRITON` entry:
- `path="auto"` (default) routes **fnuz** operands to the mfma fast path and
  keeps **fn** operands on the portable fallback (fn → f16 MFMA is *slower* than
  the portable path, so "fastest available").
- The portable kernel remains the always-correct, format-agnostic fallback.

**Requires fnuz operands for the speedup** — the standard practice for native
fp8 MFMA on MI300. Use the helper `preferred_fp8_dtype(device)` to pick the
right one portably (returns `float8_e4m3fnuz` on AMD CDNA, `float8_e4m3fn`
elsewhere), or pass `fp8_dtype="auto"` to the quant helpers. Wiring the serving
path to the xkernels op + fnuz quantization is a tokenspeed change (out of scope
here, as in the other V4 ops).

### Reproduce

```bash
# the fp8 MFMA format probe (settles fn vs fnuz)
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/probe_fp8_mfma.py
# the V4-shape sweep (ms + TFLOP/s)
scripts/cluster.sh submit --host beverin scripts/archive/issues/bench_fp8_blockscale_beverin.sbatch
# the autotune table builder
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/tune_fp8_blockscale_gemm.py
# correctness
scripts/cluster.sh run --host beverin python3 -u tests/test_mm_fp8_blockscale.py
scripts/cluster.sh run --host beverin python3 -u tests/test_mm_fp8_blockscale_mfma.py
```
