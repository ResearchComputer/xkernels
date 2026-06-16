# Issue #38 — DeepSeek-V4 fp8 block-scale dense GEMM (gfx942)

This kernel adds a portable, AMD-native fp8 block-scale dense GEMM for gfx942
(MI300A / CDNA3). On the DeepSeek-V4-Flash forward, the MLA (`q_a`/`kv_a`/`q_b`/
`kv_b`), `gate`, and shared-expert projections are stored **fp8 block-scale**.
The vendored `tokenspeed_kernel.ops.gemm.mm` backends are all NVIDIA-only
(`deep_gemm_mm_fp8_blockscale`, `flashinfer_mm_fp8_blockscale`, and
`triton_mm_fp8_blockscale` — the last gated by `vendors={"nvidia"}`), so the only
selectable kernel on gfx942 was the slow `torch_mm_fp8_blockscale` reference
(`(A.float()*A_scales) @ (B.float()*B_scales).T`): full fp32 materialization plus
a dense fp32 matmul, no MFMA. This issue is the V4 hot path (every MLA projection,
prefill **and** decode), so that reference dominates serve latency.

The new xkernels-native op is `mm_fp8_blockscale`.

## The math

Standard DeepSeek block-scale layout (`block = 128`):

```
out[M, N] = A_deq @ B_deq.T
```

- `A [M, K]` fp8 e4m3, **per-token-group** scale `A_scales [M, ceil(K/block)]`
  (one scale per contiguous `block`-length group along K — a `1×block` tile).
- `B [N, K]` fp8 e4m3 (Linear orientation), **per-block** scale
  `B_scales [ceil(N/block), ceil(K/block)]` (one scale per `block×block` tile).

| Symbol | V4 example | Meaning |
|--------|-----------|---------|
| `K` | 7168 | DeepSeek hidden (MLA in-dim) |
| `N` | 512 / 7168 | projection out-dim |
| dtype(A,B) | fp8 e4m3 | quantized operands |
| dtype(out) | bf16 | default (fp32 supported) |

The quant helpers `per_token_group_quant_fp8` / `per_block_quant_fp8` produce these
operands; they are exact dequant oracles (OCP-style `amax/FP8_MAX` per group).

## Kernel strategy

One Triton program per `(row-tile, col-tile)`. Compute tiles (`BLOCK_M`/`BLOCK_N`/
`BLOCK_K`) are decoupled from the quant `block` but constrained so each tile lands
inside one quant block on the N and K axes (`block % BLOCK_N == 0`,
`block % BLOCK_K == 0`). That makes each tile's `B` block-scale a single scalar and
its `A` group-scale per-row, so dequant is a cheap broadcast. The K loop streams
`BLOCK_K` columns: load the fp8 `A`/`B` tiles, upcast to fp32, multiply by the two
block scales, and `tl.dot`-accumulate in fp32. Dequant happens per tile in
registers — unlike `torch_mm_fp8_blockscale`, the full dequantized operands are
never materialized in DRAM. Tiles are kept small (64³) so the LDS footprint stays
under the 64 KB CDNA3 limit (an early 64×128 tile hit `OutOfResources: shared
memory, Required 73728 > 65536`).

`dot_bf16` (opt-in, default **False**): when True, the block-scaled operands are
cast to bf16 so `tl.dot` runs on the CDNA3 bf16 MFMA path (fp32 accumulate). This
is faster in principle but only ~bf16-accurate and, on these shapes, not actually
faster than the fp32 dot (see perf below). Default False keeps the exact-fp32 dot.

## Validation

Offline tests live in `tests/test_mm_fp8_blockscale.py`:

- **Interpreter** (`TRITON_INTERPRET=1`, CPU fp32): exercises tiling, masking, and
  the block-scale index math against the fp32 dequant oracle. The bf16-dot path is
  skipped (the CPU interpreter mis-evaluates a bf16 `tl.dot`).
- **GPU** (gfx942): default fp32-dot path vs oracle (tight, norm-relative `< 1e-3`);
  opt-in bf16-dot path at norm-relative `< 2e-2`; block-aligned and odd M/N/K
  (incl. K not a multiple of 128); decode `M=1`; bf16 output; empty `M`.

On-device run (beverin, AMD Instinct MI300A / gfx942, torch 2.11.0+rocm7.2,
hip 7.2.26015, real Triton compile via the `tokenspeed_triton` AMD backend —
job 383571):

| Check | Shape | Result |
|-------|-------|--------|
| `pytest tests/test_mm_fp8_blockscale.py` | — | **17 passed** (25.2 s) |
| oracle parity (fp32 dot) | M=1/8/2048/4096, K=7168/2048 | **rel 7.1e-04 – 8.7e-04** |

## Performance — an honest negative result

`triton.testing.do_bench`, bf16 output, vs the `torch_mm_fp8_blockscale` reference:

| M | N | K | triton_fp32 | triton_bf16 | torch_ref | bf16 / ref |
|--:|--:|--:|------------:|------------:|----------:|-----------:|
| 1    | 512  | 7168 | 0.363 ms | 0.447 ms | **0.187 ms** | 0.42× |
| 8    | 512  | 7168 | 0.361 ms | 0.430 ms | **0.189 ms** | 0.44× |
| 2048 | 512  | 7168 | 0.612 ms | 0.701 ms | **0.540 ms** | 0.77× |
| 4096 | 7168 | 2048 | 2.848 ms | 3.232 ms | **1.888 ms** | 0.58× |

The Triton kernel is **slower** than the torch reference on every shape, and the
bf16-dot path is slower than the fp32-dot path. This is the same lesson as
[#17](issue-17-bf16-dense-gemm.md) and the [#20](issue-20-fused-combine.md)
precedent: a correct optimization the hardware does not (yet) reward.

Why: dequant-then-fp32/bf16-`tl.dot` does **not** use the native fp8 MFMA path, and
on the rocm7.2 stack `torch_mm_fp8_blockscale` routes its dense fp32 matmul through
a tuned BLAS that beats this naive Triton tiling. The small 64³ tiles also leave the
MFMA underutilized (21–37 TFLOP/s on the large shapes vs the ~400 TFLOP/s ceiling
characterized in #17).

## What ships and what is next

**Ships:** a correct, portable, drop-in `mm_fp8_blockscale` so gfx942 finally has a
selectable, non-torch fp8 block-scale GEMM — closing the functional gap. The fp32
dot is the default (exact); the bf16 dot is opt-in and off by default.

**Does not ship as the fast path:** this kernel is not a speedup on the V4 shapes,
so it does not displace `torch_mm_fp8_blockscale` on latency grounds yet. A genuine
win needs native **fp8 MFMA** (`tl.dot` on fp8 e4m3 operands, with the block scales
applied as a post-`dot` correction per K-block) and **autotuned tiles** (larger
`BLOCK_M`/`BLOCK_N`, K-block-aligned `BLOCK_K=128`, multi-stage pipelining) — a
follow-up. Until then, the practical remedy mirrors #17: keep serving on the tuned
BLAS reference, with this kernel available as the portable correctness fallback.
