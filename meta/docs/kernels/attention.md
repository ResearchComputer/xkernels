# Attention — sparse MLA, the DSA indexer, and the KV gather

The attention family is what makes **DeepSeek-V4** servable on AMD MI300A
(gfx942). Upstream binds the V4 attention path to NVIDIA-only packages
(`flash_mla`, `deep_gemm`, CUDA mxfp4 paged gather) that were `error_fn` /
unverified on AMD. These kernels are portable Triton replacements, each validated
on-device:

- **`sparse_mla_attention`** — the sparse-MLA attention *compute* (the softmax
  over V4's latent KV) — issue #32. The last gating kernel for V4 serving.
- **`dsa_indexer_logits`** — the weighted ReLU-MQA logits that *select* which KV
  survive — issue #27.
- **`mxfp4_paged_gather`** — gather selected positions out of a paged mxfp4 KV
  cache and dequantize — issue #28.
- **`mha_merge_state`** — the online-softmax split-KV merge (issue #3), the
  support kernel the sparse-MLA path is shaped to consume for long top-k.

> **One-line summary.** V4's hybrid CSA/HCA/SWA attention is *selection*-side
> (the DSA indexer + per-ratio gather), not the softmax — so **one compute kernel
> serves all three variants**. Decode gathers + dequants to bf16 first, then
> reuses the single softmax compute kernel.

---

## `sparse_mla_attention` — the MLA softmax compute (issue #32)

The kernel that consumes the DSA indexer's top-k KV selection (#27/#28) and runs
the actual attention softmax over V4's latent KV.

### The math

MLA in latent form is **MQA**: a single shared latent KV per position of dim
`D = kv_lora_rank + rope` (V4: `512 = 448 + 64`). The score uses the full `D`;
the value is the first `d_v` dims (the `kv_lora` / nope part). Per query token
`t`, head `h`, over the indexer-selected positions `idx[t, :]`:

```
s[t,h,j] = sm_scale * (q[t,h] · kv[idx[t,j]])         # full D
p        = softmax([s[t,h,:], sink[h]])                # optional per-head sink
out[t,h] = Σ_j p[j] · kv[idx[t,j], :d_v]               # value = nope part
```

The **attention sink** is a per-head logit that joins the softmax denominator and
contributes no value (`out → 0` for a token whose only "mass" is the sink). A
column `j` is valid when `idx >= 0` **and** (when given) `j < topk_length[t]`.

**Variant-agnostic.** V4's hybrid Compressed-Sparse / Heavily-Compressed /
Sliding-Window distinction is *selection*-side, not the softmax — so this one
compute kernel serves all three. At decode the two index sets (SWA + compressed
CSA) are unioned into a single softmax.

### Entry points (upstream-faithful names)

- **`sparse_mla_attention(q, kv, indices, *, sm_scale, topk_length=, attn_sink=, d_v=)`**
  — the native op. `q [T,H,D]`, shared `kv [Kv,D]`, `indices [T,topk]`. Returns
  `(out [T,H,d_v], lse [T,H], max_logits [T,H])`.
- **`flash_mla_sparse_fwd(...)`** — prefill (faithful name). `kv` is the bf16
  latent workspace `[Kv,1,D]`, `indices` is `[T,1,topk]`.
- **`flash_mla_with_kvcache(...)`** — decode (faithful name). Gathers +
  dequantizes the paged **fp8_ds_mla** primary (SWA) and optional compressed
  (CSA) caches, unions the selections, runs one softmax.
- **`get_mla_metadata(*args, **kwargs)`** — returns `(placeholder, num_splits=1)`.

### Kernel strategy

One Triton program per `(token, head)`, streaming the top-k in `BLOCK_N=64`
chunks with online (flash) softmax: running max + denominator + a `[D]` fp32
accumulator (value stored for the first `d_v` dims). The sink folds in after the
stream. An all-masked chunk is guarded so an empty-window token never NaNs.
Decode gathers + dequants to bf16 first, then reuses this one compute kernel via
a flattened `(kv, indices)`. Deferred: split-KV + `mha_merge_state` merge for
long top-k, and a fused Triton fp8 gather.

### fp8_ds_mla latent-KV layout (decode)

Pinned from the tokenspeed cache writer. Per latent token:

- **value region** `nope_dim + rope_dim*2` bytes: `nope_dim` (448) fp8 **e4m3**
  (value-bearing) then `rope_dim` (64) **bf16** (rope, score-only).
- **scale region** `nope_dim//quant_block` (7) uint8 exponents `enc` + 1 pad byte,
  one per `quant_block=64` group along nope.

Dequant: `nope = fp8_e4m3(byte) * 2**(enc - 127)` per group; `rope = bf16`.
`src/xkernels/ops/attention/sparse_mla.py` ships `dequant_fp8_ds_mla` and a
`make_fp8_ds_mla_kv` generator (exact inverse, for offline unit tests).

### Validation (on-device gfx942, job 382459)

| Check | Shape | Result |
|-------|-------|--------|
| `pytest tests/test_sparse_mla_attention.py` (bf16) | — | **16 passed** |
| prefill Triton vs oracle | H=128, D=512, d_v=448, topk=512 | **max\|err\|=1.95e-03, rel=9.3e-04** |
| decode dual-cache + single-cache parity | nb×bs paged fp8_ds_mla | **passed** |

### Performance (issue #39 perf pass)

At V4 geometry (H=128, D=512, d_v=448, MQA): `BLOCK_N` is a pure perf knob (the
flash reduction is exact for any chunk size). **No static winner** — the best
`BLOCK_N` depends on the query-token count:

| Tq | topk | base (ms) | best (ms) | speedup | best cfg |
|---:|-----:|---:|---:|---:|---|
| 1 | 256  | 0.0206 | **0.0182** | **1.13×** | BN=128 w8 wpe=1 |
| 1 | 512  | 0.0325 | **0.0272** | **1.20×** | BN=128 w8 wpe=1 |
| 1 | 1024 | 0.0570 | **0.0459** | **1.24×** | BN=128 w8 wpe=1 |
| 8 | 256–1024 | — | — | ~1.01× | **BN=64 (default)** |

At Tq>1 the #32 default `BLOCK_N=64` is already best and larger `BLOCK_N`
*regresses* (up to ~4.3× slower at Tq=8 topk=1024 — the `[D]=512` fp32
accumulator dominates VGPR/LDS). So `BLOCK_N=64` **stays the default** (no
multi-token regression) and the Tq=1 win ships **opt-in, off by default**
(`DECODE_SPARSE_MLA_CONFIG`, overridable via `XKERNELS_SPARSE_MLA_CONFIG`).

Headline end-to-end (T=8, H=128, D=512, topk=512): **0.11 ms, 26.8×** vs the torch
gather+softmax reference. Profile verdict at decode: **compute-bound** (MI300A
AI = 70.6 F/B — the card's `memory_bound` classification reflects a different,
larger operating point). An MFMA-tiled score/value path (currently `tl.sum`, not
`tl.dot`; the `matrix_instr_nonkdim`/`kpack` knobs are threaded but inert) is the
deferred follow-up.

---

## `dsa_indexer_logits` — the KV selection (issue #27)

DeepSeek-V4's attention is driven by a **DSA indexer** that scores every cached KV
position per query and selects the top-512 (Flash) / 1024 (Pro). Upstream computes
the indexer logits with **NVIDIA-only** kernels (`deep_gemm.fp8_fp4_mqa_logits` +
a CUDA mxfp4 paged gather); the AMD branch was unverified.

### What the indexer computes

The numerically meaningful operation — the one that *selects* which KV survive —
is a **weighted ReLU MQA** dot-product followed by a masked top-k:

```
logits[t, j] = sum_h  weights[t, h] * relu( q[t, h, :] . k[j, :] )
```

with `q : [T, H, D]` (`H = index_n_heads = 64`, `D = index_head_dim = 128`), a
**single shared** `k : [K, D]` per KV position (MQA), and per-head combine
`weights : [T, H]`. An optional causal window masks out-of-range columns to
`-inf` before the top-k. The fp8/fp4 packing in the upstream CUDA kernel is a
hardware encoding detail; it does not change which KV are selected, so the gfx942
path computes the logits directly in fp32 from bf16/fp16 q/k.

This ships a **portable Triton replacement** (`xkernels.dsa_indexer_logits`) plus
a thin `dsa_indexer_topk` (a `torch.topk`) for the selection.

### Kernel shape

One Triton program handles one `(query, KV-tile)` pair: it loads the full `[H, D]`
query (`H=64`, `D=128` fit in registers), streams a `BLOCK_K=64`-row tile of the
shared MQA key, computes `tl.dot(q, kᵀ)` → ReLU → per-head weight → sum over heads
in fp32, then applies the causal mask. Grid is `(T, cdiv(K, 64))`.

### Validation (on-device gfx942, job 381969)

| Check | Result |
|---|---|
| pytest `test_dsa_indexer_logits.py` (7 cases, GPU bf16) | **7 passed** |
| V4 shape `H=64 D=128 K=4096` bf16: `max\|err\|` vs fp32 oracle | **6.10e-05** (rel 1.76e-07) |
| Flash top-512 selection: mean Jaccard(top-k set) vs oracle | **1.0000** |

### Gotcha hit (interpreter ≠ compiler)

The first on-device run failed to *compile* the masked branch: referencing a
module-level `_NEG_INF = float("-inf")` from inside the `@triton.jit` kernel is
rejected by the real Triton compiler ("Cannot access global variable … not
instantiated as constexpr"), but `TRITON_INTERPRET=1` happily allowed it. Fixed by
inlining `float("-inf")`. This is the documented interpreter-vs-compiler
constexpr-globals trap — on-device validation caught what CPU could not.

### Scope

This ships the **dequantized-math equivalent** (bf16/fp16 in, fp32 logits out):
correct, portable, validated to match the oracle. A native fp8/fp4 gfx942 logits
kernel would be a bandwidth optimization on top — its own measured follow-up, not
required for a *correct* forward path. Wiring into `models/deepseek_v4.py` lives
in the tokenspeed runtime, not this kernels package.

---

## `mxfp4_paged_gather` — gather + dequant selected KV (issue #28)

The gfx942 **Triton** replacement for the CUDA-only `indexer_mxfp4_paged_gather`
(one of the two blockers under the V4 tracking issue). DeepSeek-V4's DSA indexer
selects the top-512 (Flash) / 1024 (Pro) KV positions per query; this op gathers
those positions out of a **paged** (block-table indexed) **mxfp4** KV cache and
dequantizes them to bf16 for the attention compute.

```python
mxfp4_paged_gather(kv_packed, kv_scale, block_table, sel_pos, *,
    block_size, group_size=32, out_dtype=torch.bfloat16, backend="auto")
    # -> [num_seqs, topk, head_dim]
```
Padded selection slots (`sel_pos < 0`) yield a zero row, matching the CUDA kernel.

### mxfp4 format (OCP MX)

- **E2M1** FP4 element, two per `uint8` (low nibble = even index). The 8 magnitudes
  `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` are decoded **arithmetically** in-kernel (no LUT
  load): with code `c = nib & 7`, exp `e = (c>>1)&3`, mantissa `m = c&1`,
  `|x| = m*0.5` for `c<2` else `(1 + m*0.5) * 2**(e-1)`.
- **E8M0** block scale, one `uint8` per `group_size=32` elements along head_dim:
  `2**(byte - 127)`; the reserved `0xFF` NaN code maps to `0`.

Because every FP4 code is represented exactly in fp32/bf16, the dequant is
**exact** — the only error is the final bf16 round, which here is `0.0000` against
the oracle.

### Validation + perf (on-device gfx942, job 381968)

GPU bf16 correctness: **6/6 tests pass** (Triton vs the torch paged-gather oracle).
Decode-shape timing (head_dim=128, block_size=64), one program per `(seq, slot)`:

| num_seqs | topk | triton (ms) | max\|err\| |
|---:|---:|---:|---:|
| 16 | 512  | 0.0291 | 0.0000 |
| 32 | 512  | 0.0281 | 0.0000 |
| 64 | 512  | 0.0275 | 0.0000 |
| 64 | 1024 | 0.0423 | 0.0000 |

### Scope

This unblocks the **gather half** of #27. It does **not** provide the
`deep_gemm.fp8_fp4_mqa_logits` indexer-logits kernel (the other CUDA-only piece),
and it does not touch MoE expert parallelism. A full V4 forward still requires
the indexer-logits path plus MoE EP; this kernel is a self-contained, validated
building block toward that bring-up.

---

## `mha_merge_state` — split-KV online-softmax merge (issue #3)

The support kernel the sparse-MLA decode path is shaped to consume for long
top-k (deferred wiring). One CTA per `(t,h)` row, threads tile D; per-row scalar
weights computed locally per thread (memory-bound hides the transcendentals).
Natural exp/log (`math.exp`/`math.log`) matching the reference.

End-to-end (T=8192, H=128, D=128): **0.784 ms (MI300A) / 1.046 ms (A100), 3.1× /
4.9×** over the torch merge. Profile verdict: **balanced, compute-leaning** (the
only op where ncu Compute (66%) ≫ DRAM (41%); occupancy held to 76% by
scheduling/load imbalance, not registers). On the GB10 CUTE testbed, a
**bf16-native-read** perf pass (read bf16, accumulate fp32 — lossless, halves
read traffic) took the CUTE card **0.084→0.042 ms (2.0×)**, 28%→56% of GB10 peak
BW.

---

## Reproduce

```bash
# sparse MLA correctness + the V4 perf sweep
scripts/cluster.sh run --host beverin python3 -u tests/test_sparse_mla_attention.py
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/tune_sparse_mla.py   # #39
# DSA indexer + mxfp4 gather
scripts/cluster.sh run --host beverin python3 -u tests/test_dsa_indexer_logits.py
scripts/cluster.sh submit --host beverin scripts/archive/issues/probe_mxfp4_gather_beverin.sbatch
# merge_state
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/bench_all.py
```
