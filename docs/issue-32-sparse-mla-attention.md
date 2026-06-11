# Issue #32 — DeepSeek-V4 sparse-MLA attention compute (gfx942)

The last gating kernel for serving **DeepSeek-V4** on **AMD MI300A (gfx942)**: the
sparse-MLA attention *compute* that consumes the DSA indexer's top-k KV selection
(#27/#31, #29) and runs the actual attention softmax over V4's latent KV. Upstream
binds `flash_mla_sparse_fwd` / `flash_mla_with_kvcache` / `get_mla_metadata` to
DeepSeek's NVIDIA-only `flash_mla` package on Hopper+; on AMD they were `error_fn`.

This provides a portable Triton replacement, exposed as a clean xkernels-native op
(`sparse_mla_attention`) re-exported under those upstream-faithful names.

## The math

MLA in latent form is **MQA**: a single shared latent KV per position of dim
`D = kv_lora_rank + rope` (V4: `512 = 448 + 64`). The score uses the full `D`; the
value is the first `d_v` dims (the `kv_lora` / nope part). Per query token `t`,
head `h`, over the indexer-selected positions `idx[t, :]`:

```
s[t,h,j] = sm_scale * (q[t,h] · kv[idx[t,j]])         # full D
p        = softmax([s[t,h,:], sink[h]])                # optional per-head sink
out[t,h] = Σ_j p[j] · kv[idx[t,j], :d_v]               # value = nope part
```

The attention **sink** is a per-head logit that joins the softmax denominator and
contributes no value (`out → 0` for a token whose only "mass" is the sink). A
column `j` is valid when `idx >= 0` **and** (when given) `j < topk_length[t]`.

**Variant-agnostic.** The kernel attends over whatever indices it is handed. V4's
hybrid Compressed-Sparse / Heavily-Compressed / Sliding-Window distinction is
*selection*-side (the DSA indexer + the per-ratio gather), not the softmax — so
this one compute kernel serves all three. At decode the two index sets (SWA +
compressed CSA) are unioned into a single softmax.

## Entry points

- **`sparse_mla_attention(q, kv, indices, *, sm_scale, topk_length=, attn_sink=,
  d_v=)`** — the native op. `q [T,H,D]`, shared `kv [Kv,D]`, `indices [T,topk]`.
  Returns `(out [T,H,d_v], lse [T,H], max_logits [T,H])`.
- **`flash_mla_sparse_fwd(q, kv, indices, sm_scale, attn_sink=, topk_length=)`** —
  prefill (faithful name). `kv` is the bf16 latent workspace `[Kv,1,D]`,
  `indices` is `[T,1,topk]`. Returns `(out, max_logits, lse)`.
- **`flash_mla_with_kvcache(q, k_cache, …, indices, attn_sink=, extra_k_cache=,
  extra_indices_in_kvcache=, topk_length=, extra_topk_length=, scale_cache=,
  extra_scale_cache=, block_size=)`** — decode (faithful name). Gathers +
  dequantizes the paged **fp8_ds_mla** primary (SWA) and optional compressed (CSA)
  caches, unions the selections, runs one softmax. Returns `(out, lse)`.
- **`get_mla_metadata(*args, **kwargs)`** — returns `(placeholder, num_splits=1)`.
  The compute is correct without split-KV scheduling; the V4 no-arg call works.

## fp8_ds_mla latent-KV layout (decode)

Pinned from the tokenspeed cache writer
`_deepseek_v4_fused_sparse_compress_cache_kernel`. Per latent token:

- **value region** `nope_dim + rope_dim*2` bytes: `nope_dim` (448) fp8 **e4m3**
  (`torch.float8_e4m3fn`, value-bearing) then `rope_dim` (64) **bf16** (rope,
  score-only).
- **scale region** `nope_dim//quant_block` (7) uint8 exponents `enc` + 1 pad byte,
  one per `quant_block`=64 group along nope.

Dequant: `nope = fp8_e4m3(byte) * 2**(enc - 127)` per group; `rope = bf16`.
`src/xkernels/ops/attention/sparse_mla.py` ships `dequant_fp8_ds_mla` and a
`make_fp8_ds_mla_kv` generator (exact inverse, for offline unit tests).

## Kernel strategy

One Triton program per `(token, head)`, streaming the top-k in `BLOCK_N=64` chunks
with online (flash) softmax: running max + denominator + a `[D]` fp32 accumulator
(value stored for the first `d_v` dims). The sink folds in after the stream. An
all-masked chunk is guarded so an empty-window token never NaNs. Decode gathers +
dequants to bf16 first, then reuses this one compute kernel via a flattened
`(kv, indices)`.

Deferred (kernel kept shaped for them): split-KV + `mha_merge_state` (#3) merge for
long top-k, and a fused Triton fp8 gather (decode currently gathers with torch
index + the dequant helper; the *compute* — the missing piece — is Triton).

## Validation

- Offline: `tests/test_sparse_mla_attention.py` — Triton vs oracle (GPU bf16 /
  `TRITON_INTERPRET=1` CPU fp32), sink on/off, padded lengths + `-1` sentinels,
  empty-window, single- and dual-cache decode, fp8_ds_mla round-trip.
- On-device (gfx942 / MI300A): `slurm/test_sparse_mla_beverin.sbatch`.

| Check | Shape | Result |
|-------|-------|--------|
| prefill Triton vs oracle max\|err\| | H=128, D=512, d_v=448, topk=512 | _(filled from beverin run)_ |
| decode dual-cache parity | — | _(filled from beverin run)_ |

> `d_v` default is the full latent `D`; pin to 448 against tokenspeed's o_proj
> during on-device bring-up if a real V4 layer disagrees.
