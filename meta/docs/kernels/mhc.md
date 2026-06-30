# MHC — Multi-Head hidden-Compression (DeepSeek-V4) on gfx942

The **MHC** (hidden-Compression) layer is the last NVIDIA-only-gated piece of
DeepSeek-V4's forward path. Upstream `mhc_pre` / `mhc_post` live in a TileLang
fusion and a `deep_gemm` GEMM (`deep_gemm.tf32_hc_prenorm_gemm`); on gfx942 the
TileLang kernel **mislowers** the `layer_input` combine branch (~97% wrong) and
`deep_gemm` is NVIDIA-only. These are portable Triton replacements, each
validated on-device:

- **`hc_prenorm_gemm`** — the GEMM + RMS-prenorm-squared-sum half of `mhc_pre`
  (issue #36).
- **`mhc_pre` / `mhc_post`** — the full prenorm/postnorm fusions (issue #44).
- **The V4 perf pass** — launch-knob tuning that made them fast (issue #39).

> **One-line summary.** Replacing this one `deep_gemm` call + the mislowering
> TileLang fusions with portable Triton removes the last NVIDIA-only barrier and
> makes the complete MHC layer runnable on gfx942. The prenorm GEMM wins
> **1.48–1.63×** from a clean, uniform launch-config tuning; `mhc_pre`+`mhc_post`
> win **35.5×** vs the torch oracle.

---

## `hc_prenorm_gemm` — GEMM + RMS-prenorm squared-sum (issue #36)

Replaces `deep_gemm.tf32_hc_prenorm_gemm`. For drop-in compatibility it is
re-exported under the upstream-faithful name `tf32_hc_prenorm_gemm`.

### The math

Given a token batch reshaped to `A [T, K]` (bf16, from `residual.view(T, K)`) and
the combined hidden-compression weight matrix `fn [N, K]` (fp32, Linear
orientation), the operation computes two quantities simultaneously:

```
gemm_out    = F.linear(A, fn)   # A @ fn.T,  shape [T, N]
sqrsum_out  = Σ_k A[t, k]²      # per-row sum-of-squares,  shape [T]
```

Both are computed in fp32 (V4-Flash CDNA3 has no TF32 acceleration). **V4-Flash
dimensions** (`hc_mult=4`, `hidden=4096`): `K=16384` (`hc_mult×hidden`),
`N=24` (`2*hc_mult + hc_mult²`). This is a **memory-bound tall-skinny GEMM**
(huge K, tiny N); the dominant cost is streaming K, so fusing `Σ A²` on the same A
loads is essentially free.

### Split-K layout and the summed invariant

To exploit occupancy for small decode batch sizes (T ≪ 1024), K is partitioned
into `n_splits` contiguous blocks. Each Triton program handles one `(split_idx,
row_tile)` pair. The two outputs carry a **split axis in position 0**:

- `gemm_out_mul  [n_splits, T, N]` — partial GEMM accumulations
- `gemm_out_sqrsum [n_splits, T]` — partial squared-sum accumulations

**Key invariant:** the downstream post-fusion only ever sums across the split
axis, and summing recovers the full results — so the K-partition is numerically
free (any assignment of K indices to splits produces the same final values).
Split-K exists purely for occupancy. Empty splits (when `n_splits > ⌈K/BLOCK_K⌉`)
store explicit zeros; the invariant still holds.

### Kernel strategy

One Triton program per `(split_idx, row_tile)`:
1. **K-range selection** — each program computes its contiguous K slice; an empty
   slice stores zeros and exits.
2. **Fused load and compute** — the inner loop tiles the K range in `BLOCK_K`
   steps: load `A` tile (bf16→fp32) and a **transposed** `fn` tile `[BLOCK_K,
   BLOCK_N]` (fp32, K on axis 0 since `fn` is `[N,K]`). A single `tl.dot`
   accumulates `A @ fnᵀ`; `(A_tile * A_tile).sum(-1)` accumulates the squared-sum
   — both from the **same** A load, no extra memory traffic.
3. **Store** — partial GEMM to `gemm_out_mul[split_idx, row, :]`, partial
   squared-sum to `gemm_out_sqrsum[split_idx, row]`.

### Validation (on-device gfx942, job 383345)

| Check | Shape | Result |
|-------|-------|--------|
| `pytest tests/test_mhc_prenorm_gemm.py` | — | **15 passed** |
| sum invariant (bf16 A → fp32) | T=8, K=16384, N=24, splits=16 | **mul max\|err\|=1.53e-01, rel=3.77e-04; sqrsum max\|err\|=1.95e-03** |
| benchmark vs `F.linear`+sqsum | T=1/8/64 | **0.022 ms** vs 2.66 ms → **~120×** |

The `mul` absolute error (1.5e-1) is large only because the GEMM accumulates
K=16384 terms (output magnitudes ~O(√K)); the **relative** error 3.77e-04 is the
expected fp32-accumulation-order difference. The ~120× reflects both the kernel's
efficiency **and** that the naive `F.linear(a.float(), fn.float())` fp32 baseline
hits the slow dense-GEMM stack cliff (`kernels/gemm.md` — issue #17). The Triton
time is flat across T=1/8/64 (launch/bandwidth-bound at tiny N=24).

---

## `mhc_pre` / `mhc_post` — the full fusions (issue #44)

Issue #36 shipped `hc_prenorm_gemm`. The remaining `mhc_pre` math (prenorm
projection split into `pre`/`post`/`comb` heads, sinkhorn normalization, and the
`pre`-weighted residual combine that produces `layer_input`) and all of `mhc_post`
lived in a TileLang fusion that **mislowers the `layer_input` combine branch** on
gfx942. These ship portable Triton replacements, numerically identical to the
torch oracle.

### The math

Given `residual [T, hc_mult, hidden]` and hidden-compression weight
`fn [hc_mult3, hc_mult*hidden]` (Linear orientation, `hc_mult3 = 2*hc_mult + hc_mult²`):

**`mhc_pre`**
```
x       = residual[t].flatten()                        # [hc_mult*hidden]
rsqrt   = rsqrt(mean(x**2) + rms_eps)
mixes   = (x @ fn.T) * rsqrt                            # [hc_mult3]
pre     = sigmoid(mixes_pre  * hc_scale[0] + base) + hc_eps      # [hc_mult]
post    = sigmoid(mixes_post * hc_scale[1] + base) * 2.0         # [hc_mult]
comb    = sinkhorn(mixes_comb * hc_scale[2] + base, iters, hc_eps) # [hc_mult,hc_mult]
layer_input[t, h] = sum_n pre[n] * residual[t, n, h]
```
Returns `(layer_input [T, hidden], post [T, hc_mult, 1], comb [T, hc_mult, hc_mult])`.

**`mhc_post`**
```
out[t, m, h] = sum_n comb[t, n, m] * residual[t, n, h] + post[t, m] * hidden[t, h]
```
Returns `[T, hc_mult, hidden]`.

All math in fp32 (CDNA3 has no TF32); parity target is the pure-torch reference,
not NVIDIA bit-equality. The sinkhorn normalization matches the TileLang `comb`
branch semantics: softmax over rows, then alternating column/row normalization
for `sinkhorn_iters` total passes.

### API

```python
layer_input, post, comb = mhc_pre(
    residual,      # [T, hc_mult, hidden] bf16 (or fp32)
    fn,            # [hc_mult3, hc_mult*hidden] fp32
    hc_scale,      # [3] fp32 — scales for pre/post/comb heads
    hc_base,       # [hc_mult3] fp32 — per-channel bias
    rms_eps,       # RMS-prenorm epsilon
    hc_eps,        # sigmoid/sinkhorn stabilizing epsilon
    sinkhorn_iters,# int >= 1
    backend="auto",
)

out = mhc_post(hidden_states, residual, post, comb, backend="auto")
```

### Performance

End-to-end (T=8, hc_mult=4, hidden=4096): **0.08 ms, 35.5× / 3.9×** (MI300A /
MI250X). The large MI300A speedup comes from fusing the RMS prenorm squared-sum,
sigmoid gating, sinkhorn combination, and the post residual combine into a few
Triton kernels. On MI250X the baseline is already fast, so the speedup is ~4×.

---

## The V4 perf pass — launch-knob tuning (issue #39)

Both V4 kernels landed correctness-first. This perf pass turned their launch
parameters (block sizes + CDNA3 lowering knobs `waves_per_eu` /
`matrix_instr_nonkdim` / `kpack`) into a small, env-overridable config and
characterized the candidate space on real gfx942.

### MHC prenorm GEMM — a clean, uniform win

`BLOCK_M=32, BLOCK_K=128, waves_per_eu=4` is the fastest config at **every**
decode batch size — **promoted to the default**:

| T | baseline #36 (ms) | best #39 (ms) | speedup | best cfg | rel_err |
|--:|---:|---:|---:|---|---:|
| 1  | 0.0184 | **0.0117** | **1.57×** | BM=32 BK=128 wpe=4 | 5.0e-04 |
| 8  | 0.0191 | **0.0129** | **1.48×** | BM=32 BK=128 wpe=4 | 3.9e-04 |
| 64 | 0.0213 | **0.0130** | **1.63×** | BM=32 BK=128 (BK=256 ties) | 4.0e-04 |

Why the winner wins: the problem is memory-bound (stream A once over K=16384,
tiny N=24). The smaller `BLOCK_M=32` packs the tiny-T rows tighter and frees
VGPRs, letting `waves_per_eu=4` raise occupancy to hide the K-stream global-load
latency; `BLOCK_K=128` doubles the per-load A/fn read width. The #36 launch left
all of this on the table (default occupancy, no AMD knobs, BLOCK_K=64).

**LDS limit found on-device.** `BLOCK_K=256` fp32 at `num_stages=2` needs 96 KB
and raises `OutOfResources(98304, 65536)` — CDNA3 has 64 KB LDS/CU. The 256-wide
candidates are pinned to `num_stages=1` (48 KB); the sweep/tests treat
`OutOfResources` as "config infeasible here", not a failure.

### Sparse-MLA — no static winner

Covered in `kernels/attention.md`: `BLOCK_N=64` stays the default (no multi-token
regression); the Tq=1 win ships opt-in.

### What ships

- `ops/mhc/triton/configs.py` — `DEFAULT_MHC_GEMM_CONFIG` promoted to the measured
  winner; `BASELINE_MHC_GEMM_CONFIG` retains the #36 launch for A/B. The wrapper
  threads the AMD lowering knobs.
- Both kernels read `XKERNELS_MHC_GEMM_CONFIG` / `XKERNELS_SPARSE_MLA_CONFIG`
  (JSON dict, partial override allowed) so a deployment can pin a regime-specific
  config without code changes. The AMD knobs are ignored by non-AMD Triton and
  under `TRITON_INTERPRET=1`, so everything stays portable.

---

## Reproduce

```bash
# prenorm GEMM correctness + the perf sweep
scripts/cluster.sh run --host beverin python3 -u tests/test_mhc_prenorm_gemm.py
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/tune_mhc_prenorm_gemm.py  # #39
# pre/post fusion
scripts/cluster.sh run --host beverin python3 -u tests/test_mhc_pre_post.py
# the invariance tests for the perf knobs
scripts/cluster.sh run --host beverin python3 -u tests/test_issue39_perf_pass.py
```
