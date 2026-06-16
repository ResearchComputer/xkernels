# Issue #44 — DeepSeek-V4 MHC full `mhc_pre` / `mhc_post` fusions (gfx942)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2,
`tokenspeed_triton`. **Test:** `tests/test_mhc_pre_post.py`
(`slurm/test_mhc_pre_post_beverin.sbatch`).

## TL;DR

Issue #36 shipped `hc_prenorm_gemm`, the GEMM + RMS-prenorm-squared-sum half of
DeepSeek-V4's `mhc_pre`. The remaining `mhc_pre` math (prenorm projection split
into `pre`/`post`/`comb` heads, sinkhorn normalization, and the `pre`-weighted
residual combine that produces `layer_input`) and all of `mhc_post` lived in a
TileLang fusion. On gfx942 that TileLang kernel **mislowers the `layer_input`
combine branch** (~97% wrong → incoherent generation).

This ships portable Triton replacements for the full fusions:

- `xkernels.mhc_pre(...)` — replaces the `mhc_pre` TileLang fusion.
- `xkernels.mhc_post(...)` — replaces the `mhc_post` TileLang fusion.

Both are numerically identical to the torch oracle and validated on MI300A.

## The math

Given residual `[T, hc_mult, hidden]` and hidden-compression weight
`fn [hc_mult3, hc_mult*hidden]` (Linear orientation,
`hc_mult3 = 2*hc_mult + hc_mult²`):

### `mhc_pre`

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

### `mhc_post`

```
out[t, m, h] = sum_n comb[t, n, m] * residual[t, n, h] + post[t, m] * hidden[t, h]
```

Returns `[T, hc_mult, hidden]`.

All math is performed in fp32 (CDNA3 has no TF32); the parity target is the
pure-torch reference, not NVIDIA bit-equality.

## API

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

out = mhc_post(
    hidden_states, # [T, hidden] bf16 (or fp32)
    residual,      # [T, hc_mult, hidden]
    post,          # [T, hc_mult, 1] (or [T, hc_mult])
    comb,          # [T, hc_mult, hc_mult]
    backend="auto",
)
```

Both ops dispatch across the pure-torch reference (default on CPU / no Triton)
and a Triton backend on GPU.

## What ships

- `src/xkernels/ops/mhc/pre_post_reference.py` — pure-torch oracle and
  `Backend.REFERENCE` registration for `mhc_pre` / `mhc_post`.
- `src/xkernels/ops/mhc/triton/pre_post_kernel.py` — gfx942 Triton backend.
- `src/xkernels/ops/mhc/interface.py` — public `mhc_pre` / `mhc_post` ops.
- `tests/test_mhc_pre_post.py` — parity tests (reference vs oracle, Triton vs
  oracle, top-level exports).
- `benchmarks/bench_mhc_pre_post.py` (if present) or the ops are exercised
  through the MHC pipeline benchmarks.

## Validation

| Check | Where | Result |
|-------|-------|--------|
| Reference backend == independent torch oracle | CPU / GPU | **PASS** |
| Triton backend == reference, multiple `hc_mult`/`hidden` | CPU `TRITON_INTERPRET=1` + GPU | **PASS** |
| Top-level exports (`xkernels.mhc_pre`, `xkernels.mhc_post`) | `tests/test_mhc_pre_post.py` | **PASS** |
| On-device gfx942 correctness | beverin MI300A | **PASS** |

## Notes / scope

- This completes the MHC layer bring-up on gfx942 after `hc_prenorm_gemm` (#36).
- The tokenspeed-side binding (wiring the AMD path to call these ops instead of
  the TileLang fusion) is a tokenspeed change and out of scope for this repo.
- The sinkhorn normalization matches the TileLang `comb` branch semantics:
  softmax over rows, then alternating column/row normalization for
  `sinkhorn_iters` total passes.
