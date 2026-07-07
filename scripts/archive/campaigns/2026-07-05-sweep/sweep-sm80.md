# Sweep — arch `nvidia_sm80`

_28 ops on nid002296, 2026-07-05 03:38 (seed 1729)._


Flag: ✓ correct+timed · ~ compiles/numerics-fail · ✗ no-compile/crash · · no triton card · ? load error.


| op | flag | triton ms | native ms | correct | parity | notes |
|---|:--:|--:|--:|:--:|:--:|---|
| `apply_rope@1.0.0` | ✓ | 0.5190 |  | ✓ | agree |  |
| `dual_rmsnorm@1.0.0` | ✓ | 0.0073 |  | ✓ | agree |  |
| `fused_ffn@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `gelu_and_mul@1.0.0` | ✓ | 0.0065 |  | ✓ | agree |  |
| `gemm_bf16@1.0.0` | ✓ | 0.0229 |  | ✓ | agree |  |
| `hc_prenorm_gemm@1.0.0` | ✗ | — |  | ✗ | — | KeyError: 'Keyword argument waves_per_eu was speci |
| `mha_merge_state@1.0.0` | ✓ | 0.0151 |  | ✓ | agree |  |
| `mhc_pre@1.0.0` | ? | — |  | ✗ | DIVERGE | NOJSON rc=-11 ================================= |
| `mm_fp8_blockscale@1.0.0` | ✗ | — |  | ✗ | — | CompilationError: at 33:17:
    # N-block this til |
| `moe_align_block_size@1.0.0` | ✓ | 0.0428 |  | ✓ | agree |  |
| `moe_int4_w4a16@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `moe_sum_reduce@1.0.0` | ~ | — |  | ✗ | agree |  |
| `packed_gelu_and_mul@1.0.0` | ✓ | 0.0360 |  | ✓ | agree |  |
| `packed_silu_and_mul@1.0.0` | ✓ | 0.0273 |  | ✓ | agree |  |
| `paged_attention@1.0.0` | ✓ | 0.0170 |  | ✓ | agree |  |
| `paged_attention_prefill@1.0.0` | ✓ | 0.0467 |  | ✓ | agree |  |
| `paged_kv_gather@1.0.0` | ✓ | 0.0201 |  | ✓ | agree |  |
| `per_block_quant_fp8@1.0.0` | ✗ | — |  | ✗ | — | CompilationError: at 14:10:
    cols_B = tl.arange |
| `per_token_group_quant_fp8@1.0.0` | ✗ | — |  | ✗ | — | CompilationError: at 14:10:
    cols_B = tl.arange |
| `rmsnorm@1.0.0` | ✓ | 0.0281 |  | ✓ | agree |  |
| `rowwise_softmax@1.0.0` | ✓ | 0.0113 |  | ✓ | agree |  |
| `sampling_from_probs@1.0.0` | ✓ | 0.0072 |  | ✓ | agree |  |
| `silu_and_mul@1.0.0` | ✓ | 0.0077 |  | ✓ | agree |  |
| `sparse_mla_attention@1.0.0` | ✗ | — |  | ✗ | — | KeyError: 'Keyword argument waves_per_eu was speci |
| `temperature_softmax@1.0.0` | ✓ | 0.0294 |  | ✓ | agree |  |
| `top_k_sampling_from_probs@1.0.0` | ✓ | 0.0111 |  | ✓ | agree |  |
| `topk_softmax@1.0.0` | ✓ | 0.0060 |  | ✓ | agree |  |
| `xielu@1.0.0` | ✓ | 0.0209 |  | ✓ | agree |  |
