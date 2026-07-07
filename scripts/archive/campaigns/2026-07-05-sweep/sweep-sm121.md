# Sweep — arch `nvidia_sm121`

_28 ops on dgx-spark-05.inf.ethz.ch, 2026-07-05 01:30 (seed 1729)._


Flag: ✓ correct+timed · ~ compiles/numerics-fail · ✗ no-compile/crash · · no triton card · ? load error.


| op | flag | triton ms | native ms | correct | parity | notes |
|---|:--:|--:|--:|:--:|:--:|---|
| `apply_rope@1.0.0` | ✓ | 0.0132 |  | ✓ | agree |  |
| `dual_rmsnorm@1.0.0` | ✓ | 0.0062 |  | ✓ | agree |  |
| `fused_ffn@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `gelu_and_mul@1.0.0` | ✓ | 0.0078 |  | ✓ | agree |  |
| `gemm_bf16@1.0.0` | ✓ | 0.0131 |  | ✓ | agree |  |
| `hc_prenorm_gemm@1.0.0` | ✗ | — |  | ✗ | — | KeyError: 'Keyword argument waves_per_eu was speci |
| `mha_merge_state@1.0.0` | ✓ | 0.0509 |  | ✓ | agree |  |
| `mhc_pre@1.0.0` | ✗ | — |  | ✗ | — | CompilationError: at 35:10:
    HC_MULT: tl.conste |
| `mm_fp8_blockscale@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `moe_align_block_size@1.0.0` | ✓ | 0.0284 |  | ✓ | agree |  |
| `moe_int4_w4a16@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `moe_sum_reduce@1.0.0` | ~ | — |  | ✗ | agree |  |
| `packed_gelu_and_mul@1.0.0` | ✓ | 0.0080 |  | ✓ | agree |  |
| `packed_silu_and_mul@1.0.0` | ✓ | 0.0076 |  | ✓ | agree |  |
| `paged_attention@1.0.0` | ✓ | 0.0136 |  | ✓ | agree |  |
| `paged_attention_prefill@1.0.0` | ✓ | 0.0558 |  | ✓ | agree |  |
| `paged_kv_gather@1.0.0` | ✓ | 0.0136 |  | ✓ | agree |  |
| `per_block_quant_fp8@1.0.0` | ✓ | 0.0080 |  | ✓ | agree |  |
| `per_token_group_quant_fp8@1.0.0` | ✓ | 0.0045 |  | ✓ | agree |  |
| `rmsnorm@1.0.0` | ✓ | 0.0317 |  | ✓ | agree |  |
| `rowwise_softmax@1.0.0` | ✓ | 0.0047 |  | ✓ | agree |  |
| `sampling_from_probs@1.0.0` | ✓ | 0.0041 |  | ✓ | agree |  |
| `silu_and_mul@1.0.0` | ✓ | 0.0072 |  | ✓ | agree |  |
| `sparse_mla_attention@1.0.0` | ✗ | — |  | ✗ | — | KeyError: 'Keyword argument waves_per_eu was speci |
| `temperature_softmax@1.0.0` | ✓ | 0.0072 |  | ✓ | agree |  |
| `top_k_sampling_from_probs@1.0.0` | ✓ | 0.0083 |  | ✓ | agree |  |
| `topk_softmax@1.0.0` | ✓ | 0.0042 |  | ✓ | agree |  |
| `xielu@1.0.0` | ✓ | 0.1212 |  | ✓ | agree |  |
