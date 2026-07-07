# Sweep — arch `amd_cdna3`

_28 ops on nid003018, 2026-07-05 03:38 (seed 1729)._


Flag: ✓ correct+timed · ~ compiles/numerics-fail · ✗ no-compile/crash · · no triton card · ? load error.


| op | flag | triton ms | native ms | correct | parity | notes |
|---|:--:|--:|--:|:--:|:--:|---|
| `apply_rope@1.0.0` | ✓ | 0.7632 |  | ✓ | agree |  |
| `dual_rmsnorm@1.0.0` | ✓ | 0.0066 |  | ✓ | agree |  |
| `fused_ffn@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `gelu_and_mul@1.0.0` | ✓ | 0.0103 |  | ✓ | agree |  |
| `gemm_bf16@1.0.0` | ✓ | 0.0141 |  | ✓ | agree |  |
| `hc_prenorm_gemm@1.0.0` | ~ | — |  | ✗ | DIVERGE |  |
| `mha_merge_state@1.0.0` | ✓ | 0.0115 |  | ✓ | agree |  |
| `mhc_pre@1.0.0` | ✗ | — |  | ✗ | — | CompilationError: at 35:10:
    HC_MULT: tl.conste |
| `mm_fp8_blockscale@1.0.0` | ✓ | 0.0081 |  | ✓ | agree |  |
| `moe_align_block_size@1.0.0` | ✓ | 0.0528 |  | ✓ | agree |  |
| `moe_int4_w4a16@1.0.0` | ✗ | — |  | ✗ | — | OutOfResources: out of resource: shared memory, Re |
| `moe_sum_reduce@1.0.0` | ~ | — |  | ✗ | agree |  |
| `packed_gelu_and_mul@1.0.0` | ✓ | 0.0581 |  | ✓ | agree |  |
| `packed_silu_and_mul@1.0.0` | ✓ | 0.0447 |  | ✓ | agree |  |
| `paged_attention@1.0.0` | ✓ | 0.0089 |  | ✓ | agree |  |
| `paged_attention_prefill@1.0.0` | ✓ | 0.0261 |  | ✓ | agree |  |
| `paged_kv_gather@1.0.0` | ✓ | 0.0067 |  | ✓ | agree |  |
| `per_block_quant_fp8@1.0.0` | ✓ | 0.0713 |  | ✓ | agree |  |
| `per_token_group_quant_fp8@1.0.0` | ✓ | 0.2470 |  | ✓ | agree |  |
| `rmsnorm@1.0.0` | ✓ | 0.0528 |  | ✓ | agree |  |
| `rowwise_softmax@1.0.0` | ✓ | 0.0081 |  | ✓ | agree |  |
| `sampling_from_probs@1.0.0` | ✓ | 0.0067 |  | ✓ | agree |  |
| `silu_and_mul@1.0.0` | ✓ | 0.2072 |  | ✓ | agree |  |
| `sparse_mla_attention@1.0.0` | ✓ | 0.0091 |  | ✓ | agree |  |
| `temperature_softmax@1.0.0` | ✓ | 0.0421 |  | ✓ | agree |  |
| `top_k_sampling_from_probs@1.0.0` | ✓ | 0.0118 |  | ✓ | agree |  |
| `topk_softmax@1.0.0` | ✓ | 0.0059 |  | ✓ | agree |  |
| `xielu@1.0.0` | ✓ | 0.0171 |  | ✓ | agree |  |
