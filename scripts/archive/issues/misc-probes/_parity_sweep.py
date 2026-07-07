import sys
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")
from xkernels import verify_parity

ops = [
    "apply_rope@1.0.0", "dual_rmsnorm@1.0.0", "fused_ffn@1.0.0",
    "gelu_and_mul@1.0.0", "gemm_bf16@1.0.0", "hc_prenorm_gemm@1.0.0",
    "mha_merge_state@1.0.0", "mhc_pre@1.0.0", "mm_fp8_blockscale@1.0.0",
    "moe_align_block_size@1.0.0", "moe_int4_w4a16@1.0.0", "moe_sum_reduce@1.0.0",
    "packed_gelu_and_mul@1.0.0", "packed_silu_and_mul@1.0.0",
    "paged_attention@1.0.0", "paged_attention_prefill@1.0.0",
    "paged_kv_gather@1.0.0", "per_block_quant_fp8@1.0.0",
    "per_token_group_quant_fp8@1.0.0", "rmsnorm@1.0.0", "rowwise_softmax@1.0.0",
    "sampling_from_probs@1.0.0", "silu_and_mul@1.0.0",
    "sparse_mla_attention@1.0.0", "temperature_softmax@1.0.0",
    "top_k_sampling_from_probs@1.0.0", "topk_softmax@1.0.0",
]
print(f"{'op':<40} {'agree':<8} {'max_rel':<12} {'runnable'}")
n_pass = n_inc = n_fail = 0
for op in ops:
    try:
        r = verify_parity(op, archs=["nvidia_sm121"])
        ag = r["agree"]
        status = "PASS" if ag is True else ("INC" if ag is None else "FAIL")
        if status == "PASS": n_pass += 1
        elif status == "INC": n_inc += 1
        else: n_fail += 1
        print(f"{op:<40} {str(ag):<8} {r.get('max_pairwise_rel_err',0):<12.4g} "
              f"{r.get('n_runnable',0)}  {status}")
    except Exception as e:
        print(f"{op:<40} ERR {type(e).__name__}: {e}")
print(f"\nPASS={n_pass}  INCONCLUSIVE={n_inc}  FAIL={n_fail}")
