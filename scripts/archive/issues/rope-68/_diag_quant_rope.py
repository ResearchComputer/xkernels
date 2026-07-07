import xkernels
from xkernels import verify
for card in ["apply_rope.triton@1.0.0","per_token_group_quant_fp8.triton@1.0.0","per_block_quant_fp8.triton@1.0.0"]:
    try:
        r = verify(card, arch="nvidia_sm121")
        c=r["correctness"]; a=r.get("artifacts",{})
        print(card, "compiled=", r["compiled"], "passed=", c["passed"], "max_rel=", c["max_rel_err"], "err=", a.get("error"))
    except Exception as e:
        print(card, "RAISED", type(e).__name__, str(e)[:160])
print("apply_rope exported:", hasattr(xkernels, "apply_rope"))
print("per_token_group_quant_fp8 exported:", hasattr(xkernels, "per_token_group_quant_fp8"))
print("per_block_quant_fp8 exported:", hasattr(xkernels, "per_block_quant_fp8"))
