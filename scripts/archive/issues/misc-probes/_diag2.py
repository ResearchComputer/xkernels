import torch, xkernels
from xkernels import verify, verify_parity
cards = ["apply_rope.triton@1.0.0","per_token_group_quant_fp8.triton@1.0.0","per_block_quant_fp8.triton@1.0.0"]
ops = ["apply_rope@1.0.0","per_token_group_quant_fp8@1.0.0","per_block_quant_fp8@1.0.0"]
for card in cards:
    try:
        r = verify(card, arch="nvidia_sm121", measure_perf=True)
        c=r["correctness"]; p=r.get("perf") or {}
        print(card, "| compiled=", r["compiled"], "passed=", c["passed"], "max_rel=%.2e" % c["max_rel_err"], "ms=", p.get("ms"), "| err=", r.get("artifacts",{}).get("error"))
    except Exception as e:
        print(card, "RAISED", type(e).__name__, str(e)[:140])
for op in ops:
    try:
        pr = verify_parity(op, archs=["nvidia_sm121"])
        print("parity", op, "| agree=", pr["agree"], "max_pairwise=%.2e" % pr["max_pairwise_rel_err"], "runnable=", pr["per_backend_runnable"])
    except Exception as e:
        print("parity", op, "RAISED", type(e).__name__, str(e)[:140])
print("apply_rope exported:", hasattr(xkernels, "apply_rope"))
