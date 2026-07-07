import torch, xkernels
from xkernels import verify, verify_parity
q = torch.randn(64, 8, 128, dtype=torch.bfloat16, device="cuda")
k = torch.randn(64, 8, 128, dtype=torch.bfloat16, device="cuda")
csc = torch.randn(512, 128, dtype=torch.float32, device="cuda")
pos = torch.randint(0, 512, (64,), dtype=torch.int32, device="cuda")
qo, ko = xkernels.apply_rope(q, k, pos, csc)
print("public apply_rope: OK", tuple(qo.shape), "finite=", torch.isfinite(qo).all().item(), qo.dtype)
v = verify("apply_rope.triton@1.0.0", arch="nvidia_sm121")
print("verify via import: compiled=", v["compiled"], "passed=", v["correctness"]["passed"], "err=", v["artifacts"].get("error"))
p = verify_parity("apply_rope@1.0.0", archs=["nvidia_sm121"])
print("parity: agree=", p["agree"], "max_rel=", p["max_pairwise_rel_err"])
