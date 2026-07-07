import torch, xkernels
from xkernels._dispatch import _REGISTRY
print("apply_rope backends:", list(_REGISTRY.get("apply_rope", {}).keys()))
T,H,D,P = 8,8,128,64
q = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
pos = torch.randint(0, P, (T,), dtype=torch.int32, device="cuda")
csc = torch.randn(P, D, dtype=torch.float32, device="cuda")
oq, ok = xkernels.apply_rope(q, k, pos, csc)   # auto -> reference only
torch.cuda.synchronize()
print("apply_rope auto OK: shapes", tuple(oq.shape), tuple(ok.shape), "finite=", torch.isfinite(oq).all().item())
