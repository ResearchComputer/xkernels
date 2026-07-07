# Standalone seeded repro of the apply_rope device crash (diagnose-wrong-results §1).
import torch
from xkernels._dispatch import dispatch
from xkernels.vkl import register_dsl, spec_of
from xkernels.vkl.examples import rope
register_dsl(spec_of(rope.apply_rope), "triton")
T,H,D,P = 8,8,128,64
q = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
pos = torch.randint(0, P, (T,), dtype=torch.int32, device="cuda")
csc = torch.randn(P, D, dtype=torch.float32, device="cuda")
print("launching apply_rope triton...")
oq, ok = dispatch("apply_rope", query=q, key=k, positions=pos, cos_sin_cache=csc, backend="triton")
torch.cuda.synchronize()
print("OK", tuple(oq.shape), tuple(ok.shape))
