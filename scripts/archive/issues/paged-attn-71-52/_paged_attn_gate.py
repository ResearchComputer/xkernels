import torch
import xkernels
from xkernels import verify, verify_parity

print("=== reference card (CPU-style oracle) ===")
v = verify("paged_attention.reference@1.0.0", arch="any")
c = v["correctness"]
print("reference compiled=", v["compiled"], "passed=", c["passed"],
      "max_abs=", c["max_abs_err"], "max_rel=", c["max_rel_err"], "n=", c["n_points"])
if c["failing_shapes"]:
    print("   FAIL:", c["failing_shapes"][0])

print("=== triton card (GB10/sm_121) ===")
v = verify("paged_attention.triton@1.0.0", arch="nvidia_sm121")
c = v["correctness"]
print("triton compiled=", v["compiled"], "passed=", c["passed"],
      "max_abs=", c["max_abs_err"], "max_rel=", c["max_rel_err"], "n=", c["n_points"])
if not v["compiled"]:
    print("   err:", repr(v["artifacts"].get("error"))[:600])
if c["failing_shapes"]:
    print("   FAIL:", c["failing_shapes"][0])

print("=== parity (reference vs triton, GPU) ===")
p = verify_parity("paged_attention@1.0.0", archs=["nvidia_sm121"])
print("agree=", p.get("agree"), "max_rel=", p.get("max_pairwise_rel_err"),
      "runnable=", p.get("per_backend_runnable"))

print("=== direct triton vs reference, Qwen3-4B shape ===")
B, Hq, Hkv, D, bs, msl = 8, 32, 8, 128, 1, 256
maxb = (msl + bs - 1) // bs
g = torch.Generator(device="cuda").manual_seed(0)
q = torch.randn(B, Hq, D, generator=g, device="cuda", dtype=torch.bfloat16)
kc = torch.randn(B*maxb, bs, Hkv, D, generator=g, device="cuda", dtype=torch.bfloat16)
vc = torch.randn(B*maxb, bs, Hkv, D, generator=g, device="cuda", dtype=torch.bfloat16)
bt = torch.arange(B*maxb, device="cuda", dtype=torch.int32).reshape(B, maxb)
sl = torch.randint(1, msl+1, (B,), generator=g, device="cuda", dtype=torch.int32)
kt = xkernels.paged_attention(q, kc, vc, bt, sl, scale=D**-0.5, backend="triton")
ref = xkernels.paged_attention(q, kc, vc, bt, sl, scale=D**-0.5, backend="reference")
print("max_abs=", (kt.float()-ref.float()).abs().max().item(),
      "max_rel=", ((kt.float()-ref.float()).abs()/ref.float().abs().clamp_min(1e-6)).max().item())
