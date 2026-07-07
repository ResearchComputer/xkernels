import sys; sys.path.insert(0, "src")
import torch
from xkernels.ops.moe import fused_moe_int4_w4a16, make_w4a16_weights
from xkernels.ops.moe.reference import moe_w4a16_ref
DEV = "cuda"; dtype = torch.bfloat16
E = 8
packed, scale, _ = make_w4a16_weights(E, 256, 256, 32, device=DEV, seed=2)
A = (torch.randn(8, 256, device=DEV) * 0.1).to(dtype)
tids = torch.stack([torch.randperm(E, device=DEV)[:4] for _ in range(8)]).to(torch.int32)
tw = torch.rand(8, 4, device=DEV, dtype=torch.float32)

ref = moe_w4a16_ref(A, packed, scale, tids, tw, mul_routed_weight=True, fused_combine=False)
o_scratch = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=False, backend="triton")
o_combine = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
print(f"ref            max={ref.float().abs().max():.4f}")
print(f"scratch vs ref max_abs={(o_scratch.float()-ref.float()).abs().max():.6f}")
print(f"combine vs ref max_abs={(o_combine.float()-ref.float()).abs().max():.6f}")
print(f"combine max={o_combine.float().abs().max():.4f}  scratch max={o_scratch.float().abs().max():.4f}")
