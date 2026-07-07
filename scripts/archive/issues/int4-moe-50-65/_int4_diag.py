import sys; sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.moe import (
    MoeInt4Workspace, fused_moe_int4_w4a16, make_w4a16_weights,
)
DEV = "cuda"; dtype = torch.bfloat16
torch.manual_seed(0)
E = 8
def inputs(M, seed=2):
    packed, scale, _ = make_w4a16_weights(E, 256, 256, 32, device=DEV, seed=seed)
    A = (torch.randn(M, 256, device=DEV) * 0.1).to(dtype)
    tids = torch.stack([torch.randperm(E, device=DEV)[:4] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, 4, device=DEV, dtype=torch.float32)
    return A, packed, scale, tids, tw

A, packed, scale, tids, tw = inputs(8)
N = packed.shape[1]
ws = MoeInt4Workspace.allocate(8, 4, N, dtype=dtype, device=DEV)

# fused_combine=True: alloc vs workspace
o_alloc = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
o_ws = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton", workspace=ws)
diff = (o_alloc.float() - o_ws.float()).abs()
print(f"fused_combine=True: max_abs={diff.max().item():.6f}  mean_abs={diff.mean().item():.6f}")
print(f"  |o_alloc| max={o_alloc.float().abs().max().item():.4f}")

# is it atomic nondeterminism? run alloc TWICE
o_a1 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
o_a2 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
diff2 = (o_a1.float() - o_a2.float()).abs()
print(f"alloc-vs-alloc (determinism?): max_abs={diff2.max().item():.6f}")

# workspace twice
o_w1 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton", workspace=ws)
o_w2 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton", workspace=ws)
diff3 = (o_w1.float() - o_w2.float()).abs()
print(f"ws-vs-ws (determinism?): max_abs={diff3.max().item():.6f}")
