import sys; sys.path.insert(0, "src")
import torch
from xkernels.ops.moe import fused_moe_int4_w4a16, make_w4a16_weights, MoeInt4Workspace
from xkernels.ops.moe.triton.configs import get_moe_int4_config
DEV = "cuda"; dtype = torch.bfloat16
E = 8
packed, scale, _ = make_w4a16_weights(E, 256, 256, 32, device=DEV, seed=2)
A = (torch.randn(8, 256, device=DEV) * 0.1).to(dtype)
tids = torch.stack([torch.randperm(E, device=DEV)[:4] for _ in range(8)]).to(torch.int32)
tw = torch.rand(8, 4, device=DEV, dtype=torch.float32)
N = packed.shape[1]
print("config for M=8:", get_moe_int4_config(E, N, 256, 8))

ws = MoeInt4Workspace.allocate(8, 4, N, dtype=dtype, device=DEV)
# Manually do what the backend does for fused_combine=True
from xkernels.ops.moe.triton.moe_int4_kernel import moe_align_block_size_triton, int4_w4a16_moe_gemm, align_block_m
config = get_moe_int4_config(E, N, 256, 8)
block_m = config["BLOCK_SIZE_M"]
print("block_m =", block_m)
sorted_ids, expert_ids, num_post = moe_align_block_size_triton(tids, block_m, E, truncate=False)
out = ws.combine_out[:8]
out.zero_()
print("before GEMM: combine_out.sum =", ws.combine_out.sum().item())
int4_w4a16_moe_gemm(A, packed, scale, out, tw.reshape(-1).float(), sorted_ids, expert_ids, num_post,
    top_k=4, group_size=32, mul_routed_weight=True, compute_type=torch.float32, filter_expert=False, config=config, combine=True)
print("after GEMM: combine_out[:8].sum =", ws.combine_out[:8].sum().item())
print("after GEMM: out.sum (the slice) =", out.sum().item())
print("after GEMM: combine_out.abs().max =", ws.combine_out.abs().max().item())
