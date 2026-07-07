import sys; sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.moe import (
    dequant_w4a16, fused_moe_int4_w4a16, make_w4a16_weights,
    dequant_mxfp4_weight, fused_moe_mxfp4, make_mxfp4_moe_weights,
)
DEV = "cuda"
dtype = torch.bfloat16

def int4_inputs(M, E, N, K, top_k, dev, group_size=32):
    packed, scale, _ = make_w4a16_weights(E, N, K, group_size, device=dev, seed=1)
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, packed, scale, topk_ids, topk_w

def mxfp4_inputs(M, E, ispp, hidden, top_k, dev, group_size=32):
    d = make_mxfp4_moe_weights(E, hidden, ispp, group_size=group_size, device=dev, seed=1)
    A = (torch.randn(M, hidden, device=dev) * 0.1).to(dtype)
    topk_ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    topk_w = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, d["w13"], d["w13_scale"], d["w2"], d["w2_scale"], topk_ids, topk_w

print("=== §2 fused_moe_int4_w4a16 on GB10 ===")
for tag, M in [("decode M=8", 8), ("prefill M=128", 128)]:
    try:
        A, packed, scale, tids, tw = int4_inputs(M, 8, 256, 256, 4, DEV)
        o = fused_moe_int4_w4a16(A, packed, scale, tids, tw, backend="triton")
        print(f"  {tag}: triton OK out{tuple(o.shape)} {o.dtype}  (fused_combine auto)")
        o2 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=False, backend="triton")
        print(f"  {tag}: triton OK (fused_combine=False) out{tuple(o2.shape)}")
        o3 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
        print(f"  {tag}: triton OK (fused_combine=True) out{tuple(o3.shape)}")
    except Exception as ex:
        print(f"  {tag}: triton FAILS: {type(ex).__name__}: {str(ex)[:180]}")

print("=== §3 fused_moe_mxfp4 on GB10 ===")
for tag, M in [("decode M=8", 8), ("prefill M=128", 128)]:
    try:
        A, w13, w13s, w2, w2s, tids, tw = mxfp4_inputs(M, 8, 128, 256, 4, DEV)
        o = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton")
        print(f"  {tag}: triton OK out{tuple(o.shape)} {o.dtype}")
    except Exception as ex:
        print(f"  {tag}: triton FAILS: {type(ex).__name__}: {str(ex)[:180]}")
