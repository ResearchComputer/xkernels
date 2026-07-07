import torch
from xkernels import verify_parity
from xkernels._dispatch import dispatch
from xkernels.ops.gemm import per_token_group_quant_fp8 as ref_ptg, per_block_quant_fp8 as ref_pb
for op in ["per_token_group_quant_fp8@1.0.0","per_block_quant_fp8@1.0.0"]:
    pr = verify_parity(op, archs=["nvidia_sm121"])
    print("parity", op, "agree=", pr["agree"], "max=%.2e" % pr["max_pairwise_rel_err"], pr["per_backend_runnable"])
# grouped-view triton dispatch vs the [M,K] reference, full tiles
M,K,block = 64, 256, 128
x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
G = M*(K//block)
xg = x.view(M, K//block, block).reshape(-1, block).contiguous()
qg, sg = dispatch("per_token_group_quant_fp8", x=xg, backend="triton")
qr, sr = ref_ptg(x, block=block)
q_match = torch.equal(qg.reshape(M,K).to(torch.float32), qr.to(torch.float32))
s_match = torch.allclose(sg.view(M,K//block), sr, atol=0, rtol=0)
print("grouped triton dispatch vs [M,K] reference: q_bitexact=", q_match, "scale_bitexact=", s_match, "| q", qg.dtype, qg.shape, "scale", sg.dtype, sg.shape)
