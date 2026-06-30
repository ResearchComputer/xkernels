#!/usr/bin/env python
"""Time breakdown of mm_fp8_blockscale_cute to find where the end-to-end ms goes.
CUDA events, steady-state (warmup then median of 50). Isolates: dequant, transpose,
the raw CUTE kernel, and the bf16 cast."""
from __future__ import annotations
import torch
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute
from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8

torch.manual_seed(1729)
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda"); b = torch.randn(N, K, device="cuda")
a8, as_ = per_token_group_quant_fp8(a, block=128); b8, bs = per_block_quant_fp8(b, block=128)

def ev(): return torch.cuda.Event(enable_timing=True)

def time(fn, n=50, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = ev(), ev()
    s.record(); 
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n   # ms per call

# full op (what verify/benchmark see)
from xkernels.ops.gemm.cute.entry import mm_fp8_blockscale_cute
print(f"FULL op (dequant+kernel+cast):     {time(lambda: mm_fp8_blockscale_cute(a8, as_, b8, bs, block=128, out_dtype=torch.bfloat16)):.4f} ms")

# precompute dequanted fp32 operands
a_deq = (a8.to(torch.float32) * as_.repeat_interleave(128, dim=1)[:, :K]).contiguous()
b_deq = (b8.to(torch.float32) * bs.repeat_interleave(128, dim=0)[:N].repeat_interleave(128, dim=1)[:, :K]).contiguous()

print(f"  raw CUTE kernel (fp32 operands): {time(lambda: fp32_matmul_cute(a_deq, b_deq)):.4f} ms")
print(f"  host transpose B.t().contig:     {time(lambda: b_deq.t().contiguous()):.4f} ms")
print(f"  dequant A (to+repeat+mul):       {time(lambda: a8.to(torch.float32) * as_.repeat_interleave(128, dim=1)[:, :K]):.4f} ms")
print(f"  torch ref matmul a_deq@b_deq.T:  {time(lambda: a_deq @ b_deq.t()):.4f} ms")
