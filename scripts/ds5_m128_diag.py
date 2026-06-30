#!/usr/bin/env python
"""Diagnostic: fp32-vs-fp32 agreement at the marginal M=128 bf16 point, to decide
whether the verify failure is (a) reducible fp32 accumulation divergence or
(b) pure bf16 rounding on top of already-tight fp32 agreement.

Compares the CUTE kernel's raw fp32 output (pre-bf16-cast) to torch's fp32
matmul on identical dequantized operands, at M=128 N=512 K=512, seed=1729 (the
verify seed), and reports the worst element + how many bf16-ULP it represents.
"""
from __future__ import annotations

import torch

from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute
from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8

torch.manual_seed(1729)  # the verify seed
M, N, K = 128, 512, 512
a = torch.randn(M, K, device="cuda", dtype=torch.float32)
b = torch.randn(N, K, device="cuda", dtype=torch.float32)
a_fp8, a_scales = per_token_group_quant_fp8(a, block=128)
b_fp8, b_scales = per_block_quant_fp8(b, block=128)

# Identical dequant (both paths) -> isolate the matmul.
a_deq = a_fp8.to(torch.float32) * a_scales.repeat_interleave(128, dim=1)[:, :K]
b_deq = b_fp8.to(torch.float32) * b_scales.repeat_interleave(128, dim=0)[:N].repeat_interleave(128, dim=1)[:, :K]

got_fp32 = fp32_matmul_cute(a_deq, b_deq)            # my kernel, fp32
ref_fp32 = a_deq @ b_deq.t()                          # torch, fp32

diff = (got_fp32 - ref_fp32).abs()
rel = diff / ref_fp32.abs().clamp_min(1e-6)
print(f"M={M} N={N} K={K} (fp32 vs fp32, pre-bf16):")
print(f"  max_abs = {diff.max().item():.4e}")
print(f"  max_rel = {rel.max().item():.4e}")
flat = diff.view(-1)
worst = int(flat.argmax().item())
ref_val = ref_fp32.view(-1)[worst].item()
got_val = got_fp32.view(-1)[worst].item()
# bf16 ULP at the reference value
ulp = 2.0 ** (max(0, int(torch.tensor(ref_val).abs().log2())) - 7)
print(f"  worst element: ref={ref_val:.6f} got={got_val:.6f} "
      f"|diff|={abs(got_val-ref_val):.4e} (~{abs(got_val-ref_val)/ulp:.1f} bf16-ULP)")

# Now after bf16 cast (what verify sees):
got_bf = got_fp32.to(torch.bfloat16).to(torch.float32)
ref_bf = ref_fp32.to(torch.bfloat16).to(torch.float32)
dbf = (got_bf - ref_bf).abs()
rbf = dbf / ref_bf.abs().clamp_min(1e-6)
print(f"\nafter bf16 cast (what verify checks, rtol=1e-2 atol=1e-1):")
print(f"  max_abs = {dbf.max().item():.4e}  (atol=0.1 -> {'ok' if dbf.max().item()<=0.1 else 'FAIL'})")
print(f"  max_rel = {rbf.max().item():.4e}  (rtol=0.01 -> {'ok' if rbf.max().item()<=0.01 else 'FAIL'})")
n_ulp_diff = (dbf / (2.0 ** (ref_bf.abs().clamp_min(1e-6).log2().floor() - 7))).max().item()
print(f"  worst bf16-ULP difference = {n_ulp_diff:.1f}")
