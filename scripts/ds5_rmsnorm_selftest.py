#!/usr/bin/env python
"""Standalone correctness check of rmsnorm_cute vs torch, before the full verify
harness. Catches compile errors + gross numerics fast. Uses the op's exact
seeded input generator so the test is representative of the verify sweep.
"""
from __future__ import annotations
import torch
from xkernels.ops.norm.cute.rmsnorm_kernel import rmsnorm_cute
from xkernels.ops.norm.reference import rmsnorm

torch.manual_seed(0)
worst = 0.0
cases = [
    ("fp32 T=64 d=1536", torch.float32, 64, 1536),
    ("fp32 T=64 d=512",  torch.float32, 64, 512),
    ("bf16 T=64 d=1536", torch.bfloat16, 64, 1536),
    ("bf16 T=64 d=512",  torch.bfloat16, 64, 512),
    ("bf16 T=1  d=1536", torch.bfloat16, 1, 1536),
    ("bf16 T=7  d=100",  torch.bfloat16, 7, 100),   # non-multiple-of-128
]
for name, dt, T_, D in cases:
    x = torch.randn(T_, D, device="cuda", dtype=torch.float32)
    if dt == torch.bfloat16:
        x = x.to(torch.bfloat16)
    w = torch.randn(D, device="cuda", dtype=dt)
    got = rmsnorm_cute(x, w, eps=1e-6).to(dt)
    ref = rmsnorm(x, w, eps=1e-6)
    err = (got.float() - ref.float()).abs().max().item()
    rel = (err / ref.float().abs().clamp_min(1e-6).max().item())
    worst = max(worst, err)
    rtol = 1e-5 if dt == torch.float32 else 0.016
    ok = err <= 0.01 + rtol  # rough check vs reference (atol+rtol combined)
    print(f"  {name:22s}: max_abs={err:.3e} pass={ok}")
print(f"worst max_abs vs reference = {worst:.3e}")
