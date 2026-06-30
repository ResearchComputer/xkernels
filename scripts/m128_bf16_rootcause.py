#!/usr/bin/env python
"""CPU numerics probe (no GPU needed): is the bf16 M=128 verify failure a
*bf16-mantissa rounding-boundary artifact* (→ tolerance fix per
mixed-precision-convert) or a real fp32 accumulation gap (→ kernel fix)?

The reference is `a_deq @ b_deq.t()` (torch's impl-defined blocked fp32 matmul),
then `.to(bf16)`. Every other valid fp32 reduction of the same dot product
differs from torch by a tiny fp32 amount; after the bf16 cast, an element whose
fp32 value sits near a bf16 rounding boundary will JUMP by 1-2 bf16-ULP between
reductions. bf16 has 7 mantissa bits -> 2^-7 ~= 7.8e-3 per ULP, so 2 ULP ~=
1.56e-2 rel at an O(1) value -- exactly the observed failure (rel 1.36e-2).

We compute, on identical dequantized fp32 operands (seed 1729, the verify seed),
FOUR independent fp32 reductions of the same A@B.T and report each one's
fp32-vs-torch agreement AND its bf16-cast-vs-torch-bf16 agreement. If SEVERAL
reductions exceed rtol=1e-2 after the bf16 cast (while all agree to ~1e-5 in
fp32), the failure is a bf16-boundary artifact and the fix is the bf16 tolerance
-- not any one kernel.
"""
from __future__ import annotations

import torch

from xkernels.ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8

torch.manual_seed(1729)  # the verify seed
M, N, K = 128, 512, 512
a = torch.randn(M, K, dtype=torch.float32)
b = torch.randn(N, K, dtype=torch.float32)
a_fp8, a_scales = per_token_group_quant_fp8(a, block=128)
b_fp8, b_scales = per_block_quant_fp8(b, block=128)

# Bit-identical dequant to the reference (both paths dequant the same fp8 bits).
a_deq = a_fp8.to(torch.float32) * a_scales.repeat_interleave(128, dim=1)[:, :K]
b_deq = (b_fp8.to(torch.float32)
         * b_scales.repeat_interleave(128, dim=0)[:N].repeat_interleave(128, dim=1)[:, :K])

ref_fp32 = a_deq @ b_deq.t()                      # torch matmul (the reference)


def kahan_sum(a, b):
    """Sequential Kahan-compensated fp32 dot over K (mimics the CUTE kernel)."""
    acc = torch.zeros(M, N, dtype=torch.float32)
    c = torch.zeros(M, N, dtype=torch.float32)
    for k in range(K):
        y = a[:, k:k+1] * b[:, k:k+1].t() - c
        t = acc + y
        c = (t - acc) - y
        acc = t
    return acc


def naive_sum(a, b):
    """Naive sequential fp32 dot (no compensation) -- the 'bad' baseline."""
    acc = torch.zeros(M, N, dtype=torch.float32)
    for k in range(K):
        acc = acc + a[:, k:k+1] * b[:, k:k+1].t()
    return acc


def blocked_sum(a, b, chunk=32):
    """Blocked fp32 dot (reshape K into chunks, pairwise) -- a 2nd independent
    'good' reduction, like a tiled kernel would do. Different order from torch."""
    acc = torch.zeros(M, N, dtype=torch.float32)
    for k0 in range(0, K, chunk):
        acc = acc + a[:, k0:k0+chunk] @ b[:, k0:k0+chunk].t()
    return acc


def tree_sum(a, b):
    """Pairwise tree reduction over K (split K in halves recursively)."""
    # flatten K reduction via repeated halving
    partials = [a[:, k:k+1] * b[:, k:k+1].t() for k in range(K)]
    while len(partials) > 1:
        nxt = []
        for i in range(0, len(partials) - 1, 2):
            nxt.append(partials[i] + partials[i + 1])
        if len(partials) % 2:
            nxt.append(partials[-1])
        partials = nxt
    return partials[0]


def report(name, fp32_out):
    df = (fp32_out - ref_fp32).abs()
    relf = df / ref_fp32.abs().clamp_min(1e-8)
    bf = fp32_out.to(torch.bfloat16).to(torch.float32)
    rb = ref_fp32.to(torch.bfloat16).to(torch.float32)
    db = (bf - rb).abs()
    relb = db / rb.abs().clamp_min(1e-8)
    # bf16 ULP at each element
    ulp = (2.0 ** (rb.abs().clamp_min(1e-8).log2().floor() - 7))
    max_ulp = (db / ulp.clamp_min(1e-12)).max().item()
    fp32_ok = "ok" if relf.max().item() <= 1e-3 else "FAIL(fp32 rtol 1e-3)"
    bf16_ok = "ok" if relb.max().item() <= 1e-2 else "FAIL(bf16 rtol 1e-2)"
    print(f"  {name:22s} fp32 max_rel={relf.max().item():.3e} [{fp32_ok}]  "
          f"|  bf16 max_rel={relb.max().item():.3e} max_ulp={max_ulp:.1f} [{bf16_ok}]")

    print(f"  {name:22s} fp32 max_abs={df.max().item():.3e}                  "
          f"|  bf16 max_abs={db.max().item():.3e}  (atol=0.1 -> "
          f"{'ok' if db.max().item()<=0.1 else 'FAIL'})")


print(f"M={M} N={N} K={K}, seed=1729 (the verify seed), pure fp32 on CPU")
print(f"reference = torch matmul a_deq @ b_deq.t(); bf16 ULP at 1.0 ~= {2**-7:.3e}")
print()
report("kahan sequential", kahan_sum(a_deq, b_deq))
report("naive sequential", naive_sum(a_deq, b_deq))
report("blocked (chunk=32)", blocked_sum(a_deq, b_deq, 32))
report("blocked (chunk=128)", blocked_sum(a_deq, b_deq, 128))
report("tree pairwise", tree_sum(a_deq, b_deq))
print()
print("If SEVERAL reductions FAIL bf16 rtol=1e-2 while all PASS fp32 rtol=1e-3,")
print("the failure is a bf16-mantissa boundary artifact -> tolerance fix, not kernel.")
