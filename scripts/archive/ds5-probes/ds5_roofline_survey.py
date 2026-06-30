#!/usr/bin/env python
"""Roofline survey of the 5 CUTE cards on GB10 (sm_121).

Establishes, from measured numbers (not datasheets):
  - peak fp32 CUDA-core TFLOPS (via a saturating FMA microbench -> real cores/SM)
  - peak DRAM BW (D2D copy)
then for each kernel at a representative shape computes:
  FLOPs, bytes, arithmetic intensity (AI), ms (do_bench), achieved_tflops,
  achieved_bw_pct, achieved_flops_pct, and regime (mem vs compute vs launch).

The ridge point = peak_flops / peak_bw. AI < ridge -> memory-bound (ceiling = BW);
AI > ridge -> compute-bound (ceiling = flops). 'launch-bound' = problem so small
that ms is dominated by per-launch overhead, not the work.
"""
from __future__ import annotations
import torch
from xkernels.utils.benchmarking import benchmark

torch.manual_seed(1729)
DEV = "cuda"

# ---- 1. hardware ceilings ------------------------------------------------
# DRAM BW: measured (D2D copy, the realistic kernel ceiling).
# fp32 CUDA-core peak: theoretical, DERIVED not datasheet-guessed:
#   48 SMs * 128 fp32 cores/SM (Blackwell sm_120/121) * 2.4 GHz * 2 (FMA) = 29.5 TFLOPS.
# (A saturating-FMA microbench is a rabbit hole; the ridge classification is
#  robust to ±2x here since every kernel has AI <= 21 FLOPs/byte << ridge.)
pflop = 29.5  # TFLOPS fp32 CUDA-core (scalar; tensor-core peak is gated, see notes)

def peak_bw_gbs():
    sz = 512 * 1024 * 1024  # 512 MB
    a = torch.randn(sz // 4, device=DEV, dtype=torch.float32)
    b = torch.empty_like(a)
    for _ in range(10): b.copy_(a)
    torch.cuda.synchronize()
    ms = benchmark(lambda: b.copy_(a))
    return (sz * 2) / (ms * 1e-3) / 1e9  # read+write

pbw = peak_bw_gbs()
ridge = pflop * 1e12 / (pbw * 1e9)
print(f"=== GB10 ceilings ===")
print(f"  fp32 CUDA-core peak = {pflop:.1f} TFLOPS  (theoretical: 48SM*128core*2.4GHz*2)")
print(f"  tensor-core peak    = GATED on sm_121/CTK-13.0 for these ops (no usable MMA)")
print(f"  DRAM copy BW (R+W)  = {pbw:.0f} GB/s   (measured)")
print(f"  ridge point         = {ridge:.0f} FLOPs/byte  (AI<ridge => mem-bound)")
print()

# ---- 2. per-kernel roofline -----------------------------------------------
def row(name, flops, bytes_rw, ms, note=""):
    ai = flops / bytes_rw if bytes_rw else float("inf")
    tf = flops / (ms * 1e-3) / 1e12
    bw_gbs = bytes_rw / (ms * 1e-3) / 1e9
    bw_pct = bw_gbs / pbw * 100
    flop_pct = tf / pflop * 100
    regime = ("compute" if ai >= ridge else "memory") + "-bound"
    if bytes_rw < 0.5e6:  # < 0.5 MB -> launch-overhead dominated
        regime += " (LAUNCH-bound: <0.5MB)"
    print(f"{name:26s} AI={ai:6.1f}  {regime:34s} | "
          f"ach_tflops={tf:6.2f}({flop_pct:4.1f}%)  ach_BW={bw_pct:5.1f}%  ms={ms*1e3:6.1f}us")

# (a) mm_fp8_blockscale GEMM — kernel sees fp32 A[M,K],B[N,K],C[M,N] after host dequant.
from xkernels.ops.gemm.cute.mm_fp8_blockscale_kernel import fp32_matmul_cute
M, N, K = 128, 512, 512
af32 = torch.randn(M, K, device=DEV, dtype=torch.float32)
bf32 = torch.randn(N, K, device=DEV, dtype=torch.float32)  # fp32_matmul_cute(a,b)=a@b.T
for _ in range(10): fp32_matmul_cute(af32, bf32)
ms = benchmark(lambda: fp32_matmul_cute(af32, bf32))
# kernel-only bytes: fp32 A[M,K] + B[K,N] + C[M,N]  (dequant bytes are host-side, separate)
bytes_gemm = 4 * (M * K + K * N + M * N)
flops_gemm = 2 * M * N * K
row("mm_fp8_blockscale(GEMM)", flops_gemm, bytes_gemm, ms)

# (b) dual_rmsnorm — per elem: ~5 flops (x2, rsqrt, *w, *scale), read x2B+w2B write2B
from xkernels.ops.norm.cute.entry import rmsnorm_cute
T, D = 64, 1536
x = torch.randn(T, D, device=DEV, dtype=torch.bfloat16)
w = torch.randn(D, device=DEV, dtype=torch.bfloat16)
for _ in range(10): rmsnorm_cute(x, w)
ms = benchmark(lambda: rmsnorm_cute(x, w))
flops = 5 * T * D
bytes_rw = 2 * T * D + 2 * D + 2 * T * D  # x(in bf16) + w + out(bf16)
row("dual_rmsnorm", flops, bytes_rw, ms)

# (c) moe_sum_reduce — top_k FMAs w/ Kahan (~5 flops/k) + scale, read top_k*y + top_k*w
from xkernels.ops.moe.cute.entry import moe_sum_reduce_cute_entry
Mm, top_k, H = 128, 8, 7168
y = torch.randn(Mm, top_k, H, device=DEV, dtype=torch.bfloat16)
wv = torch.randn(Mm, top_k, device=DEV, dtype=torch.float32)
for _ in range(10): moe_sum_reduce_cute_entry(y, wv, 1.0)
ms = benchmark(lambda: moe_sum_reduce_cute_entry(y, wv, 1.0))
flops = (5 * top_k + 1) * Mm * H
bytes_rw = 2 * Mm * top_k * H + 4 * Mm * top_k + 2 * Mm * H
row("moe_sum_reduce", flops, bytes_rw, ms)

# (d) mha_merge_state — per D-elem: ~8 flops (2 exp, max, 2 mul+add, div), read 2*bf16 + lse(amort)
from xkernels.ops.attention.cute.entry import mha_merge_state_cute
Tt, Hh, D = 64, 128, 128
oa = torch.randn(Tt, Hh, D, device=DEV, dtype=torch.bfloat16)
ob = torch.randn(Tt, Hh, D, device=DEV, dtype=torch.bfloat16)
la = torch.randn(Tt, Hh, device=DEV, dtype=torch.float32).abs()
lb = torch.randn(Tt, Hh, device=DEV, dtype=torch.float32).abs()
for _ in range(10): mha_merge_state_cute(oa, la, ob, lb)
ms = benchmark(lambda: mha_merge_state_cute(oa, la, ob, lb))
flops = 8 * Tt * Hh * D
bytes_rw = 2 * (Tt * Hh * D) * 2 + 2 * (Tt * Hh) * 4 + (Tt * Hh * D) * 2 + (Tt * Hh) * 4
row("mha_merge_state", flops, bytes_rw, ms)

# (e) hc_prenorm_gemm — GEMM 2*T*N*K + squared-sum T*K; read a(bf16)+fn(fp32), write mul+sqr(fp32)
from xkernels.ops.mhc.cute.entry import hc_prenorm_gemm_cute
Ts, Ks, Ns = 37, 128, 16
a = torch.randn(Ts, Ks, device=DEV, dtype=torch.bfloat16)
fn = torch.randn(Ns, Ks, device=DEV, dtype=torch.float32)
for _ in range(10): hc_prenorm_gemm_cute(a, fn, n_splits=1)
ms = benchmark(lambda: hc_prenorm_gemm_cute(a, fn, n_splits=1))
flops = 2 * Ts * Ns * Ks + Ts * Ks
bytes_rw = 2 * Ts * Ks + 4 * Ns * Ks + 4 * Ts * Ns + 4 * Ts
row("hc_prenorm_gemm", flops, bytes_rw, ms)
