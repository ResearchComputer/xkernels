#!/usr/bin/env python
"""Characterize the fp32 GEMM divergence on GB10/triton-3.6: is ieee honored?"""
import torch
import triton
import triton.language as tl


# Hand-written minimal GEMM to test tl.dot input_precision on GB10
@triton.jit
def gemm_ieee(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BK
        b_ptrs += BK * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

M = N = K = 256
a = torch.randn(M, N, device="cuda", dtype=torch.float32)
b = torch.randn(M, N, device="cuda", dtype=torch.float32)
c = torch.empty_like(a)
gemm_ieee[(M // 64, N // 64)](a, b, c, M, N, K, BM=64, BN=64, BK=64)
ref = a @ b
print("triton", triton.__version__, "| cap", torch.cuda.get_device_capability(0))
print("fp32 ieee dot  max_abs_err:", float((c - ref).abs().max()))
print("  ~1e-5 = true ieee fp32 ; ~1e-3 = tf32 being used despite ieee")
