# SPDX-License-Identifier: MIT
"""Diagnose dense bf16 GEMM speed on gfx942: which Linear shapes miss the
MFMA/hipBLASLt fast path, whether a blas-routing mode restores it, and the
Triton tl.dot bf16 ceiling (issue #17, phase 1)."""
from __future__ import annotations

import argparse
import os

import torch
import triton
import triton.language as tl


def _ms(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]


def _tflops(ms: float, M: int, K: int, N: int) -> float:
    """GEMM throughput in TFLOP/s (2*M*K*N flops). 0.0 if ms is non-positive."""
    if ms <= 0:
        return 0.0
    return 2 * M * K * N / (ms * 1e-3) / 1e12


def _apply_blas_mode(mode: str) -> dict:
    """Apply a bf16 GEMM routing mode and report the resolved state.

    Runtime-settable knobs are applied here; import-time env knobs
    (``TORCH_BLAS_PREFER_HIPBLASLT``, ``PYTORCH_TUNABLEOP_ENABLED``) are set by
    the SLURM job per invocation and only reported. Never raises — unavailable
    knobs are recorded in the returned dict so the log stays honest.
    """
    state: dict = {"mode": mode}
    setlib = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if mode == "hipblaslt" and callable(setlib):
        try:
            setlib("hipblaslt")
        except Exception as exc:
            state["setlib_err"] = str(exc)[:80]
    elif mode == "no-hipblaslt" and callable(setlib):
        try:
            setlib("cublas")  # maps to rocBLAS on ROCm
        except Exception as exc:
            state["setlib_err"] = str(exc)[:80]
    if mode == "tunableop":
        try:
            torch.cuda.tunable.enable(True)
            torch.cuda.tunable.tuning_enable(True)
        except Exception as exc:
            state["tunable_err"] = str(exc)[:80]
    try:
        if callable(setlib):
            state["preferred_blas"] = str(setlib())
    except Exception as exc:
        state["preferred_blas_err"] = str(exc)[:80]
    try:
        state["tunable_enabled"] = bool(torch.cuda.tunable.is_enabled())
    except Exception:
        pass
    state["env_TORCH_BLAS_PREFER_HIPBLASLT"] = os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT")
    state["env_PYTORCH_TUNABLEOP_ENABLED"] = os.environ.get("PYTORCH_TUNABLEOP_ENABLED")
    return state


@triton.jit
def _gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BN)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        krem = K - k0 * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < krem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < krem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = a @ b for a [M,K], b [K,N]; fp32 accumulate, single fixed tile.

    A ceiling reference for the bf16 MFMA path (issue #17) — not a registered
    backend. Works for any float dtype (tested in fp32 under the interpreter).
    """
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BM, BN, BK = 64, 128, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _gemm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=2,
    )
    return c


# Kimi-K2.6 dense / MLA / shared-expert / head shapes, as (K -> N). F.linear
# weight is [N, K]; flops = 2*M*K*N. lm_head N is a representative large tile.
SHAPES = [
    ("q_a_proj", 7168, 1536),
    ("kv_a_proj", 7168, 576),
    ("shexp_gate_up", 7168, 2048),
    ("shexp_down", 2048, 7168),
    ("lm_head", 7168, 32768),
    ("ffn_gate_up", 4096, 11008),
]
DECODE_M = [1, 2, 4, 8, 16, 32]
PREFILL_M = [512, 2048, 4096]


def _bench_linear(M, K, N, dtype):
    """Time F.linear (x[M,K] @ W[N,K]^T) and return (ms, TFLOP/s)."""
    import torch.nn.functional as F

    x = torch.randn(M, K, device="cuda", dtype=dtype)
    w = torch.randn(N, K, device="cuda", dtype=dtype)
    ms = _ms(lambda: F.linear(x, w))
    return ms, _tflops(ms, M, K, N)


def _bench_triton(M, K, N):
    """Time the single-tile Triton bf16 GEMM and return (ms, TFLOP/s)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ms = _ms(lambda: _triton_gemm(a, b))
    return ms, _tflops(ms, M, K, N)


def _sanity_check_triton():
    """One small bf16 check vs torch before timing, so a wrong kernel can't pass."""
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    got = _triton_gemm(a, b).float()
    ref = a.float() @ b.float()
    ok = torch.allclose(got, ref, atol=2e-1, rtol=2e-2)
    print(f"# triton bf16 GEMM sanity vs torch: {'OK' if ok else 'MISMATCH'} "
          f"(max|err|={(got - ref).abs().max().item():.3f})")


def sweep(mode, Ms):
    state = _apply_blas_mode(mode)
    print(f"\n# ==== mode={mode} ====")
    print(f"# state: {state}")
    print(f"{'shape':14} {'M':>5} {'K':>6} {'N':>6} "
          f"{'fp16':>8} {'bf16':>8} {'tritbf16':>9} {'bf16/fp16':>10} flag   (TFLOP/s)")
    for tag, K, N in SHAPES:
        for M in Ms:
            try:
                _, f16 = _bench_linear(M, K, N, torch.float16)
                _, b16 = _bench_linear(M, K, N, torch.bfloat16)
                _, tb16 = _bench_triton(M, K, N)
            except Exception as exc:  # OOM / unsupported -> record, keep sweeping
                print(f"{tag:14} {M:5d} {K:6d} {N:6d}  SKIP: {str(exc)[:48]}")
                continue
            ratio = b16 / f16 if f16 > 0 else 0.0
            flag = "MISS" if ratio < 0.5 else ""
            print(f"{tag:14} {M:5d} {K:6d} {N:6d} "
                  f"{f16:8.1f} {b16:8.1f} {tb16:9.1f} {ratio:10.2f} {flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", default="default",
        choices=["default", "hipblaslt", "no-hipblaslt", "tunableop"],
    )
    ap.add_argument("--regime", default="all", choices=["decode", "prefill", "all"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No GPU; the dense bf16 GEMM sweep requires gfx942 (or any CUDA/ROCm GPU).")
        return

    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    _sanity_check_triton()
    Ms = []
    if args.regime in ("decode", "all"):
        Ms += DECODE_M
    if args.regime in ("prefill", "all"):
        Ms += PREFILL_M
    sweep(args.mode, Ms)


if __name__ == "__main__":
    main()
