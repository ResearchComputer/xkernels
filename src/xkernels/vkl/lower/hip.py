# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Native HIP lowering for per-target override bodies (Phase D, issue #75).

This is the AMD twin of ``lower/cuda.py``. A per-target
``@kernel.target("hip", arch="amd_cdna3")`` override builds the SAME math IR as
the portable body (the oracle property, enforced by ``check_override_math_ir``);
THIS module lowers that IR to a REAL native HIP kernel, JIT-compiled by hipcc via
``torch.utils.cpp_extension.load_inline``, and registers it as the ``hip`` backend
so ``verify`` runs it on the chip (beverin / MI300A / gfx942).

Maturity bar (mirrors the CUDA twin — ``lower/cuda.py`` — exactly): what ships
here is a **correctness-first wavefront-FMA** tiled GEMM (``__hip_bfloat16`` with
fp32 accumulation for bf16; true fp32 via FMA for fp32). It reaches the MFMA
matrix-core ceiling for NEITHER yet — that is the ``v_mfma_*`` follow-up (the
``map-to-matrix-cores`` skill's native-intrinsic path). Its load-bearing value
TODAY is **mechanism validation**: it proves a per-target HIP override body
compiles to a REAL native kernel (hipcc, not a torch.matmul wrapper), registers
as the ``hip`` backend, passes ``verify`` against the exact oracle on real AMD
hardware (gfx942), and honors the oracle invariant (same math IR as the portable
body — checked by ``check_override_math_ir``). This is the issue-#75 criterion #1
gate at the same maturity the CUDA path ships at; the literal "MFMA kernel"
wording in the issue is the full aspirational version, parallel to the still-
unshipped CUTLASS/wgmma ceiling work on the CUDA side (Phase 2.1b).

Portability stance (AGENTS.md / ``port-cuda-to-hip`` skill): this is the
FUNCTIONAL port — "it runs correctly on AMD". The wave size (64) is honored in
the launch arithmetic (256 threads/block = 4 wavefronts); the tile is not
restaged for LDS-async or retuned for 64-lane occupancy, and the inner product is
generic FMA, not MFMA. Turning "it runs on AMD" into "it's good on AMD" is the
``tune-for-cdna`` + ``map-to-matrix-cores`` follow-up.

The generated source is written to a real file and cached by
``(override, dtype, pattern)`` — same lifecycle as ``lower/cuda.py``.
"""
from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load_inline

from ..._backends import Backend
from ..._dispatch import register
from ...registry.dtypes import to_short_dtype
from ..ir.math import MMA
from ..reference import trace_ir
from ..surface import KernelSpec, OverrideBody

# One-time-per-(override,dtype) native compile cache. hipcc is slow; reuse.
_NATIVE_CACHE: dict[tuple[int, str], Any] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# §1  The generated HIP GEMM (tiled, fp32 accumulate, wavefront-FMA)
# ═══════════════════════════════════════════════════════════════════════════════

# BM=BN=BK=16, one thread per output element, 256 threads/block. On CDNA that is
# 4 wavefronts of 64 (vs 8 warps of 32 on NVIDIA) — the kernel logic is
# vendor-neutral C++; only the wave/warp arithmetic differs. LDS stages the
# K-tiles. Correctness-first; NOT the MFMA ceiling (see module docstring).
_BM = _BN = _BK = 16

# HIP-native spellings proven on beverin gfx942 / ROCm 7.2 (hipcc via load_inline):
#   * __hip_bfloat16  +  __bfloat162float / __float2bfloat16  (hip/hip_bf16.h)
#   * at::cuda::getCurrentCUDAStream()  (the ROCm torch build exposes the
#     at::cuda stream API — include <ATen/cuda/CUDAContext.h>; do NOT spell
#     at::hip::, which is not in this build's symbol surface)
#   * triple-chevron <<<grid, block, 0, stream>>> launch (HIP supports it)
_HIP_GEMM_SRC = r"""
#include <hip/hip_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define BM 16
#define BN 16
#define BK 16

// bf16 GEMM, fp32 accumulation, bf16 output (wavefront FMA: correct, not ceiling).
__global__ void gemm_bf16_hk(
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ b,
    __hip_bfloat16* __restrict__ c,
    int M, int N, int K) {
  __shared__ __hip_bfloat16 sa[BM][BK];
  __shared__ __hip_bfloat16 sb[BK][BN];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * BM + ty;
  int col = blockIdx.x * BN + tx;
  float acc = 0.0f;
  for (int kb = 0; kb < K; kb += BK) {
    if (row < M && (kb + tx) < K) sa[ty][tx] = a[row * K + kb + tx];
    if ((kb + ty) < K && col < N) sb[ty][tx] = b[(kb + ty) * N + col];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < BK; ++i)
      acc += __bfloat162float(sa[ty][i]) * __bfloat162float(sb[i][tx]);
    __syncthreads();
  }
  if (row < M && col < N) c[row * N + col] = __float2bfloat16(acc);
}

// fp32 GEMM via TRUE fp32 FMA (no downcast). Bit-exact with the CPU fp32 ref.
__global__ void gemm_fp32_hk(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K) {
  __shared__ float sa[BM][BK];
  __shared__ float sb[BK][BN];
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * BM + ty;
  int col = blockIdx.x * BN + tx;
  float acc = 0.0f;
  for (int kb = 0; kb < K; kb += BK) {
    if (row < M && (kb + tx) < K) sa[ty][tx] = a[row * K + kb + tx];
    if ((kb + ty) < K && col < N) sb[ty][tx] = b[(kb + ty) * N + col];
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < BK; ++i) acc += sa[ty][i] * sb[i][tx];
    __syncthreads();
  }
  if (row < M && col < N) c[row * N + col] = acc;
}

torch::Tensor gemm_bf16(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  dim3 block(BN, BM, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  // launch on torch's CURRENT stream (c10::cuda), not the raw default stream 0.
  // Same stream-discipline fix as lower/cuda.py: after graph capture (or any
  // non-default-stream work) raw stream 0 races the kernel's inputs.
  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_bf16_hk<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __hip_bfloat16*>(b.data_ptr()),
      reinterpret_cast<__hip_bfloat16*>(c.data_ptr()), M, N, K);
  return c;
}

torch::Tensor gemm_fp32(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  dim3 block(BN, BM, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_fp32_hk<<<grid, block, 0, stream>>>(
      a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
  return c;
}
"""

_CPP_BINDINGS = r"""
torch::Tensor gemm_bf16(torch::Tensor a, torch::Tensor b);
torch::Tensor gemm_fp32(torch::Tensor a, torch::Tensor b);
"""


def _find_mma(nodes) -> MMA:
    mmas = [n for n in nodes if isinstance(n, MMA)]
    if len(mmas) != 1:
        raise NotImplementedError(
            f"hip override currently handles exactly one MMA node; found {len(mmas)} "
            f"(Phase D: bare GEMM only; fusion + multi-MMA is follow-up)"
        )
    return mmas[0]


def _compile_override(spec: KernelSpec, override: OverrideBody, out_dtype: str):
    """JIT-compile the native HIP module for (override, dtype). Cached.

    On a ROCm torch build, ``load_inline(cuda_sources=...)`` routes through
    hipcc (hipify is a no-op on the already-HIP-native spellings here). The
    build targets every ROCm arch by default (slow); we pin ``gfx942`` for the
    cdna3 card via ``PYTORCH_ROCM_ARCH`` so an iteration is one arch, not ~15.
    """
    key = (id(override), out_dtype)
    if key in _NATIVE_CACHE:
        return _NATIVE_CACHE[key]
    # sanity: the override must be a single-MMA tiled body (the only shape
    # the generated GEMM handles today)
    body = trace_ir(spec)
    if spec.launch is None or spec.launch.pattern != "tiled_2d":
        raise NotImplementedError("hip override lowers only tiled_2d + MMA bodies")
    _find_mma(body.ir.nodes)
    tag = hashlib.md5(
        f"{spec.id}:{override.backend}:{override.arch}:{out_dtype}".encode()
    ).hexdigest()[:10]
    build_dir = Path(tempfile.gettempdir()) / "xkernels_vkl_hip_build" / tag
    build_dir.mkdir(parents=True, exist_ok=True)
    # Pin the offload arch so the compile is one arch (gfx942 = MI300A / cdna3),
    # not torch's default ~15-arch sweep. Set as an env var around the call so
    # it REPLACES torch's arch list (a cflag would APPEND another --offload-arch).
    import os

    arch_for = {"amd_cdna3": "gfx942", "amd_cdna2": "gfx90a"}
    saved = os.environ.get("PYTORCH_ROCM_ARCH")
    try:
        if override.arch in arch_for:
            os.environ["PYTORCH_ROCM_ARCH"] = arch_for[override.arch]
        mod = load_inline(
            name=f"vkl_gemm_hip_{tag}",
            cpp_sources=[_CPP_BINDINGS],
            cuda_sources=[_HIP_GEMM_SRC],
            functions=["gemm_bf16", "gemm_fp32"],
            extra_cflags=["-std=c++17"],
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if saved is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = saved
    _NATIVE_CACHE[key] = mod
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Public lowering + registration (mirror lower/cuda.py)
# ═══════════════════════════════════════════════════════════════════════════════


def lower_to_hip(spec: KernelSpec, override: OverrideBody) -> Callable[..., Any]:
    """Return a host launcher that runs the override's native HIP kernel.

    Signature matches the spec's INPUT order; outputs returned as a tuple in spec
    output order (the same contract as the triton + cuda launchers).
    """
    input_names = tuple(spec.inputs)

    def launcher(*args: torch.Tensor, **kwargs: Any) -> Any:
        inputs: dict[str, torch.Tensor] = {}
        if args:
            for name, val in zip(input_names, args, strict=True):
                inputs[name] = val.contiguous()
        for name, val in kwargs.items():
            if name in spec.inputs:
                inputs[name] = val.contiguous()
        missing = set(spec.inputs) - set(inputs)
        if missing:
            raise TypeError(f"missing required inputs: {sorted(missing)}")
        a_name, b_name = input_names[0], input_names[1]
        a, b = inputs[a_name].cuda(), inputs[b_name].cuda()
        out_dtype = to_short_dtype(a.dtype)
        mod = _compile_override(spec, override, out_dtype)
        if out_dtype == "fp32":
            c = mod.gemm_fp32(a, b)
        else:
            c = mod.gemm_bf16(a, b)
        return (c,)

    return launcher


def register_dsl_hip(spec: KernelSpec, override: OverrideBody) -> Callable[..., Any]:
    """Compile + register the override's native kernel as the ``hip`` backend.

    After this, ``verify('<op>.hip@<ver>', ..., arch='amd_cdna3')`` runs the
    native override on the AMD chip — the Phase D loop closing on hardware.
    Also wires the input generator + graph node (mirrors register_dsl_cuda).
    """
    launcher = lower_to_hip(spec, override)
    register(spec.kernel, Backend.HIP)(launcher)
    from ..graph import register_graph_node

    register_graph_node(spec)
    from ...registry.input_gen import register_input_gen
    from ..reference import make_inputs

    def _gen(point, seed, device):
        return make_inputs(spec, point, seed=seed, device=device)

    register_input_gen(spec.id, _gen)
    return launcher
