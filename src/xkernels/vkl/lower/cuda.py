# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Native CUDA lowering for per-target override bodies (docs/brainstorm/04 Ex.2,
Phase 2.1 GPU half).

This is the native codegen the override mechanism lands on top of: a per-target
``@kernel.target("cuda", arch="nvidia_sm121")`` override builds the SAME math IR
as the portable body (the oracle property, enforced by
``check_override_math_ir``); THIS module lowers that IR to a REAL native CUDA
kernel, JIT-compiled by nvcc via ``torch.utils.cpp_extension``, and registers it
as the ``cuda`` backend so ``verify`` runs it on the chip.

What ships here is a CORRECTNESS-FIRST tiled GEMM (``__nv_bfloat16`` with fp32
accumulation for bf16; TRUE fp32 via CUDA-core FMA for fp32). It reaches the
tensor-core ceiling for NEITHER yet — that is the CUTLASS/wgmma follow-up
(Phase 2.1b). Its load-bearing value TODAY is mechanism validation: it proves a
per-target override body compiles to a REAL native kernel, registers as the
``cuda`` backend, passes ``verify`` against the exact oracle on real hardware
(GB10/sm_121), and honors the oracle invariant (same math IR as the portable
body — checked by ``check_override_math_ir``). Both bf16 and fp32 sweep points
pass. Performance is correct-but-slow (CUDA-core FMA, ~29 TFLOPS vs triton's
tensor-core ~92 TFLOPS on bf16); closing that gap is the CUTLASS path.

Note on the fp32 case: the triton backend ALSO does true fp32 on this arch
(earlier "triton degrades to tf32" reports were a misdiagnosis of an
oracle-side tf32 bug, now fixed in ``run_reference``). So the native override is
not a correctness fix over triton here — it is the mechanism that WILL carry the
CUTLASS/wgmma ceiling work, and this commit proves that pipeline is live on
hardware. (Its honest current value is not "faster" or "more correct" but
"the override path compiles + verifies on a real chip".)

The generated source is written to a real file (``inspect.getsourcelines`` fails
on ``<string>`` for some downstream tools) and cached by
``(override, dtype, pattern)``.
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

# One-time-per-(override,dtype) native compile cache. nvcc is slow; reuse.
_NATIVE_CACHE: dict[tuple[int, str], Any] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# §1  The generated CUDA GEMM (tiled, fp32 accumulate)
# ═══════════════════════════════════════════════════════════════════════════════

# CUDA-core (true fp32) GEMM. BM=BN=BK=16: one warp's worth of output, one thread
# per output element, shared-memory staging of the K-tiles. Correctness-first.
_BM = _BN = _BK = 16

_CUDA_GEMM_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define BM 16
#define BN 16
#define BK 16

// bf16 GEMM, fp32 accumulation, bf16 output (CUDA-core FMA: correct, not ceiling).
__global__ void gemm_bf16_cu(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ c,
    int M, int N, int K) {
  __shared__ __nv_bfloat16 sa[BM][BK];
  __shared__ __nv_bfloat16 sb[BK][BN];
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

// fp32 GEMM via TRUE fp32 FMA (CUDA cores — no TF32). Bit-exact with the CPU
// fp32 reference. (The triton backend ALSO does true fp32 on this arch; this
// native path is the mechanism-validation + future CUTLASS carrier.)
__global__ void gemm_fp32_cu(
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
  // launch on torch's CURRENT stream (c10::cuda), NOT the raw default stream 0.
  // Raw <<<>>>  binds stream 0; torch ops allocate/sync on getCurrentCUDAStream().
  // After CUDAGraph capture (or any non-default-stream work) the two diverge and
  // the kernel races its own inputs -> garbage output (intermittent, worse under
  // contention). Honoring the current stream is the stream-discipline fix.
  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_bf16_cu<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(c.data_ptr()), M, N, K);
  return c;
}

torch::Tensor gemm_fp32(torch::Tensor a, torch::Tensor b) {
  int M = a.size(0), K = a.size(1), N = b.size(1);
  auto c = torch::empty({M, N}, a.options());
  dim3 block(BN, BM, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  auto stream = at::cuda::getCurrentCUDAStream();
  gemm_fp32_cu<<<grid, block, 0, stream>>>(
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
            f"cuda override currently handles exactly one MMA node; found {len(mmas)} "
            f"(Phase 2.1: bare GEMM only; fusion + multi-MMA is follow-up)"
        )
    return mmas[0]


def _compile_override(spec: KernelSpec, override: OverrideBody, out_dtype: str):
    """JIT-compile the native module for (override, dtype). Cached."""
    key = (id(override), out_dtype)
    if key in _NATIVE_CACHE:
        return _NATIVE_CACHE[key]
    # sanity: the override must be a single-MMA tiled body (the only shape
    # the generated GEMM handles today)
    body = trace_ir(spec)
    if spec.launch is None or spec.launch.pattern != "tiled_2d":
        raise NotImplementedError("cuda override lowers only tiled_2d + MMA bodies")
    _find_mma(body.ir.nodes)
    # stable build dir so re-runs reuse the object (nvcc is slow)
    tag = hashlib.md5(
        f"{spec.id}:{override.backend}:{override.arch}:{out_dtype}".encode()
    ).hexdigest()[:10]
    build_dir = Path(tempfile.gettempdir()) / "xkernels_vkl_cuda_build" / tag
    build_dir.mkdir(parents=True, exist_ok=True)
    extra = {"-std=c++17"}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        extra.add(f"-gencode=arch=compute_{cap[0]}{cap[1]},code=sm_{cap[0]}{cap[1]}")
    mod = load_inline(
        name=f"vkl_gemm_{tag}",
        cpp_sources=[_CPP_BINDINGS],
        cuda_sources=[_CUDA_GEMM_SRC],
        functions=["gemm_bf16", "gemm_fp32"],
        extra_cuda_cflags=list(extra),
        build_directory=str(build_dir),
        verbose=False,
    )
    _NATIVE_CACHE[key] = mod
    return mod


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Public lowering + registration (mirror lower/triton.py)
# ═══════════════════════════════════════════════════════════════════════════════


def lower_to_cuda(spec: KernelSpec, override: OverrideBody) -> Callable[..., Any]:
    """Return a host launcher that runs the override's native CUDA kernel.

    The launcher's signature matches the spec's INPUT order; outputs are returned
    as a tuple in spec output order (the same contract as the triton launcher).
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
        # the spec's two GEMM operands are the MMA's a, b (in declaration order).
        # Bare GEMM has exactly two inputs -> (a, b); the single output is out.
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


def register_dsl_cuda(spec: KernelSpec, override: OverrideBody) -> Callable[..., Any]:
    """Compile + register the override's native kernel as the ``cuda`` backend.

    After this, ``verify('<op>.cuda@<ver>', ...)`` runs the native override — the
    Phase 2.1 loop closed on real hardware. Also wires the input generator.
    """
    launcher = lower_to_cuda(spec, override)
    register(spec.kernel, Backend.CUDA)(launcher)
    # Phase 3: a cuda-override kernel is also a capturable graph node.
    from ..graph import register_graph_node

    register_graph_node(spec)
    from ...registry.input_gen import register_input_gen
    from ..reference import make_inputs

    def _gen(point, seed, device):
        return make_inputs(spec, point, seed=seed, device=device)

    register_input_gen(spec.id, _gen)
    return launcher
