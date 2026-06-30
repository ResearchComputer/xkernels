#!/usr/bin/env python
"""Probe: can a CUTE DSL kernel index a bf16 tensor and accumulate in fp32?
If yes, the moe/rmsnorm/merge kernels can read bf16 directly (halve mem traffic)
instead of host-upcasting to fp32 first — the real lever for these mem-bound ops.
"""
import torch, cutlass, cutlass.cute as cute
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Tensor
from cutlass.cutlass_dsl import T

@cute.kernel
def k(gBf: Tensor, gFp: Tensor, N: cutlass.Constexpr):
    tidx = nvvm.read_ptx_sreg_tid_x(T.i32())
    if tidx < N:
        b = gBf[(tidx,)]      # bf16 load
        acc = cutlass.Float32(0.0)
        acc = acc + b         # does bf16 promote into fp32 acc?
        acc = acc * cutlass.Float32(2.0)
        gFp[(tidx,)] = acc

@cute.jit
def j(Bf, Fp, N: cutlass.Constexpr):
    k(Bf, Fp, N).launch(grid=[1,1,1], block=[64,1,1])

N = 32
bf = torch.randn(N, device="cuda", dtype=torch.bfloat16)
out = torch.zeros(N, device="cuda", dtype=torch.float32)
gBf = from_dlpack(bf); gFp = from_dlpack(out)
j(gBf, gFp, N)
torch.cuda.synchronize()
ref = bf.float() * 2.0
print("bf16-direct-load + fp32-accumulate works:", torch.allclose(out, ref, rtol=1e-3))
print("sample out[:4]:", out[:4].tolist())
print("ref  [:4]:", ref[:4].tolist())
