#!/usr/bin/env python
"""The Phase 2.1 payoff, on ds5/GB10: triton (tf32) vs native (true fp32).

Run inside the container. Verifies that the triton card FAILS the fp32 point on
Blackwell (tf32 degradation) while the native CUDA override PASSES it (true fp32).
"""
from xkernels import verify
from xkernels.vkl import register_dsl, register_dsl_cuda, spec_of
from xkernels.vkl.examples import gemm_bf16

spec = spec_of(gemm_bf16)
register_dsl(spec, backend="triton")
register_dsl_cuda(spec, spec.override_for("cuda", "nvidia_sm121"))

pt = [{"dtype": "fp32", "M": 128, "N": 256, "K": 256}]
tr = verify("gemm_bf16.triton@1.0.0", arch="nvidia_sm121", shapes=pt)
cu = verify("gemm_bf16.cuda@1.0.0", arch="nvidia_sm121", shapes=pt)
print(f"triton fp32 (tf32 on Blackwell): passed={tr['correctness']['passed']} "
      f"abs_err={tr['correctness']['max_abs_err']:.4f}")
print(f"native  fp32 (true fp32 FMA):     passed={cu['correctness']['passed']} "
      f"abs_err={cu['correctness']['max_abs_err']:.6f}")
