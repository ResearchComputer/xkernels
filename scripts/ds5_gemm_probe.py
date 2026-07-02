#!/usr/bin/env python
"""Capture the generated Triton GEMM source for fp32 — does it emit input_precision=ieee?"""
from xkernels._dispatch import dispatch
from xkernels.vkl import make_inputs, register_dsl, spec_of
from xkernels.vkl.examples import gemm_bf16
from xkernels.vkl.lower.mathbody import _get_kernel
from xkernels.vkl.reference import trace_ir

spec = spec_of(gemm_bf16)
body = trace_ir(spec)
# compile for fp32
k = _get_kernel(body, "fp32", pattern="tiled_2d")
src = k.fn.src if hasattr(k.fn, "src") else k.src
print("=== generated kernel (fp32) ===")
for ln in src.splitlines():
    if "dot" in ln or "precision" in ln:
        print(">>", ln.strip())
# now run the fp32 case directly and compare
ins = make_inputs(spec, {"dtype": "fp32", "M": 128, "N": 256, "K": 256}, device="cuda")
register_dsl(spec, backend="triton")
out = dispatch(spec.kernel, backend="triton")(**ins)
ref = ins["a"] @ ins["b"]
print("fp32 max_abs_err:", float((out[0] - ref).abs().max()))
print("expected (ieee fp32, K=256): ~1e-4 ; if ~5e-3 then tf32 is in use")
