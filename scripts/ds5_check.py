#!/usr/bin/env python
"""ds5 environment sanity check: GPU + native ext + triton + xkernels import."""
import torch
import triton

import xkernels
from xkernels.ops.ffn.cuda import _cuda  # noqa: F401

print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("xkernels:", xkernels.__name__)
print("native_ext_OK: True")
print("triton:", triton.__version__)
# CUDAGraph capture (Phase 3 unblock probe)
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        torch.matmul(a, b)
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = torch.matmul(a, b)
g.replay()
print("cuda_graph_capture_OK:", float((out - torch.matmul(a, b)).abs().max()) == 0.0)
