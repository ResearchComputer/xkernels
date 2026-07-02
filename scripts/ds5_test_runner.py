#!/usr/bin/env python
"""ds5 Phase 2.1 native-CUDA-override runner. Compiled inside the container."""
import subprocess
import sys

# run pytest, capture, filter the NGC banner noise
r = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_vkl_override_cuda.py",
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
    ],
    capture_output=True,
    text=True,
)
NOISE = (
    "conversion_method",
    "Copyright (c)",
    "NVIDIA CORPORATION",
    "GOVERNING",
    "found at",
    "Product-Specific",
    "=====",
    "== PyTorch",
    "All rights",
    "NVIDIA Release",
    "PyTorch Version",
    "WARNING: Running pip",
    "NOTE: CUDA",
    "Using CUDA",
    "See https",
    "recommend",
    "Building editable",
    "Created wheel",
    "Stored in dir",
    "Successfully",
    "forward compat",
    "Compatibility mode",
    "Using NuGet",
    "ninja",
    "gpu_arch",
    "warning generated",
    "note generated",
    "^  ",
    "cuda_gemm",
    "^/tmp",
    "^In file",
    "instantiation of",
)
out = "\n".join(r.stderr.splitlines() + r.stdout.splitlines())
for ln in out.splitlines():
    if not ln.strip():
        continue
    if any(n in ln for n in NOISE):
        continue
    print(ln)
print("EXIT:", r.returncode)
