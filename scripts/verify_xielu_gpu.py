# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""One-off GPU gate for the xielu op (issue #80): verify the Triton card on
amd_cdna3 + cross-backend parity vs the reference. Run inside the minisgl ROCm
container (gfx942) where the persistent venv + this editable xkernels checkout live.

    srun --environment=minisgl-rocm -n1 --gpus-per-task=1 python verify_xielu_gpu.py
"""
from __future__ import annotations

import json
import time
import traceback

import torch

from xkernels import verify, verify_parity, xielu
from xkernels._backends import Backend, detect_vendor
from xkernels._dispatch import registered_backends
from xkernels.ops.activation.reference import xielu as xielu_ref

_dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
print(f"torch={torch.__version__} hip={getattr(torch.version, 'hip', None)} "
      f"cuda_avail={torch.cuda.is_available()} vendor={detect_vendor()} device={_dev_name}")
print("xielu registered backends:", [b.value for b in registered_backends("xielu")])
assert registered_backends("xielu"), "xielu has no registered backends"

# --- 1) direct numerical check: triton kernel vs the reference oracle, bf16 ---
dev = "cuda"
for dtype in (torch.float32, torch.bfloat16):
    torch.manual_seed(0)
    x = torch.randn(256, 21504, device=dev).to(dtype)         # Apertus FFN intermediate
    alpha_p = (torch.rand(1, device=dev) * 0.5 + 0.2).to(dtype)
    alpha_n = (torch.rand(1, device=dev) * 0.5 + 0.2).to(dtype)
    out = xielu(x, alpha_p, alpha_n, backend=Backend.TRITON)
    ref = xielu_ref(x, alpha_p, alpha_n)
    max_abs = (out.float() - ref.float()).abs().max().item()
    print(f"[direct] dtype={dtype} triton-vs-ref max_abs_err={max_abs:.3e} "
          f"(rtol bar 1.6e-2 bf16 / 1e-5 fp32)")

# --- 2) harness verify on the Triton card (the full shape sweep, vs reference) ---
try:
    r = verify("xielu.triton@1.0.0", arch="amd_cdna3")
    print("VERIFY xielu.triton@1.0.0:", json.dumps({
        "compiled": r["compiled"],
        "passed": r["correctness"]["passed"],
        "max_abs_err": r["correctness"]["max_abs_err"],
        "max_rel_err": r["correctness"]["max_rel_err"],
        "n_points": r["correctness"]["n_points"],
        "failing": r["correctness"]["failing_shapes"],
    }, indent=2))
except Exception:
    print("VERIFY xielu.triton FAILED:")
    traceback.print_exc()

# --- 3) cross-backend parity (reference vs triton) ---
try:
    p = verify_parity("xielu@1.0.0", archs=["amd_cdna3"], device="cuda")
    print("PARITY xielu@1.0.0:", json.dumps({
        "agree": p["agree"],
        "per_backend_runnable": p["per_backend_runnable"],
    }, indent=2))
except Exception:
    print("PARITY xielu@1.0.0 FAILED:")
    traceback.print_exc()

# --- 4) perf microbenchmark vs the pure-torch path mini-sglang/vLLM use today ---
n_iter = 50
x = torch.randn(2048, 21504, device=dev, dtype=torch.bfloat16)
ap = (torch.rand(1, device=dev) * 0.5 + 0.2).to(torch.bfloat16)
an = (torch.rand(1, device=dev) * 0.5 + 0.2).to(torch.bfloat16)
for name, fn in (("triton", lambda: xielu(x, ap, an, backend=Backend.TRITON)),
                 ("reference_torch", lambda: xielu(x, ap, an, backend=Backend.REFERENCE))):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter * 1e3
    print(f"[perf] {name}: {dt:.3f} ms/iter (n_iter={n_iter}, shape={tuple(x.shape)})")
print("XIELU_GPU_GATE_DONE")
