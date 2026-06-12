# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase-0 probe (issue #41): does ``tl.dot`` on fp8 e4m3 reach native fp8 MFMA on
gfx942? Compiles a minimal fp8 dot for ``float8_e4m3fn`` AND ``float8_e4m3fnuz``,
greps the AMDGCN for ``v_mfma_*_fp8*``, and reports parity + TFLOP/s per format.

Run on one gfx942 GPU (see ``slurm/probe_fp8_mfma_beverin.sbatch``):

    AMDGCN_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python benchmarks/probe_fp8_mfma.py
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def _dot(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + ks[None, :])
        b = tl.load(b_ptr + rn[None, :] * K + ks[:, None])
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


def probe(dt, name):
    M = N = 2048
    K = 4096
    BM = BN = 128
    BK = 128
    a = torch.randn(M, K, device="cuda").to(dt)
    b = torch.randn(N, K, device="cuda").to(dt)
    c = torch.empty(M, N, device="cuda", dtype=torch.float32)
    comp = _dot[(M // BM, N // BN)](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK)
    asm = comp.asm.get("amdgcn", "") if hasattr(comp, "asm") else ""
    mfma_lines = [ln for ln in asm.splitlines() if "v_mfma" in ln]
    fp8_mfma = sorted({ln.strip().split()[0] for ln in mfma_lines if "fp8" in ln})
    any_mfma = sorted({ln.strip().split()[0] for ln in mfma_lines})
    ref = a.float() @ b.float().t()
    rel = (c - ref).abs().max().item() / ref.abs().max().clamp_min(1e-6).item()
    # Best-fit scalar s minimizing |c - s*ref|: exposes a pure decode/bias-scale
    # mismatch (s ~ const) vs genuine garbage (residual stays large after scaling).
    s = (c * ref).sum().item() / (ref * ref).sum().clamp_min(1e-6).item()
    rel_scaled = (c - s * ref).abs().max().item() / (s * ref).abs().max().clamp_min(1e-6).item()
    t = do_bench(lambda: _dot[(M // BM, N // BN)](a, b, c, M, N, K, BM=BM, BN=BN, BK=BK))
    tf = 2 * M * N * K / t / 1e9
    print(
        f"[{name}] rel={rel:.3e} bestfit_s={s:.4f} rel_after_s={rel_scaled:.3e} "
        f"time={t:.4f}ms {tf:.1f}TFLOP/s mfma={any_mfma} fp8_mfma={fp8_mfma}"
    )


def main():
    fmts = [(torch.float8_e4m3fn, "e4m3fn")]
    if hasattr(torch, "float8_e4m3fnuz"):
        fmts.append((torch.float8_e4m3fnuz, "e4m3fnuz"))
    for dt, name in fmts:
        try:
            probe(dt, name)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED:", repr(e)[:300])
    print("DECISION: format(s) with fp8_mfma>0 AND tight rel reach native fp8 MFMA.")


if __name__ == "__main__":
    main()
