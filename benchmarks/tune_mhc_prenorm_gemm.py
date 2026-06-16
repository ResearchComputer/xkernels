# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #39 perf sweep for the MHC prenorm GEMM (#37) on gfx942.

Sweeps the candidate launch configs (``get_autotune_configs``) over the V4-Flash
MHC shape (``K=16384``, ``N=24``) at decode batch sizes, and reports the median
ms + speedup vs the #36 baseline config and vs the naive ``F.linear+sqsum`` torch
path. Correctness of every config is checked against the summed invariant. Run on
one gfx942 GPU (see ``slurm/tune_v4_perf_beverin.sbatch``)::

    python benchmarks/tune_mhc_prenorm_gemm.py
"""
from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F


def _set_cfg(cfg: dict) -> None:
    if cfg is None:
        os.environ.pop("XKERNELS_MHC_GEMM_CONFIG", None)
    else:
        os.environ["XKERNELS_MHC_GEMM_CONFIG"] = json.dumps(cfg)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; needs a gfx942 (or any CUDA/ROCm) GPU.")
        return
    import triton

    from xkernels import hc_prenorm_gemm
    from xkernels._backends import Backend
    from xkernels.ops.mhc.triton.configs import (
        BASELINE_MHC_GEMM_CONFIG,
        get_autotune_configs,
    )

    dev = "cuda"
    print("dev:", torch.cuda.get_device_name(0))

    # candidate config dicts (Config -> our launch dict)
    cands = []
    for c in get_autotune_configs():
        d = dict(c.kwargs)
        d["num_warps"] = c.num_warps
        d["num_stages"] = c.num_stages
        cands.append(d)

    for T in (1, 8, 64):
        hc_mult, hidden = 4, 4096
        K, N, n_splits = hc_mult * hidden, 24, 16
        a = torch.randn(T, K, device=dev, dtype=torch.bfloat16)
        fn = torch.randn(N, K, device=dev, dtype=torch.float32)
        fmul = F.linear(a.float(), fn.float())
        fsqr = (a.float() ** 2).sum(-1)
        denom = fmul.abs().max().clamp_min(1e-6).item()

        def naive(a=a, fn=fn):
            af = a.float()
            return F.linear(af, fn.float()), (af * af).sum(-1)

        t_naive = triton.testing.do_bench(naive)

        # Baseline = the original #36 launch, so "vs_base" stays a true #36
        # comparison even after the winner was promoted to the default.
        _set_cfg(BASELINE_MHC_GEMM_CONFIG)
        base = triton.testing.do_bench(
            lambda a=a, fn=fn, n_splits=n_splits: hc_prenorm_gemm(
                a, fn, n_splits=n_splits, backend=Backend.TRITON
            )
        )

        print(f"\n=== MHC prenorm GEMM  T={T}  K={K} N={N} splits={n_splits} ===")
        print(f"naive F.linear+sqsum: {t_naive:.4f} ms   baseline(#36): {base:.4f} ms")
        print(f"{'BLOCK_M':>7} {'BLOCK_K':>7} {'warps':>5} {'stg':>3} "
              f"{'wpe':>3} {'ms':>9} {'vs_base':>8} {'vs_naive':>8} {'rel_err':>9}")
        results = []
        for d in cands:
            _set_cfg(d)
            try:
                mul, sqr = hc_prenorm_gemm(a, fn, n_splits=n_splits,
                                           backend=Backend.TRITON)
            except Exception as exc:  # config not runnable on this shape
                print(f"{d['BLOCK_M']:7d} {d['BLOCK_K']:7d}  SKIP ({type(exc).__name__})")
                continue
            rel = (mul.sum(0) - fmul).abs().max().item() / denom
            ok = rel < 5e-2 and torch.allclose(
                sqr.sum(0), fsqr, atol=2e-2, rtol=2e-2
            )
            ms = triton.testing.do_bench(
                lambda a=a, fn=fn, n_splits=n_splits: hc_prenorm_gemm(
                    a, fn, n_splits=n_splits, backend=Backend.TRITON
                )
            )
            flag = "" if ok else "  !!CORRECTNESS"
            print(f"{d['BLOCK_M']:7d} {d['BLOCK_K']:7d} {d['num_warps']:5d} "
                  f"{d['num_stages']:3d} {d['waves_per_eu']:3d} {ms:9.4f} "
                  f"{base / ms:7.2f}x {t_naive / ms:7.2f}x {rel:9.2e}{flag}")
            if ok:
                results.append((ms, d))
        _set_cfg(None)
        if results:
            results.sort(key=lambda x: x[0])
            best_ms, best = results[0]
            print(f"BEST T={T}: {best_ms:.4f} ms  ({base / best_ms:.2f}x vs base)  "
                  f"cfg={json.dumps(best)}")


if __name__ == "__main__":
    main()
