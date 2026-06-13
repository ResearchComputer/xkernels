# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #39 perf sweep for the sparse-MLA attention compute (#33) on gfx942.

Sweeps ``BLOCK_N`` (and a couple of occupancy knobs) over the V4 latent-MLA
geometry (``H=128``, ``D=512``, ``d_v=448``, MQA) at decode/prefill ``T`` and a
sweep of top-k, reporting median ms + speedup vs the #33 baseline (``BLOCK_N=64``)
and vs a naive gather+dense-softmax torch path. Correctness of each config is
checked against the baseline output. Run on one gfx942 GPU (see
``slurm/tune_v4_perf_beverin.sbatch``)::

    python benchmarks/tune_sparse_mla.py
"""
from __future__ import annotations

import json
import os

import torch

T, H, D, D_V = 1, 128, 512, 448  # V4 decode tile (single query token, MQA)


def _set_cfg(cfg: dict) -> None:
    if cfg is None:
        os.environ.pop("XKERNELS_SPARSE_MLA_CONFIG", None)
    else:
        os.environ["XKERNELS_SPARSE_MLA_CONFIG"] = json.dumps(cfg)


def main() -> None:
    if not torch.cuda.is_available():
        print("No GPU available; run the correctness test under TRITON_INTERPRET=1.")
        return
    import triton

    from xkernels import sparse_mla_attention
    from xkernels._backends import Backend
    from xkernels.ops.attention.triton.sparse_mla_config import (
        DEFAULT_SPARSE_MLA_CONFIG,
    )

    dev = "cuda"
    print("dev:", torch.cuda.get_device_name(0))
    sm_scale = 1.0 / (D**0.5)

    # Candidate configs: sweep BLOCK_N x (num_warps, waves_per_eu).
    cands = []
    for bn in (32, 64, 128, 256):
        for nw, wpe in ((4, 0), (4, 2), (8, 1), (8, 2)):
            cands.append({"BLOCK_N": bn, "num_warps": nw, "num_stages": 1,
                          "waves_per_eu": wpe})

    Kv = 8192
    for topk in (256, 512, 1024):
        for Tq in (1, 8):
            q = torch.randn(Tq, H, D, device=dev, dtype=torch.bfloat16)
            kv = torch.randn(Kv, D, device=dev, dtype=torch.bfloat16)
            idx = torch.randint(0, Kv, (Tq, topk), device=dev, dtype=torch.int32)
            sink = torch.randn(H, device=dev)

            def naive(q=q, kv=kv, idx=idx, sink=sink, topk=topk):
                ks = kv[idx.long()]
                s = torch.einsum("thd,tkd->thk", q.float(), ks.float()) * sm_scale
                s = torch.cat([s, sink.float().view(1, H, 1).expand(s.shape[0], H, 1)],
                              dim=-1)
                p = s.softmax(-1)[..., :topk]
                return torch.einsum("thk,tkd->thd", p, ks[..., :D_V].float())

            t_naive = triton.testing.do_bench(naive)

            _set_cfg(DEFAULT_SPARSE_MLA_CONFIG)
            ref = sparse_mla_attention(
                q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
                backend=Backend.TRITON,
            )
            base = triton.testing.do_bench(
                lambda q=q, kv=kv, idx=idx, sink=sink: sparse_mla_attention(
                    q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
                    backend=Backend.TRITON,
                )
            )

            print(f"\n=== sparse-MLA  Tq={Tq} topk={topk}  H={H} D={D} d_v={D_V} ===")
            print(f"naive: {t_naive:.4f} ms   baseline(#33 BLOCK_N=64): {base:.4f} ms")
            print(f"{'BLK_N':>5} {'warps':>5} {'wpe':>3} {'ms':>9} "
                  f"{'vs_base':>8} {'vs_naive':>8} {'max_err':>9}")
            results = []
            for d in cands:
                _set_cfg(d)
                try:
                    got = sparse_mla_attention(
                        q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
                        backend=Backend.TRITON,
                    )
                except Exception as exc:
                    print(f"{d['BLOCK_N']:5d}  SKIP ({type(exc).__name__})")
                    continue
                err = (got[0].float() - ref[0].float()).abs().max().item()
                ms = triton.testing.do_bench(
                    lambda q=q, kv=kv, idx=idx, sink=sink: sparse_mla_attention(
                        q, kv, idx, sm_scale=sm_scale, attn_sink=sink, d_v=D_V,
                        backend=Backend.TRITON,
                    )
                )
                flag = "" if err < 3e-2 else "  !!CORRECTNESS"
                print(f"{d['BLOCK_N']:5d} {d['num_warps']:5d} {d['waves_per_eu']:3d} "
                      f"{ms:9.4f} {base / ms:7.2f}x {t_naive / ms:7.2f}x "
                      f"{err:9.2e}{flag}")
                if err < 3e-2:
                    results.append((ms, d))
            _set_cfg(None)
            if results:
                results.sort(key=lambda x: x[0])
                best_ms, best = results[0]
                print(f"BEST Tq={Tq} topk={topk}: {best_ms:.4f} ms "
                      f"({base / best_ms:.2f}x vs base)  cfg={json.dumps(best)}")


if __name__ == "__main__":
    main()
