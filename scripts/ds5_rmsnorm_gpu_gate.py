# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #66 GPU-gated follow-up: verify + parity + perf for the DSL-authored
``rmsnorm`` portable-Triton card on ds5 (NVIDIA GB10, sm_121).

On CPU the triton card is *honestly uncompiled* (``test_triton_card_honestly_
uncompiled_without_gpu``). This is the on-device counterpart: it registers the
DSL-generated Triton kernel via ``register_dsl``, then runs ``verify`` +
``verify_parity`` + a timed perf pass on the GB10. The numbers feed the card's
``perf.measured`` (§6.2 compounding loop).

NB: ds5 is NVIDIA (sm_121). The card's AMD/CDNA3 ceiling is a *separate*
GPU-gated follow-up (tune-for-cdna, on beverin/gfx942). The portable Triton
kernel is arch-agnostic, so this run is the same code that will run there.
"""

from __future__ import annotations

import json

import torch

from xkernels import verify, verify_parity
from xkernels.vkl import register_dsl, spec_of
from xkernels.vkl.examples import rmsnorm

ARCH = "nvidia_sm121"
CARD = "rmsnorm.triton@1.0.0"
OP = "rmsnorm@1.0.0"


def main(measure_perf: bool = False) -> None:
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | {dev} cap {cap} arch {ARCH}")
    register_dsl(spec_of(rmsnorm), "triton")

    v = verify(CARD, arch=ARCH, measure_perf=measure_perf)
    c = v["correctness"]
    print(f"\n=== verify({CARD}) ===")
    print(f"compiled={v['compiled']} determinism={v['determinism_check']}")
    print(
        f"passed={c['passed']} n_points={c['n_points']} "
        f"max_abs={c['max_abs_err']:.3e} max_rel={c['max_rel_err']:.3e}"
    )
    print(f"failing={c['failing_shapes']}")
    if v["artifacts"].get("error"):
        print("ERROR:", v["artifacts"]["error"])
    perf = v.get("perf") or {}
    if perf.get("ms") is not None:
        print(
            f"perf: ms={perf['ms']:.4f}  tflops={perf.get('tflops')}  "
            f"bw_pct={perf.get('achieved_bw_pct')}"
        )
    else:
        print("perf: (not measured; pass --perf to time the main sweep point)")

    p = verify_parity(OP, archs=[ARCH])
    print(f"\n=== verify_parity({OP}) ===")
    print(
        f"agree={p['agree']} inconclusive={p['inconclusive']} "
        f"n_runnable={p['n_runnable']} max_rel={p['max_pairwise_rel_err']:.3e}"
    )
    print(f"runnable={p['per_backend_runnable']}")

    # Full result blob (machine-readable, for writeback).
    out = {"arch": ARCH, "card": CARD, "verify": v, "parity": p}
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    import sys

    main(measure_perf="--perf" in sys.argv)
