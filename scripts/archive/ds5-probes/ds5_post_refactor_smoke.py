#!/usr/bin/env python
"""Post-refactor GPU smoke for the 5 CUTE cards on ds5 (sm_121).

Confirms the shared ``_cute_backend/launch.py`` refactor (A1) didn't change
behavior: every card still PASSES verify on the GPU, and verify(measure_perf=True)
now fills tflops / achieved_bw_pct from the new cost model (B1) instead of None.

Run on ds5::

    export CUDA_HOME=/usr/local/cuda-13.0 && . .venv/bin/activate && \
    python scripts/archive/ds5-probes/ds5_post_refactor_smoke.py
"""
from __future__ import annotations

import torch

from xkernels import verify

assert torch.cuda.is_available(), "this smoke needs the ds5 GPU"

CARDS = [
    ("mm_fp8_blockscale.cuda@1.0.0", "nvidia_sm121"),
    ("dual_rmsnorm.cuda@1.0.0", "nvidia_sm121"),
    ("moe_sum_reduce.cuda@1.0.0", "nvidia_sm121"),
    ("mha_merge_state.cuda@1.0.0", "nvidia_sm121"),
    ("hc_prenorm_gemm.cuda@1.0.0", "nvidia_sm121"),
]

print(f"device: {torch.cuda.get_device_name()} "
      f"(cc {''.join(map(str, torch.cuda.get_device_capability()))})\n")

all_pass = True
for card_id, arch in CARDS:
    v = verify(card_id, arch=arch, measure_perf=True)
    ok = v["correctness"]["passed"] and v["compiled"]
    all_pass = all_pass and ok
    perf = v.get("perf", {})
    print(f"{'PASS' if ok else 'FAIL'}  {card_id:32s} "
          f"n={v['correctness']['n_points']} "
          f"max_rel={v['correctness']['max_rel_err']:.2e}  "
          f"ms={perf.get('ms')}  tflops={perf.get('tflops')}  "
          f"bw%={perf.get('achieved_bw_pct')}")

print("\n" + ("ALL GREEN" if all_pass else "FAILURES PRESENT"))
raise SystemExit(0 if all_pass else 1)
