# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #73 GPU-gated follow-up: flow real ``verify`` ms through ``record_trace``.

Phase E landed the persisted ``{edit, predicted, measured, rationale}`` tuning_trace
store with the cross-task *mechanism* CPU-tested (``tests/test_vkl_trace.py``).
The **predicted** half is closed-form; the **measured** half (``ms``) is
GPU-gated. This script is the on-device confirmation: it registers the DSL-
authored ``gemm_bf16`` Triton card, runs ``verify(measure_perf=True)`` on the GPU,
and calls ``record_trace`` with the real measured ``ms`` + the cost model's
auto-filled prediction. It then simulates a *second task* loading the schedule
for the same point and reading the prior record back — the cross-task compounding
win, now carrying a genuine GPU measurement instead of a stub.

Arch is selected by the device: ds5 (GB10) -> ``nvidia_sm121``, beverin (MI300A)
-> ``amd_cdna3``. The portable Triton kernel is arch-agnostic, so the same script
runs on both. ``verify``'s ``perf.tflops`` / ``achieved_bw_pct`` are ``None``
(open §11); the richer metrics absorb once track C (#74) feeds them — ``ms``-only
first, per the issue's own staging.

Run (ds5, via rcc + docker — see ``meta/docs/usage/ds5-testbed.md``)::

    rcc --profile ds5 push
    rcc --profile ds5 run --docker -s 'python -u scripts/vkl_trace_gpu_gate.py'

Run (beverin, via slurm — see ``meta/docs/usage/clusters.md``)::

    scripts/cluster.sh run --host beverin \
      srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 \
           --gpus-per-node=1 --time=00:10:00 \
      bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && python3 -u scripts/vkl_trace_gpu_gate.py'
"""

from __future__ import annotations

import json
import os

import torch

from xkernels import verify
from xkernels.vkl import prior_traces, record_trace, register_dsl, spec_of
from xkernels.vkl.examples import gemm_bf16

OP = "gemm_bf16@1.0.0"
CARD = "gemm_bf16.triton@1.0.0"

# The measurement point: the sweep's largest bf16 GEMM (a real compute regime,
# not a launch-bound toy). Override via env for a different point.
POINT = {
    "M": int(os.environ.get("XKL_M", "512")),
    "N": int(os.environ.get("XKL_N", "512")),
    "K": int(os.environ.get("XKL_K", "512")),
    "dtype": "bf16",
}

# The edit recorded: the gate's default BLOCK_M binding at this point. A real
# tuning task would sweep these; here one point is enough to prove the measured
# half flows through the store end-to-end.
EDIT = {"kind": "set_knob", "name": "BLOCK_M", "value": 128}


def arch_of() -> str:
    """Map the device to its xkernels arch id (ds5->sm121, beverin->cdna3)."""
    cap = torch.cuda.get_device_capability(0)
    # GB10 reports (12, 1); MI300A gfx942 is not a CUDA cap -> arch from env.
    env_arch = os.environ.get("XKL_ARCH")
    if env_arch:
        return env_arch
    if cap == (12, 1):
        return "nvidia_sm121"
    if cap[0] >= 0 and torch.version.hip:  # type: ignore[attr-defined]
        return "amd_cdna3"
    if cap[0] >= 8:
        return f"nvidia_sm{cap[0]}{cap[1]}"
    raise RuntimeError(f"cannot map device cap {cap} to an arch id; set XKL_ARCH")


def main() -> None:
    arch = arch_of()
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"torch {torch.__version__} | {dev} cap {cap} arch {arch}")
    register_dsl(spec_of(gemm_bf16), "triton")

    # ── Task 1: measure on the GPU, record the {predicted, measured, rationale} triple ──
    shape = {k: POINT[k] for k in ("M", "N", "K")}
    v = verify(CARD, arch=arch, shapes=[POINT], measure_perf=True)
    c = v["correctness"]
    perf = v.get("perf") or {}
    run_id = v.get("artifacts", {}).get("run_id", "")
    print(f"\n=== verify({CARD}, arch={arch}, point={POINT}) ===")
    print(
        f"compiled={v['compiled']} passed={c['passed']} "
        f"max_abs={c['max_abs_err']:.3e} max_rel={c['max_rel_err']:.3e}"
    )
    if v["artifacts"].get("error"):
        print("VERIFY ERROR:", v["artifacts"]["error"])
        return
    if perf.get("ms") is None:
        print("no perf.ms (measure_perf on a GPU device required)")
        return
    print(f"perf.ms={perf['ms']:.4f}  (tflops/bw_pct stubbed to {perf.get('tflops')})")

    record = record_trace(
        OP, arch, EDIT,
        shape=shape, dtype=POINT["dtype"], point=POINT,
        check="ok",
        measured={"ms": perf["ms"]},
        rationale=(
            f"GPU-measured baseline on {dev} (arch={arch}); BLOCK_M=128 default "
            f"binding at M=N=K=512 bf16. run_id={run_id}. Reuse this point; do "
            f"not re-search unless the edit changes."
        ),
        source=run_id,
    )
    print("\n=== record_trace (Task 1 wrote) ===")
    print(json.dumps(record, indent=2, default=str))

    # ── Task 2: load the schedule for the SAME point — the prior record is there ──
    prior = prior_traces(OP, arch, shape=shape, dtype=POINT["dtype"])
    print(f"\n=== prior_traces (Task 2 reads) — {len(prior)} record(s) ===")
    for rec in prior:
        print(
            f"  edit_key={rec['edit_key']} check={rec['check']} "
            f"measured.ms={rec.get('measured', {}).get('ms')} "
            f"rationale={rec['rationale'][:60]}..."
        )
    assert any(
        rec["edit_key"] == record["edit_key"]
        and rec["measured"].get("ms") == perf["ms"]
        for rec in prior
    ), "Task 2 did not retrieve the record Task 1 just wrote"
    print("\nGPU GATE PASSED: real verify ms flowed through record_trace "
          "and a second task retrieved it with the prior rationale.")


if __name__ == "__main__":
    main()
