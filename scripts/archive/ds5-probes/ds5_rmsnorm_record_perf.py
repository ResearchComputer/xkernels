# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Issue #66 GPU-gated follow-up (part 2): record ``perf.measured`` for the
DSL-authored ``rmsnorm`` triton card on ds5 (NVIDIA GB10, sm_121).

The card's regime is ``memory_bound`` (a single-pass row-wise reduce: read x +
read w, write out; the variance accumulation is in registers/L1, no extra DRAM
traffic). So the load-bearing roofline metric is ``achieved_bw_pct`` — the
algorithmic byte traffic vs the arch's measured DRAM ceiling. GB10's ceiling
(``nvidia_sm121: dram_bw_gbs=243.0``) comes from ``registry/cost_model.py``,
itself validated against measured GB10 numbers by the roofline survey — NOT
hand-waved.

Each entry is written via ``record_measurement`` with:
  * ``arch = nvidia_sm121``        (the target this was measured on)
  * ``source = <gate run_id>``      (reproducible: re-run ``ds5_rmsnorm_gpu_gate.py``)
  * ``ms``                         (median, triton do_bench over the sweep point)
  * ``achieved_bw_pct`` / ``tflops`` (derived from the algorithmic traffic vs the ceiling)

The §2.4 invariants (sourced + arch-cited) are enforced by ``record_measurement``.
"""

from __future__ import annotations

from xkernels import verify
from xkernels.registry import backend_callable, load_shape_sweep
from xkernels.registry.cost_model import arch_peaks
from xkernels.registry.input_gen import generate_inputs
from xkernels.registry.writeback import record_measurement
from xkernels.utils.benchmarking import benchmark
from xkernels.vkl import register_dsl, spec_of
from xkernels.vkl.examples import rmsnorm

ARCH = "nvidia_sm121"
CARD = "rmsnorm.triton@1.0.0"
OP = "rmsnorm@1.0.0"
_DT_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

# rmsnorm per row of width d: x*x (d), reduce-add (d), mean(1), rsqrt(1),
# x*inv (d), *w (d)  => ~3d + O(1) flops. (Memory-bound; tflops is the
# uninteresting but still-reported compute ceiling ratio.)
_FLOPS_PER_ELEM = 3.0


def _traffic_bytes(point: dict) -> float:
    """Algorithmic DRAM traffic (bytes): read x + read w + write out, single pass."""
    T, d = point["T"], point["d"]
    b = _DT_BYTES[point["dtype"]]
    return b * (T * d + d + T * d)  # x + w + out


def main() -> None:
    register_dsl(spec_of(rmsnorm), "triton")

    # The correctness+parity gate (run once) — its run_id is the reproducible
    # source handle for the perf entries below (same session, arch, kernel build).
    v = verify(CARD, arch=ARCH, measure_perf=True)
    assert v["compiled"] and v["correctness"]["passed"], v
    run_id = v["artifacts"]["run_id"]
    print(f"gate run_id={run_id}  (ms @ points[-1]={v['perf']['ms']:.4f})")

    peaks = arch_peaks(ARCH)
    bw_gbs, fp32_tflops = peaks["dram_bw_gbs"], peaks["fp32_tflops"]
    print(f"ceiling: dram_bw={bw_gbs} GB/s  fp32={fp32_tflops} TFLOPS\n")
    print(f"{'dtype':5} {'T':>5} {'d':>5} {'ms':>9} {'GB/s':>8} {'bw%':>7} {'tflops':>8}")

    fn = backend_callable(OP, "TRITON")
    for point in load_shape_sweep("rmsnorm"):
        ins = generate_inputs(OP, point, seed=0, device="cuda")
        ms = benchmark(lambda ins=ins: fn(**ins))
        bytes_ = _traffic_bytes(point)
        gbs = bytes_ / (ms * 1e-3) / 1e9
        bw_pct = gbs / bw_gbs * 100.0
        tflops = (_FLOPS_PER_ELEM * point["T"] * point["d"]) / (ms * 1e-3) / 1e12
        print(
            f"{point['dtype']:5} {point['T']:>5} {point['d']:>5} {ms:>9.4f} "
            f"{gbs:>8.1f} {bw_pct:>6.1f}% {tflops:>8.4f}"
        )
        record_measurement(
            CARD,
            arch=ARCH,
            shape={"T": point["T"], "d": point["d"]},
            dtype=point["dtype"],
            source=run_id,
            ms=ms,
            achieved_bw_pct=round(bw_pct, 1),
            tflops=round(tflops, 4),
        )

    print(f"\nrecorded {len(load_shape_sweep('rmsnorm'))} entries to {CARD} perf.measured")


if __name__ == "__main__":
    main()
