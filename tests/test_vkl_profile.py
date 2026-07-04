# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase C: profile feedback (ncu / rocprof) onto schedule-IR nodes (issue #74).

CPU-doable plumbing, tested here end-to-end against SYNTHETIC profile fixtures
modeled on the real on-device numbers the ``use-nsight-compute`` /
``use-rocprof-compute`` skills document:

  * ncu dual_rmsnorm A100:  DRAM 68.14% vs Compute 53.87%, IPC 2.31,
    FMA pipeline 42.9%, stall "L1TEX … 42.3% of the total" → memory-bound.
  * rocprof dual_rmsnorm MI300A:  Wavefront Occupancy 50.37% → memory-bound.

The parsers are format-tolerant (label-scanning, not column-fixed) so a live
``.report.txt`` / ``.analyze.txt`` only confirms them. Confirming against a LIVE
file is the GPU gate (bristen sm_80 / beverin gfx942); this test lands the
plumbing that gate will consume unchanged — parse → normalize → key to nodes →
route → serialize, all CPU-doable.

The four layers exercised:
  * normalize + route (``route``): the causal diagnose decision.
  * parse (``parse_ncu_report`` / ``parse_rocprof_compute``): text → metrics.
  * key to nodes (``annotate_schedule``): kernel-level metrics → node ids.
  * MCP surface (``vkl_annotate_profile`` / ``vkl_route_from_profile`` / the
    ``profile`` field ``vkl_read_cost`` now surfaces).
"""
from __future__ import annotations

import pytest

from xkernels.mcp_server import _dispatch
from xkernels.vkl import (
    ProfileMetrics,
    annotate_schedule,
    parse_ncu_report,
    parse_omniperf_analyze,
    parse_profile,
    parse_rocprof_compute,
    route,
    route_of,
    schedule_from_spec,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16

# ─── synthetic fixtures (modeled on the skills' real on-device profiles) ──────

# ncu .report.txt section tables. The exact column alignment varies by ncu
# version; the parser scans by label, so this fixture only needs to name the
# metrics in the documented shape. Numbers mirror the dual_rmsnorm A100 run the
# use-nsight-compute skill verified.
NCU_DUAL_RMSNORM_A100 = """
Section: GPU Speed Of Light Throughput
----------------------------------------------------------------------
DRAM Frequency                                                           (cycle/s) 1217 Mhz
SM Frequency                                                             (cycle/s) 1.40 gHz
Memory Throughput                                                            (%) 68.14
DRAM Throughput                                                              (%) 68.14
Compute (SM) Throughput                                                     (%) 53.87
Duration                                                            (second) 38.50 us

Section: Compute Workload Analysis
----------------------------------------------------------------------
SM Busy                                                                    (%) 57.90
Executed Ipc Active                                                           2.31
FMA                                                                     (inst) 42.9 %

Section: Scheduler Statistics
----------------------------------------------------------------------
Active Warps Per Scheduler      14.99 / 16      93.75 %
Eligible Warps Per Scheduler     1.69
Warps Issue Eligible                                    (%)  ...

Section: Warp State Statistics
----------------------------------------------------------------------
Warp Cycles Per Issued Instruction                                    cycle 26.00
OPT   11.0 cycles stalled waiting for a L1TEX (cache) operation 42.3% of the total
"""

# rocprof <name>.analyze.txt numbered tables. Numbers mirror the dual_rmsnorm
# MI300A run the use-rocprof-compute skill verified.
ROCPROF_DUAL_RMSNORM_MI300A = """
0.1 Top Kernels
    dual_rmsnorm_kernel   88.8 %   mean 75.7 us

1. System Info
    wave_size                       64
    max_waves_per_cu                40

2.1.15 Wavefront Occupancy
    Wavefront Occupancy                          50.37 %

5. SQ Instr Issue
    s_mem_load wait-cnt stalled                  45.2 %
    VALU Dep                                      8.1 %

Speed-of-light
    DRAM                                          70.0 %
"""


class TestRoute:
    """The causal diagnose routing (the skills' "dominant stall reason" rule)."""

    def test_memory_latency_stall_routes_to_memory_bound(self):
        m = ProfileMetrics(bottleneck="memory", profiler="ncu",
                           dominant_stall="memory_latency", dominant_stall_pct=42.3)
        assert route(m) == "diagnose-memory-bound"

    def test_dependency_stall_routes_to_low_occupancy(self):
        m = ProfileMetrics(bottleneck="latency", profiler="ncu",
                           dominant_stall="dependency")
        assert route(m) == "diagnose-low-occupancy"

    def test_vgpr_pressure_routes_to_low_occupancy(self):
        m = ProfileMetrics(bottleneck="latency", profiler="rocprof",
                           dominant_stall="vgpr")
        assert route(m) == "diagnose-low-occupancy"

    def test_compute_bound_idle_matrix_engine_routes_to_matrix_cores(self):
        # Compute-bound, but the matrix engine is idle (low tensor pipe util).
        m = ProfileMetrics(bottleneck="compute", profiler="ncu",
                           compute_throughput_pct=80.0, tensor_pipe_util_pct=5.0)
        assert route(m) == "map-to-matrix-cores"

    def test_compute_bound_busy_engine_low_occupancy_routes_to_low_occupancy(self):
        # Compute-bound AND tensor engine busy AND occupancy starved.
        m = ProfileMetrics(bottleneck="compute", profiler="ncu",
                           tensor_pipe_util_pct=70.0, occupancy_fraction=0.25)
        assert route(m) == "diagnose-low-occupancy"

    def test_sparse_profile_defaults_to_safest_probe(self):
        # A profile too sparse to decide: route to the cheapest first probe.
        m = ProfileMetrics(bottleneck="latency", profiler="ncu")
        assert route(m) == "diagnose-memory-bound"


class TestParseNcu:
    def test_dual_rmsnorm_a100_fixture_is_memory_bound(self):
        m = parse_ncu_report(NCU_DUAL_RMSNORM_A100, arch="nvidia_sm80")
        assert m.profiler == "ncu"
        assert m.bottleneck == "memory"            # DRAM 68 > Compute 54
        assert m.achieved_bw_pct == pytest.approx(68.14, abs=0.01)
        assert m.compute_throughput_pct == pytest.approx(53.87, abs=0.01)
        assert m.ipc_active == pytest.approx(2.31, abs=0.01)
        # The stall OPT line names L1TEX -> normalized to memory_latency.
        assert m.dominant_stall == "memory_latency"
        assert m.dominant_stall_pct == pytest.approx(42.3, abs=0.1)
        assert route(m) == "diagnose-memory-bound"

    def test_absent_metrics_degrade_to_none(self):
        m = parse_ncu_report("nothing recognizable here", arch="nvidia_sm80")
        assert m.achieved_bw_pct is None
        assert m.compute_throughput_pct is None
        assert m.dominant_stall is None
        # bottleneck can't be decided from throughput -> latency; route defaults.
        assert m.bottleneck == "latency"
        assert route(m) == "diagnose-memory-bound"


class TestParseRocprof:
    def test_dual_rmsnorm_mi300a_fixture_is_memory_bound(self):
        m = parse_rocprof_compute(ROCPROF_DUAL_RMSNORM_MI300A, arch="amd_cdna3")
        assert m.profiler == "rocprof"
        assert m.occupancy_fraction == pytest.approx(0.5037, abs=1e-4)
        assert m.dominant_stall == "memory_latency"
        assert m.bottleneck == "memory"
        assert route(m) == "diagnose-memory-bound"

    def test_omniperf_alias_matches(self):
        # The skill + older docs still say Omniperf; the alias must agree.
        assert parse_omniperf_analyze is parse_rocprof_compute


class TestParseProfileDispatch:
    def test_ncu_aliases_dispatch(self):
        for name in ("ncu", "nsight", "nsight-compute"):
            m = parse_profile(name, NCU_DUAL_RMSNORM_A100)
            assert m.profiler == "ncu"

    def test_rocprof_aliases_dispatch(self):
        for name in ("rocprof", "omniperf", "rocprof-compute"):
            m = parse_profile(name, ROCPROF_DUAL_RMSNORM_MI300A)
            assert m.profiler == "rocprof"

    def test_unknown_profiler_raises(self):
        with pytest.raises(ValueError):
            parse_profile("vtune", "...")


# ─── the schedule: gemm_bf16 lowers to out/a_tile/b_tile Tiles, one mma0 MapTo,
#     stage_a/stage_b Stages, plus declared Knobs ───────────────────────────────


@pytest.fixture
def gemm_sched():
    return schedule_from_spec(spec_of(gemm_bf16), arch="nvidia_sm90")


class TestAnnotateSchedule:
    def test_keys_metrics_to_mapto_and_load_pipeline(self):
        sched = schedule_from_spec(spec_of(gemm_bf16), arch="nvidia_sm90")
        m = ProfileMetrics(bottleneck="memory", profiler="ncu",
                           dominant_stall="memory_latency", achieved_bw_pct=68.14)
        out = annotate_schedule(sched, m)
        # The MapTo (the heavy op diagnose routes on) is annotated.
        assert "mma0" in out.profile
        # The load-pipeline nodes (Stages, Tiles) carry the bandwidth signal.
        assert "stage_a" in out.profile
        assert "stage_b" in out.profile
        assert "out" in out.profile
        # Knobs are NOT annotated (they're meta, not the compute graph).
        assert "BLOCK_M" not in out.profile
        # Frozen/immutable: the original schedule is untouched.
        assert sched.profile == {}

    def test_route_of_reads_the_mapto_annotation(self, gemm_sched):
        m = ProfileMetrics(bottleneck="memory", profiler="ncu",
                           dominant_stall="memory_latency", dominant_stall_pct=42.3)
        out = annotate_schedule(gemm_sched, m)
        decision = route_of(out)
        assert decision == {
            "node_id": "mma0",
            "skill": "diagnose-memory-bound",
            "bottleneck": "memory",
            "dominant_stall": "memory_latency",
            "dominant_stall_pct": 42.3,
        }

    def test_route_of_none_when_unannotated(self, gemm_sched):
        # No profile feedback yet -> the caller falls back to the external profiler.
        assert route_of(gemm_sched) is None


class TestMcpSurface:
    """The stateless agent loop: parse → annotate → route, via MCP tools."""

    def test_vkl_annotate_profile_keys_metrics_to_nodes_and_routes(self):
        r = _dispatch("vkl_annotate_profile", {
            "spec_id": "gemm_bf16",
            "arch": "nvidia_sm90",
            "applied_edits": [],
            "profiler": "ncu",
            "profile_text": NCU_DUAL_RMSNORM_A100,
        })
        # The annotated schedule view carries per-node profile inline + summary.
        assert "profile" in r
        assert "mma0" in r["profile"]
        mma0 = next(n for n in r["nodes"] if n["id"] == "mma0")
        assert "profile" in mma0                      # inline on the node
        assert mma0["profile"]["dominant_stall"] == "memory_latency"
        # The causal route reads off the MapTo node.
        assert r["route"]["node_id"] == "mma0"
        assert r["route"]["skill"] == "diagnose-memory-bound"
        # The closed-form cost PREDICTION is still there (Phase C is additive).
        assert "cost" in r and r["cost"]["pattern"] == "tiled_2d"
        assert "scratch_bytes" in r["cost"]

    def test_vkl_route_from_profile_returns_just_the_decision(self):
        r = _dispatch("vkl_route_from_profile", {
            "spec_id": "gemm_bf16",
            "arch": "amd_cdna3",
            "applied_edits": [],
            "profiler": "rocprof",
            "profile_text": ROCPROF_DUAL_RMSNORM_MI300A,
        })
        assert r["node_id"] == "mma0"
        assert r["skill"] == "diagnose-memory-bound"
        assert r["dominant_stall"] == "memory_latency"

    def test_vkl_route_routes_dependency_stall_to_low_occupancy(self):
        # A synthetic ncu fixture whose stall is a Wait (the canonical ncu
        # dependency-latency stall; per the skill, Wait/scoreboard -> low occupancy,
        # while a *Long* Scoreboard on a compute-bound kernel -> matrix cores).
        ncu_dep = """
        Section: GPU Speed Of Light Throughput
        Compute (SM) Throughput                                                     (%) 12.0
        DRAM Throughput                                                              (%) 10.0
        Section: Warp State Statistics
        OPT   18.0 cycles stalled on a Wait 60.0% of the total
        """
        r = _dispatch("vkl_route_from_profile", {
            "spec_id": "gemm_bf16",
            "arch": "nvidia_sm90",
            "applied_edits": [],
            "profiler": "ncu",
            "profile_text": ncu_dep,
        })
        assert r["skill"] == "diagnose-low-occupancy"
