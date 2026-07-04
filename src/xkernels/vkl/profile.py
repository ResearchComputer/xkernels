# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase C: profile feedback (ncu / rocprof) onto schedule-IR nodes (issue #74).

The gap this closes: ``vkl_read_cost`` surfaces only the resolved binding summary
(the closed-form PREDICTION); the diagnose skills route by the dominant stall
reason, which lives in the ncu / rocprof MEASUREMENT, not on the schedule nodes.
Today an agent reads a profile file and maps it to schedule nodes mentally. This
module keys the measured metrics to the schedule's **node ids** so the routing
decision reads straight off the ``MapTo`` node.

Three jobs, each honest about its GPU gate:

  * **normalize** (``ProfileMetrics`` + ``route``): map the raw ncu / rocprof tables
    onto the cross-vendor §10 vocabulary (``bottleneck`` / ``dominant_stall`` /
    ``achieved_bw`` / ``compute_throughput`` / ``tensor_pipe_util`` /
    ``occupancy``). ``route`` is the CAUSAL routing the diagnose skills branch on
    (the dominant stall reason, not the throughput ratio alone).
  * **parse** (``parse_ncu_report`` / ``parse_rocprof_compute``): a label-scanning
    parser for each profiler's text-table output, built to the formats the
    ``use-nsight-compute`` / ``use-rocprof-compute`` skills document and tested
    against synthetic fixtures modeled on their real on-device numbers. Validating
    against a LIVE ``.report.txt`` / ``.analyze.txt`` is the GPU gate (bristen
    sm_80 / beverin gfx942) — the parser's job here is to be format-tolerant enough
    that the GPU run only CONFIRMS, not redesigns. Missing metrics degrade to
    ``None``; the router never raises on a sparse profile.
  * **key to nodes** (``annotate_schedule``): project one kernel-level
    ``ProfileMetrics`` onto the schedule's node ids — the ``MapTo`` node carries
    the bottleneck + dominant stall + compute/tensor signal; the ``Stage`` /
    ``Tile`` nodes carry the achieved-bandwidth signal (the load pipeline). This is
    criterion 1 (metrics keyed to node ids, not just the kernel symbol).

Pure logic (no torch, no GPU). The parser runs on text; the measurement that
produces that text is GPU-gated. ``route_of(sched)`` is the one-line consumer a
diagnose skill calls (criterion 3).
"""
from __future__ import annotations

import re
from typing import Any

from .ir.schedule import MapTo, ProfileMetrics, ScheduleIR, Stage, Tile

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Routing — the causal diagnose-skill decision (dominant stall reason)
# ═══════════════════════════════════════════════════════════════════════════════


def route(m: ProfileMetrics) -> str:
    """The diagnose skill ``m`` routes to, by the skills' causal rule.

    Precedence (the skills' "trust the dominant stall reason, it's causal" rule;
    the compute/mem ratio is a cross-check, never the primary signal):

      1. memory-latency stall (or memory bottleneck)   -> ``diagnose-memory-bound``
      2. dependency / scheduling latency                -> ``diagnose-low-occupancy``
      3. VGPR / scratch pressure                        -> ``diagnose-low-occupancy``
      4. compute-bound with an idle matrix engine       -> ``map-to-matrix-cores``
      5. compute-bound, tensor engine busy, low occ.    -> ``diagnose-low-occupancy``
      6. otherwise (sparse profile, can't decide)       -> ``diagnose-memory-bound``
         (the safest first probe; a memory-bound kernel is the common case and the
         memory probe is cheap)

    Returns a skill ``id`` slug (``diagnose-memory-bound`` / …), ready for the MCP
    tool to surface and a diagnose skill to branch on.
    """
    # (1) memory latency is the dominant stall, OR the bottleneck label says so.
    if m.dominant_stall == "memory_latency" or m.bottleneck == "memory":
        return "diagnose-memory-bound"
    # (2)–(3) latency / occupancy stalls -> diagnose-low-occupancy.
    if m.dominant_stall in {"dependency", "scheduling", "vgpr", "scratch"}:
        return "diagnose-low-occupancy"
    # (4)–(5) compute-bound: is the matrix engine doing the work?
    if m.bottleneck == "compute":
        tensor_busy = (
            m.tensor_pipe_util_pct is not None and m.tensor_pipe_util_pct >= 30.0
        )
        if not tensor_busy:
            return "map-to-matrix-cores"  # spending compute on FMA, not the engine
        if m.occupancy_fraction is not None and m.occupancy_fraction < 0.5:
            return "diagnose-low-occupancy"  # engine busy but starved of work
        return "map-to-matrix-cores"
    # (6) sparse profile: safest first probe.
    return "diagnose-memory-bound"


def route_of(sched: ScheduleIR) -> dict[str, Any] | None:
    """The routing decision read straight off the annotated schedule (criterion 3).

    Looks for the first ``MapTo`` node carrying a ``ProfileMetrics`` annotation and
    returns ``{node_id, skill, bottleneck, dominant_stall, dominant_stall_pct}``.
    ``None`` if no ``MapTo`` is annotated (the schedule has no profile feedback
    yet — the caller falls back to the external profiler).
    """
    for node in sched.maps():
        pm = sched.profile.get(node.id)
        if pm is not None:
            return {
                "node_id": node.id,
                "skill": route(pm),
                "bottleneck": pm.bottleneck,
                "dominant_stall": pm.dominant_stall,
                "dominant_stall_pct": pm.dominant_stall_pct,
            }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Format-tolerant text scanners (the parsers are built to the skills' formats)
# ═══════════════════════════════════════════════════════════════════════════════


def _lines_with(text: str, label: str):
    """Yield the rest-of-line text for every line containing ``label``.

    A metric label often appears twice — once in a section header (``2.1.15
    Wavefront Occupancy`` with no value) and once on the value line itself. This
    yields them in order so the float/pct helpers can take the first line that
    actually carries a number.
    """
    needle = label.lower()
    for line in text.splitlines():
        j = line.lower().find(needle)
        if j >= 0:
            yield line[j + len(label):]


def _first_float_after(text: str, label: str, max_chars: int = 120) -> float | None:
    """The first ``float`` after ``label`` (case-insensitive).

    The robust primitive both parsers share: ncu / rocprof section tables align
    label and value in columns whose exact width varies by version, but the value
    is always the first number on the label's line. Scans every line carrying the
    label (a label may appear in a section header with no value) and returns the
    first number found, tolerating column drift.
    """
    for rest in _lines_with(text, label):
        m = re.search(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", rest[:max_chars])
        if m is not None:
            try:
                return float(m.group(0))
            except ValueError:  # pragma: no cover - regex guarantees a number
                continue
    return None


def _first_pct_after(text: str, label: str, max_chars: int = 120) -> float | None:
    """Like ``_first_float_after`` but constrained to a percentage value.

    ncu / rocprof place the ``%`` in a *unit column* (``DRAM Throughput    (%)
    68.14``) rather than adjacent to the number, so matching on the ``%`` glyph is
    fragile. This is only ever called for percentage metrics (Throughput /
    Occupancy / pipeline-utilization), so the robust rule is: the first float in
    ``[0, 100]`` on any line carrying the label IS the percentage. (A unit-bearing
    field like SM Frequency in GHz also falls in this range, but that field is
    never fetched through this helper — ``Duration`` and ``SM Frequency`` use
    ``_first_float_after``.)
    """
    for rest in _lines_with(text, label):
        for m in re.finditer(r"\d+(?:\.\d+)?", rest[:max_chars]):
            try:
                val = float(m.group(0))
            except ValueError:  # pragma: no cover
                continue
            if 0.0 <= val <= 100.0:
                return val
    return None


def _bottleneck(dram_pct: float | None, compute_pct: float | None) -> str:
    """The bottleneck label from the two throughput percentages (the route cross-check)."""
    if dram_pct is None and compute_pct is None:
        return "latency"  # can't decide from throughput alone
    d = dram_pct or 0.0
    c = compute_pct or 0.0
    if d > c:
        return "memory"
    return "compute"


# ─── ncu (NVIDIA Nsight Compute) ──────────────────────────────────────────────
# Built to the metric names + example numbers the ``use-nsight-compute`` skill
# documents (a real dual_rmsnorm A100 profile):
#   SpeedOfLight:    DRAM 68.14% vs Compute 53.87%, SM Freq 1.40 GHz, Duration 38.50 µs
#   ComputeWorkload: Executed Ipc Active 2.31, SM Busy 57.90%, FMA pipeline 42.9%
#   SchedulerStats:  Active Warps/Scheduler 14.99 (of 16), Eligible 1.69, No Eligible 42.35%
#   WarpStateStats:  Warp Cycles/Issued Instr 26.00; "11.0 cycles stalled ... L1TEX ... 42.3%"


# NVIDIA pipeline tokens that name the matrix engine (a high % here = compute is
# on the tensor engine; a low % with high FMA = compute is on CUDA cores).
_NVIDIA_TENSOR_PIPES = ("hmma", "wmma", "tensor", "wgmma", "fma_hmma")


def _ncu_dominant_stall(text: str) -> tuple[str | None, float | None]:
    """The (normalized stall reason, pct) from a Warp State Statistics section.

    The OPT line names the winner: e.g. *"11.0 cycles stalled waiting for a L1TEX
    … operation … 42.3% of the total"*. Map the raw name onto the §10 vocabulary.
    """
    # Find the Warp State / Stall section once, then scan within it.
    low = text.lower()
    sec = low.find("warp state")
    region = text[sec:] if sec >= 0 else text
    rlow = region.lower()
    # A stall OPT line: "stalled ... <TOKEN> ... <pct>% of the total".
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s+(?:the\s+)?total", rlow)
    pct = float(pct_match.group(1)) if pct_match else None
    # Classify by the resource named in the stall.
    if re.search(r"l1tex|\blg\b|mio\s*throttle|memory", rlow):
        return "memory_latency", pct
    if re.search(r"long scoreboard|tensor|mma", rlow):
        # Long scoreboard on a compute-bound kernel = idle tensor pipe upstream;
        # ``route`` confirms the compute-bound context before routing to matrix cores.
        return "tensor_pipe", pct
    if re.search(r"wait|scoreboard|dependency", rlow):
        return "dependency", pct
    if re.search(r"not selected|no instruction|eligible", rlow):
        return "scheduling", pct
    return None, pct


def parse_ncu_report(text: str, arch: str = "nvidia_sm80") -> ProfileMetrics:
    """Parse an ncu ``.report.txt`` (section tables) into normalized ``ProfileMetrics``.

    Format-tolerant: scans for the metric labels the ``use-nsight-compute`` skill
    documents (SpeedOfLight / ComputeWorkloadAnalysis / SchedulerStats /
    WarpStateStats), extracting the first numeric value in each label's
    neighborhood. Unknown / absent metrics degrade to ``None``; the router handles
    a sparse profile. **GPU gate**: validated against synthetic fixtures modeled on
    the skill's real on-device numbers; confirming against a live ``.report.txt``
    from bristen (sm_80) is the remaining GPU step.
    """
    dram = _first_pct_after(text, "DRAM Throughput")
    compute = _first_pct_after(text, "Compute (SM) Throughput")
    if compute is None:
        # Older ncu builds spell it "SM Throughput".
        compute = _first_pct_after(text, "SM Throughput")
    ipc = _first_float_after(text, "Executed Ipc Active")
    if ipc is None:
        ipc = _first_float_after(text, "Instruction Issued Per Cycle")
    duration_us = _first_float_after(text, "Duration")
    # Occupancy: prefer the achieved ratio when present; fall back to active/peak.
    occ_pct = _first_pct_after(text, "Achieved Active Warps")
    if occ_pct is None:
        occ_pct = _first_pct_after(text, "Achieved Occupancy")
    occ_frac = (occ_pct / 100.0) if occ_pct is not None else None
    # The named pipeline utilization (FMA / HMMA / tensor). We want the matrix-
    # engine pipeline's %; if only FMA is named, the tensor engine is ~idle.
    tensor_pct: float | None = None
    low = text.lower()
    for pipe in _NVIDIA_TENSOR_PIPES:
        idx = low.find(pipe)
        if idx >= 0:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", text[idx: idx + 80])
            if m:
                tensor_pct = float(m.group(1))
                break
    stall, stall_pct = _ncu_dominant_stall(text)
    bottleneck = _bottleneck(dram, compute)
    return ProfileMetrics(
        bottleneck=bottleneck,
        profiler="ncu",
        dominant_stall=stall,
        dominant_stall_pct=stall_pct,
        achieved_bw_pct=dram,
        compute_throughput_pct=compute,
        tensor_pipe_util_pct=tensor_pct,
        ipc_active=ipc,
        occupancy_fraction=occ_frac,
        duration_us=duration_us,
    )


# ─── rocprof (AMD ROCm Compute Profiler, formerly Omniperf) ───────────────────
# Built to the table numbers the ``use-rocprof-compute`` skill documents (a real
# dual_rmsnorm MI300A profile):
#   2.1.15 Wavefront Occupancy = 50.37% (achieved/peak wavefronts)
#   5.x Stall sections (CPF / CPC / SQ)  -> the dominant stall reason
#   roofline.csv places the dot; ASCII speed-of-light bars give a one-glance read


def _rocprof_dominant_stall(text: str) -> tuple[str | None, float | None]:
    """The (normalized stall reason, pct) from the rocprof 5.x stall tables."""
    low = text.lower()
    sec = low.find("5.")
    region = text[sec:] if sec >= 0 else text
    rlow = region.lower()
    # rocprof stall tables report a "%" share per stall bucket; grab the largest.
    pcts = re.findall(r"(\d+(?:\.\d+)?)\s*%", rlow)
    pct = max((float(p) for p in pcts), default=None)
    if re.search(r"wait|memory|tcc|l1|lds|lds_bound", rlow):
        return "memory_latency", pct
    if re.search(r"valuv?_dep|scoreboard|dependency|inst_fetch", rlow):
        return "dependency", pct
    if re.search(r"vgpr|register", rlow):
        return "vgpr", pct
    if re.search(r"scratch", rlow):
        return "scratch", pct
    return None, pct


def parse_rocprof_compute(text: str, arch: str = "amd_cdna3") -> ProfileMetrics:
    """Parse a rocprof ``<name>.analyze.txt`` into normalized ``ProfileMetrics``.

    Format-tolerant: scans for the metric labels the ``use-rocprof-compute`` skill
    documents (``2.1.15 Wavefront Occupancy``, the ``5.x`` stall tables, the
    roofline dot). Unknown / absent metrics degrade to ``None``. **GPU gate**:
    validated against synthetic fixtures modeled on the skill's real on-device
    numbers; confirming against a live ``.analyze.txt`` from beverin (gfx942) is
    the remaining GPU step.

    Aliased as ``parse_omniperf_analyze`` for discoverability (the tool was renamed
    from Omniperf to the ROCm Compute Profiler).
    """
    occ_pct = _first_pct_after(text, "Wavefront Occupancy")
    if occ_pct is None:
        occ_pct = _first_pct_after(text, "Occupancy")
    occ_frac = (occ_pct / 100.0) if occ_pct is not None else None
    # AMD's speed-of-light ASCII bars carry a DRAM/HBM utilization label; the
    # roofline.csv names the memory region. Best-effort percentage scan.
    bw_pct = _first_pct_after(text, "DRAM")
    if bw_pct is None:
        bw_pct = _first_pct_after(text, "HBM")
    stall, stall_pct = _rocprof_dominant_stall(text)
    bottleneck = "memory" if (bw_pct is not None and (stall == "memory_latency")) else (
        "compute" if stall == "tensor_pipe" else _bottleneck(bw_pct, None)
    )
    return ProfileMetrics(
        bottleneck=bottleneck,
        profiler="rocprof",
        dominant_stall=stall,
        dominant_stall_pct=stall_pct,
        achieved_bw_pct=bw_pct,
        compute_throughput_pct=None,  # rocprof surfaces this via the roofline dot, not a single %
        tensor_pipe_util_pct=None,    # ditto — the SQ+CPC MFMA utilization is a separate probe
        ipc_active=None,
        occupancy_fraction=occ_frac,
        duration_us=_first_float_after(text, "Mean"),
    )


# Stable alias: the skill + older docs still say "Omniperf".
parse_omniperf_analyze = parse_rocprof_compute


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Node-keying — one kernel-level profile projected onto schedule node ids
# ═══════════════════════════════════════════════════════════════════════════════


def annotate_schedule(
    sched: ScheduleIR,
    metrics: ProfileMetrics,
) -> ScheduleIR:
    """Attach ``metrics`` to the schedule's node ids (criterion 1).

    The keying heuristic (the whole point of Phase C — metrics keyed to nodes, not
    just the kernel symbol):

      * every **MapTo** node (the heavy MMA — the one the diagnose skills route on)
        carries the full annotation: bottleneck + dominant stall + compute +
        tensor utilization. This is where ``route_of`` looks.
      * every **Stage** / **Tile** node (the K-loop load pipeline) carries a
        *copy* of the annotation, but with its meaningful field the **achieved
        bandwidth** — the load pipeline is what a memory-bound kernel stalls on,
        so the ``diagnose-memory-bound`` coalescing / vector-load branch reads it
        there.

    The full ``ProfileMetrics`` is attached to every annotated node (not a
    trimmed view): the agent reasons over one shape, and a diagnose skill that
    wants the bandwidth signal off a Stage node shouldn't have to jump to the
    MapTo to get it. Returns a NEW frozen schedule (the trace-immutable form).
    """
    annotations: dict[str, ProfileMetrics] = {}
    for node in sched.nodes:
        nid = getattr(node, "id", None) or getattr(node, "name", None)
        if nid is None:
            continue
        if isinstance(node, MapTo):
            annotations[nid] = metrics
        elif isinstance(node, (Stage, Tile)):
            # The load pipeline: the bandwidth signal is the load-bearing field
            # here. Attach the whole blob for uniform reasoning, but flag the
            # node's role via the bottleneck (memory-bound kernels stall here).
            annotations[nid] = metrics
    return sched.with_profile(annotations)


def parse_profile(profiler: str, text: str, arch: str = "any") -> ProfileMetrics:
    """Dispatch to the matching parser by profiler name (``"ncu"`` | ``"rocprof"``).

    The MCP ``vkl_annotate_profile`` tool's entry: pick the parser, get normalized
    metrics, then ``annotate_schedule`` keys them to nodes.
    """
    profiler = (profiler or "").lower()
    if profiler in {"ncu", "nsight", "nsight-compute", "ncu_compute"}:
        return parse_ncu_report(text, arch=arch)
    if profiler in {"rocprof", "rocprof-compute", "omniperf", "rccl"}:
        return parse_rocprof_compute(text, arch=arch)
    raise ValueError(
        f"unknown profiler {profiler!r} (want one of ncu | rocprof | omniperf)"
    )


__all__ = [
    "ProfileMetrics",
    "route",
    "route_of",
    "parse_ncu_report",
    "parse_rocprof_compute",
    "parse_omniperf_analyze",
    "parse_profile",
    "annotate_schedule",
]
