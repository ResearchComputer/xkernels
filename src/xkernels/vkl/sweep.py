# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The DSL's programmatic ``autotune-knob-sweep`` (docs/brainstorm/09 §7, 11 §2).

The substrate's ``autotune-knob-sweep`` skill is "enumerate the card's declared
``specialization_knobs``, ``verify``-per-point, record the winner." This module
is the DSL's realization of that procedure, with one thesis-level difference:
**the search is driven by schedule-IR edits**, not by hand-rolled dict mutation.

Concretely, for an op + target arch + concrete shape + dtype:

  1. ``schedule_from_card(card)`` rebuilds a ``ScheduleIR`` carrying one ``Knob``
     per declared specialization (the search space IS the card's declared space).
  2. Each candidate config is a sequence of ``SetKnob`` edits, validated by
     ``run_gate`` (docs/brainstorm/10 §5). For pure tile/meta knobs every edit is
     trivially ``Ok`` (value ∈ choices); the gate starts *rejecting* once
     ``MapTo``/``AddStage`` edits bring L5-divisibility / scratch-budget
     constraints (Phase 2.2b) — but the architecture is in place now.
  3. Each gate-passing config is measured by the unchanged ``verify(measure_perf=
     True, knobs=<config>)`` — the substrate's own path, so the winner is a real
     ``do_bench`` median, not a DSL reinvention.
  4. The min-``ms`` passing config is written to ``perf.measured`` (via the
     substrate's ``record_measurement``) and the whole sweep is appended to the
     card's ``provenance.tuning_trace`` (the compounding loop, §6.2: the next
     agent facing the same ``(arch, shape, dtype)`` reads the trace and skips the
     configs already tried).

This is where the Phase 2.0a 25%-of-ceiling gap closes toward Triton's reachable
ceiling: 2.0a ran ONE hardcoded config; 2.2 sweeps the declared space and records
the winner. The remaining gap to the vendor ceiling (~70%) is the Phase 2.1
native-override body's job (CUDA/CUTE / HIP/CK), graded against the vendor
roofline — never against this Triton winner.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..registry import get_card
from ..registry.loader import registry_root
from ..registry.writeback import record_measurement
from ..verify import verify
from .edits import SetKnob
from .gate import run_gate
from .ir.schedule import Knob, ScheduleIR

# ═══════════════════════════════════════════════════════════════════════════════
# §1  Schedule IR from a committed card (the declared search space)
# ═══════════════════════════════════════════════════════════════════════════════


def schedule_from_card(card) -> ScheduleIR:
    """Rebuild a ``ScheduleIR`` from a card's declared ``specialization_knobs``.

    Each declared knob (``{choices: [...]}``) becomes a ``Knob`` node bound to its
    first choice (the default point). The schedule is the editable truth; the
    card's declared space is the bound on what the sweep may try (the
    autotune-knob-sweep skill's "do not freestyle outside the declared space").
    """
    nodes: list[Knob] = []
    for name, spec in card.specialization_knobs.items():
        choices = tuple(spec["choices"])
        if not choices:
            continue
        nodes.append(Knob(name=name, value=choices[0], choices=choices))
    sched = ScheduleIR()
    for n in nodes:
        sched = sched.with_node(n)
    return sched


def enumerate_configs(
    schedule: ScheduleIR, *, max_configs: int | None = None
) -> Iterator[dict[str, int]]:
    """Yield the Cartesian product of the schedule's knob choices.

    ``max_configs`` caps the enumeration (a curated subset for fast iteration);
    the full product is the default. Order is deterministic (knob declaration
    order, then lexicographic within each), so the sweep is reproducible.
    """
    names = list(schedule.knobs)
    if not names:
        yield {}
        return
    choice_lists = [schedule.knobs[n].choices for n in names]
    count = 0
    for combo in itertools.product(*choice_lists):
        yield dict(zip(names, combo, strict=True))
        count += 1
        if max_configs is not None and count >= max_configs:
            return


# ═══════════════════════════════════════════════════════════════════════════════
# §2  One config -> validated schedule (the agent-editable primitive)
# ═══════════════════════════════════════════════════════════════════════════════


def apply_config(
    schedule: ScheduleIR, config: dict[str, int], arch: str
) -> tuple[ScheduleIR, bool, str]:
    """Bind ``config`` onto ``schedule`` via ``SetKnob`` edits through the gate.

    Returns ``(new_schedule, ok, reason)``. ``ok`` is False if any edit was
    rejected (the gate's decidability — a pure function of *(args, IR, arch)*).
    For tile/meta knobs this is always Ok today; it rejects once MapTo/Stage
    constraints arrive (Phase 2.2b).
    """
    edits = [SetKnob(name=n, value=int(v)) for n, v in config.items()]
    res = run_gate(edits, schedule, arch)
    ok = res.applied == len(edits) and res.rejected == 0
    reason = "; ".join(t.reason for t in res.trace if t.check == "reject")
    return res.final_ir, ok, reason


# ═══════════════════════════════════════════════════════════════════════════════
# §3  The sweep
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SweepEntry:
    """One config the sweep measured (the tuning-trace unit)."""

    config: dict[str, int]
    passed: bool
    ms: float | None
    winner: bool = False
    reason: str = ""


@dataclass(frozen=True)
class SweepResult:
    """Outcome of an autotune sweep."""

    impl_card_id: str
    arch: str
    point: dict[str, Any]
    winner: dict[str, int] | None
    winner_ms: float | None
    n_configs: int
    n_passed: int
    entries: tuple[SweepEntry, ...] = field(default_factory=tuple)
    gate: dict[str, Any] | None = None  # the Phase 2 roofline-gate verdict (2.3)

    def trace(self) -> list[dict[str, Any]]:
        """The ``provenance.tuning_trace`` payload (token-compact, agent-readable)."""
        out: list[dict[str, Any]] = []
        for e in self.entries:
            d: dict[str, Any] = {
                "config": e.config, "passed": e.passed, "ms": e.ms, "winner": e.winner,
            }
            if e.reason:
                d["reason"] = e.reason
            out.append(d)
        return out


def autotune(
    impl_card_id: str,
    *,
    arch: str,
    point: dict[str, Any],
    max_configs: int | None = None,
    seed: int = 0,
    record: bool = True,
    verbose: bool = False,
) -> SweepResult:
    """Sweep the card's declared specialization_knobs; record the winner + trace.

    Args:
      impl_card_id: e.g. ``"gemm_bf16.triton@1.0.0"`` (must be registered).
      arch: target arch (selects the device + the gate's arch constraints).
      point: the concrete ``{dtype, M, N, K, ...}`` to tune for.
      max_configs: cap the enumeration (None = full Cartesian product).
      record: if True, write the winner to ``perf.measured`` + the sweep to
        ``provenance.tuning_trace`` (the compounding loop).

    Returns a ``SweepResult``. Discards any config that fails correctness — a
    fast-but-wrong kernel is never the winner (autotune-knob-sweep pitfall).
    """
    card = get_card(impl_card_id)
    schedule = schedule_from_card(card)
    dtype = point.get("dtype", "")
    pattern = _launch_pattern_for(impl_card_id)

    entries: list[SweepEntry] = []
    winner: dict[str, int] | None = None
    winner_ms: float | None = None
    n_configs = 0

    for config in enumerate_configs(schedule, max_configs=max_configs):
        n_configs += 1
        _sched, ok, reason = apply_config(schedule, config, arch)
        if not ok:
            entries.append(SweepEntry(config=config, passed=False, ms=None, reason=reason))
            if verbose:
                print(f"  REJECT {config}: {reason}")
            continue
        # Phase 2.2b: the decidable scratch-overflow pre-check (the gate's
        # cost-model half). A config that overflows the arch's scratch budget is
        # skipped BEFORE launch — the 2.2a smem-overflow kernel crashes become
        # clean ``overflow`` rejects, recorded in the trace for the next agent.
        if pattern is not None and dtype:
            from . import cost
            if cost.overflows_scratch(pattern, config, dtype, arch):
                kb = cost.predict_scratch(pattern, config, dtype, arch) // 1024
                reason = f"scratch overflow: {kb} KB > arch budget"
                entries.append(SweepEntry(config=config, passed=False, ms=None, reason=reason))
                if verbose:
                    print(f"  OVERFLOW {config}: {reason}")
                continue
        try:
            r = verify(
                impl_card_id, arch=arch, knobs=config, shapes=[point],
                seed=seed, measure_perf=True,
            )
        except Exception as exc:  # noqa: BLE001 — a config that crashes is not the winner
            entries.append(SweepEntry(config=config, passed=False, ms=None, reason=str(exc)))
            if verbose:
                print(f"  ERROR {config}: {exc}")
            continue
        passed = bool(r["correctness"]["passed"])
        ms = r.get("perf", {}).get("ms")
        entries.append(SweepEntry(config=config, passed=passed, ms=ms))
        if verbose:
            tag = "ok" if passed else "FAIL"
            print(f"  {tag:4s} ms={ms} {config}")
        if passed and ms is not None and (winner_ms is None or ms < winner_ms):
            winner_ms = ms
            winner = dict(config)

    # Mark the winning entry (mutate a copy — entries are frozen).
    if winner is not None:
        entries = [
            SweepEntry(**{**asdict(e), "winner": (e.config == winner)}) for e in entries
        ]

    # Phase 2.3: the roofline gate on the measured winner (the §2 decision rule).
    # Graded against the VENDOR wgmma ceiling — the honest verdict (the autotuned
    # Triton GEMM is cuBLAS-parity but ~47% of the theoretical peak; BELOW_BAR is
    # the trigger for the Phase 2.1 native-override conversation, recorded honestly).
    gate_dict: dict[str, Any] | None = None
    if winner_ms is not None:
        op_id = card.implements
        from . import cost

        verdict = cost.roofline_gate(winner_ms, op_id, point, arch, instruction="wgmma")
        if verdict is not None:
            gate_dict = verdict.to_dict()

    result = SweepResult(
        impl_card_id=impl_card_id, arch=arch, point=dict(point),
        winner=winner, winner_ms=winner_ms, n_configs=n_configs,
        n_passed=sum(1 for e in entries if e.passed), entries=tuple(entries),
        gate=gate_dict,
    )

    if record and winner is not None:
        _record_sweep(card, result, dtype, source_run=impl_card_id)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Writeback — perf.measured (substrate) + tuning_trace (vkl)
# ═══════════════════════════════════════════════════════════════════════════════


def _record_sweep(card, result: SweepResult, dtype: str, *, source_run: str) -> None:
    """Write the winner to ``perf.measured`` and the sweep + gate to ``tuning_trace``."""
    # perf.measured: the substrate's own write-back (no-touch; invariant-enforced).
    record_measurement(
        result.impl_card_id, result.arch, result.point, dtype,
        source=f"vkl-sweep:{source_run}", knobs=result.winner, ms=result.winner_ms,
    )
    # tuning_trace: append the sweep history + the Phase 2 gate verdict (the
    # next agent's skip-list + the "is this worth re-sweeping?" answer).
    payload = result.trace()
    if result.gate is not None:
        payload = [{"_gate": result.gate}] + payload
    record_tuning_trace(result.impl_card_id, payload)


def record_tuning_trace(impl_card_id: str, trace: list[dict[str, Any]]) -> None:
    """Append ``trace`` to the card's ``provenance.tuning_trace`` and persist.

    The trace is namespaced provenance (the Phase 1 schema edit —
    ``additionalProperties``; the substrate loads it opaquely). Mirror's
    ``record_measurement``'s read-modify-write against the committed card JSON.
    """
    path = _card_path(impl_card_id)
    with path.open() as f:
        doc = json.load(f)
    prov = doc.setdefault("provenance", {})
    existing = list(prov.get("tuning_trace", []))
    existing.extend(trace)
    prov["tuning_trace"] = existing
    with path.open("w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


def _card_path(impl_card_id: str) -> Path:
    """Resolve the committed card JSON path (mirrors writeback._card_path)."""
    short = impl_card_id.split("@", 1)[0]
    return registry_root() / "impls" / f"{short}.card.json"


def _launch_pattern_for(impl_card_id: str) -> str | None:
    """The DSL kernel's launch pattern, if the op is DSL-authored.

    DSL ops register their ``KernelSpec`` on example import; the launch pattern
    (``tiled_2d`` | ``rowwise``) drives the scratch-footprint model (Phase 2.2b).
    Hand-op cards (no DSL spec) return ``None`` → no scratch pre-check (the
    substrate's own launch is trusted).
    """
    try:
        import xkernels.vkl.examples as _ex

        # The card's implements is "<short>@<v>"; short is "<kernel>.<backend>".
        short = impl_card_id.split(".")[0] if "." in impl_card_id else impl_card_id.split("@")[0]
        for attr in dir(_ex):
            spec = getattr(getattr(_ex, attr, None), "_vkl_spec", None)
            if spec is not None and spec.short_name == short:
                return spec.launch.pattern if spec.launch else None
    except Exception:
        return None
    return None
