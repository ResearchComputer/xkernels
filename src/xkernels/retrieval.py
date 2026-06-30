"""Structured retrieval over contract fields — meta/docs/library.md §3.

Two-stage to mirror the Op Spec / Impl Card split:
  1. Op stage:   filter Op Specs by canonical_op + decidable constraints + fusions.
  2. Impl stage: among matching ops, keep cards whose arch.family + arch.requires
                 fit the target arch / available features.

Every result (match or not) carries ``reject_reasons`` — the agent learns *why*
a card was excluded, which is itself training signal (§3.2). If an op exists but
no implementation matches the target backend, the op is returned with an
explicit ``missing_backend`` signal (the trigger for a porting skill §7.2).
"""
from __future__ import annotations

from typing import Any

from .registry import all_specs, cards_for
from .registry.archs import vendor_of as _vendor_of
from .registry.constraints import UndecidableConstraintError, evaluate
from .registry.models import ImplCard, OpSpec

# --- arch / feature matching --------------------------------------------------


def _arch_reject_reasons(card: ImplCard, target_arch: str, features: set[str]) -> list[str]:
    reasons: list[str] = []
    family = card.arch.family
    if family != "any":
        if family != target_arch:
            # allow same-vendor cross-arch? No: §2.3 perf is per-arch; a card
            # tuned for sm90 is not valid for sm80 without re-validation.
            reasons.append(f"arch.family {family!r} != target {target_arch!r}")
    for req in card.arch.requires:
        if req not in features:
            reasons.append(f"missing required feature {req!r}")
    # backend vendor coherence: a cuda card is not a candidate on an amd target
    # even if family was 'any' (e.g. a portable card mislabeled). The backend
    # enum is the source of truth for vendor.
    if card.backend.name == "cuda" and _vendor_of(target_arch) == "amd":
        reasons.append("cuda backend on an amd target")
    if card.backend.name == "hip" and _vendor_of(target_arch) == "nvidia":
        reasons.append("hip backend on an nvidia target")
    return reasons


# --- op-stage constraint evaluation -------------------------------------------

def _build_bindings(op: OpSpec, input_specs: dict[str, dict]) -> tuple[dict[str, Any], list[str]]:
    """Bind shape symbols + dtype from the query's concrete input specs.

    Returns (bindings, problems). An unbound symbol is *not* a rejection — it
    means the constraint is undecidable for this partial query, so we conservatively
    pass it through (we reject only on provable violation).
    """
    bindings: dict[str, Any] = {}
    problems: list[str] = []
    for arg, contract in op.inputs.items():
        qspec = input_specs.get(arg)
        if qspec is None:
            problems.append(f"no query binding for input {arg!r}")
            continue
        symbols = contract.get("shape_symbols", [])
        shape = qspec.get("shape")
        if shape is not None and len(shape) == len(symbols):
            for sym, val in zip(symbols, shape, strict=True):
                bindings[sym] = int(val)
        dtype = qspec.get("dtype")
        if dtype is not None:
            bindings[f"dtype:{arg}"] = dtype
    return bindings, problems


def _constraint_reject_reasons(op: OpSpec, bindings: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for c in op.constraints:
        try:
            if not evaluate(c, bindings):
                reasons.append(f"constraint violated: {c!r}")
        except UndecidableConstraintError:
            # couldn't decide (unbound symbol) — don't reject, just skip.
            continue
    return reasons


# --- the public query ---------------------------------------------------------

def find_impl(
    canonical_op: str,
    input_specs: dict[str, dict] | None = None,
    target_arch: str = "any",
    available_features: list[str] | None = None,
    required_fusions: list[str] | None = None,
    objective: str = "throughput",
) -> list[dict[str, Any]]:
    """Ranked retrieval (§3.1).

    Each result:
        {op_id, impl_card_id, backend, arch, applicable: bool,
         reject_reasons: [...], score: float, matched_measurement: {...}|None}
    Non-applicable candidates are included (with reject_reasons) so the caller —
    agent or human — sees *why*. When an op matches but no card applies to the
    target backend, the op appears with ``missing_backend: True``.
    """
    input_specs = input_specs or {}
    features = set(available_features or [])
    required_fusions = set(required_fusions or [])
    results: list[dict[str, Any]] = []

    for op in all_specs().values():
        # --- Op stage 1: canonical_op + fusions ---
        if op.canonical_op != canonical_op:
            continue
        missing_fusions = required_fusions - set(op.fusions)
        bindings, _ = _build_bindings(op, input_specs)
        op_reject = _constraint_reject_reasons(op, bindings)
        if missing_fusions:
            op_reject += [f"missing required fusion {f!r}" for f in sorted(missing_fusions)]
        op_applicable = not op_reject

        bucket = cards_for(op.id)
        if not bucket:
            results.append({
                "op_id": op.id, "impl_card_id": None, "backend": None,
                "arch": target_arch, "applicable": op_applicable,
                "reject_reasons": op_reject or ["no implementation cards registered"],
                "score": 0.0, "matched_measurement": None,
                "missing_backend": True,
            })
            continue

        # --- Impl stage ---
        any_card_applicable = False
        for card in bucket.values():
            reasons = list(op_reject)
            reasons += _arch_reject_reasons(card, target_arch, features)
            applicable = not reasons
            any_card_applicable = any_card_applicable or applicable
            matched = _matched_measurement(card, input_specs, target_arch)
            results.append({
                "op_id": op.id,
                "impl_card_id": card.id,
                "backend": card.backend.name,
                "arch": target_arch,
                "applicable": applicable,
                "reject_reasons": reasons,
                "score": _score(card, applicable, matched, objective),
                "matched_measurement": _measurement_to_dict(matched),
            })

        if op_applicable and not any_card_applicable:
            # op fits the contract but no card fits this backend/arch -> port trigger
            results[-1]["missing_backend"] = True

    results.sort(key=lambda r: (r["applicable"], r["score"]), reverse=True)
    return results


def _matched_measurement(card: ImplCard, input_specs: dict[str, dict], arch: str):
    """Find a measured entry matching the query's concrete (arch, shape, dtype)."""
    if not input_specs:
        return None
    # Try to read a concrete dtype from any input spec; presence of a concrete
    # shape is noted but measurements are matched on arch+dtype below.
    dtype: str | None = None
    for contract in input_specs.values():
        if isinstance(contract.get("shape"), list):
            break
    for contract in input_specs.values():
        if contract.get("dtype"):
            dtype = contract["dtype"]
            break
    if dtype is None:
        return None
    # The seed measurements key shapes by their symbolic names; without a symbol
    # binding here we can't match precisely, so match on arch+dtype only.
    for m in card.measured:
        if m.arch == arch and m.dtype == dtype:
            return m
    return None


def _measurement_to_dict(m) -> dict | None:
    if m is None:
        return None
    return {"arch": m.arch, "shape": dict(m.shape), "dtype": m.dtype,
            "tflops": m.tflops, "ms": m.ms, "achieved_bw_pct": m.achieved_bw_pct,
            "source": m.source}


def _score(card: ImplCard, applicable: bool, matched, objective: str) -> float:
    if not applicable:
        return 0.0
    score = 0.1
    if matched is not None:
        score = 1.0  # a concrete measured tuning is the strongest signal (§3.2.3)
    elif card.backend.name != "reference":
        score = 0.5  # an optimized backend without an exact measurement
    # objective alignment: memory_bound card is great for a memory objective
    if objective in ("memory", "bandwidth") and card.roofline == "memory_bound":
        score += 0.05
    if objective in ("throughput",) and card.roofline == "compute_bound":
        score += 0.05
    if objective == "latency" and card.backend.name == "reference":
        score -= 0.05  # reference is launch-heavy, bad for latency
    return round(score, 3)
