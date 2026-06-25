"""Deterministic verification harness — the agent's correctness + parity surface.

Implements docs/library.md §5. ``verify`` checks one Implementation Card against
its Op Spec's single backend-neutral reference and tolerances; ``verify_parity``
checks that >=2 backends agree with *each other* within ``cross_backend_rtol``.

Output is a parseable blob (no raw profiler text the model has to squint at).
Correctness is checked first; perf is opt-in (``measure_perf=True``) and separate
— mirroring the agent loop §6.1 ("correctness first, then perf").
"""
from __future__ import annotations

import hashlib
import traceback
from collections.abc import Callable
from typing import Any

import torch

from .registry import backend_callable, get_card, get_spec, load_shape_sweep, reference_callable
from .registry.input_gen import generate_inputs
from .registry.models import ImplCard, OpSpec
from .utils.benchmarking import benchmark

DEFAULT_SEED = 1729


def _as_device(arch: str, fallback: str = "cpu") -> str:
    """Map an arch id to a torch device. GPU archs -> 'cuda' if available."""
    if arch in ("any", ""):
        return fallback
    if torch.cuda.is_available():
        return "cuda"
    return fallback


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, (tuple, list)):
        return [t for t in out if isinstance(t, torch.Tensor)]
    if isinstance(out, torch.Tensor):
        return [out]
    raise TypeError(f"unsupported op output type {type(out).__name__}")


def _accepted_knobs(fn: Callable[..., Any], inputs: dict[str, Any]) -> tuple[set[str], bool]:
    """Inspect ``fn``'s signature: which knob names may be passed as kwargs?

    Returns (accepted_names, accepts_arbitrary_kwargs). A knob is accepted if it
    names a keyword-or-positional parameter not already in ``inputs``, or if the
    callable declares ``**kwargs`` (then any knob is accepted).
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return set(), False
    accepted: set[str] = set()
    has_kwargs = False
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            has_kwargs = True
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if name not in inputs:
            accepted.add(name)
    return accepted, has_kwargs


def _apply_knobs(
    fn: Callable[..., Any], inputs: dict[str, Any], knobs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Call ``fn(**inputs, **<knobs it accepts>)``.

    Returns (output, knobs_applied). Knobs the callable does not accept are
    silently dropped (and reported back by the caller as requested-but-not-applied,
    so the agent knows specialization didn't take effect — the honesty §10 demands).
    """
    accepted, has_kwargs = _accepted_knobs(fn, inputs)
    if has_kwargs:
        applicable = dict(knobs)
    else:
        applicable = {k: v for k, v in knobs.items() if k in accepted}
    return fn(**inputs, **applicable), applicable


def _errors(actual: list[torch.Tensor], expected: list[torch.Tensor]) -> tuple[float, float]:
    max_abs = 0.0
    max_rel = 0.0
    for a, e in zip(actual, expected, strict=True):
        af = a.detach().float()
        ef = e.detach().float()
        diff = (af - ef).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        denom = ef.abs().clamp_min(1e-8)
        rel = (diff / denom).max().item()
        max_rel = max(max_rel, float(rel))
    return max_abs, max_rel


def _resolve_shapes(shapes: str | list[dict]) -> list[dict]:
    if isinstance(shapes, str):
        return load_shape_sweep(shapes)
    if isinstance(shapes, list):
        return shapes
    raise TypeError("shapes must be a sweep id (str) or a list of point dicts")


def _run(card: ImplCard, op: OpSpec, point: dict, seed: int, device: str,
         knobs: dict[str, Any] | None = None) -> Any:
    """Run the card's backend callable on generated inputs, applying ``knobs``
    the callable actually accepts (signature-aware specialization)."""
    inputs = generate_inputs(op.id, point, seed, device)
    fn = backend_callable(op.id, card.backend.name)
    if knobs:
        return _apply_knobs(fn, inputs, knobs)[0]
    return fn(**inputs)


def _run_reference(op: OpSpec, point: dict, seed: int, device: str) -> Any:
    inputs = generate_inputs(op.id, point, seed, device)
    return reference_callable(op.id)(**inputs)


def verify(
    impl_card_id: str,
    arch: str = "any",
    *,
    knobs: dict[str, Any] | None = None,
    shapes: str | list[dict] | None = None,
    seed: int = DEFAULT_SEED,
    measure_perf: bool = False,
) -> dict[str, Any]:
    """Verify one card against its Op Spec reference + tolerances (§5.2).

    Args:
        impl_card_id: e.g. "fused_ffn.triton@1.0.0".
        arch: target arch id; selects the device when a GPU arch is given.
        knobs: requested specialization knobs (recorded; per-kernel plumbing is
            per-card work — see Impl Card specialization_knobs).
        shapes: a sweep id (str) or explicit point list; defaults to the op's
            mandatory ``shape_sweep``.
        seed: deterministic input seed.
        measure_perf: if True, also time the card and fill ``perf``.

    Returns a structured blob:
        {compiled, correctness: {passed, max_abs_err, max_rel_err,
         failing_shapes, n_points}, determinism_check, knobs_requested,
         perf: {ms, tflops, achieved_bw_pct}, artifacts: {run_id, error?}}
    """
    card = get_card(impl_card_id)
    op = get_spec(card.implements)
    sweep_id = shapes if shapes is not None else op.shape_sweep
    points = _resolve_shapes(sweep_id)
    device = _as_device(arch)
    knobs = knobs or {}

    compiled = True
    error: str | None = None
    failing: list[dict] = []
    max_abs = 0.0
    max_rel = 0.0
    n_points = len(points)

    try:
        ref_outputs = []
        card_outputs = []
        knobs_applied: dict[str, Any] = {}
        for p in points:
            ref_out = _normalize_outputs(_run_reference(op, p, seed, device))
            ref_outputs.append(ref_out)
            out, applied = _apply_knobs(
                backend_callable(op.id, card.backend.name),
                generate_inputs(op.id, p, seed, device),
                knobs,
            )
            knobs_applied = applied  # same signature every point; last wins
            card_outputs.append(_normalize_outputs(out))

        # Determinism: re-run the first point, compare to the stored run.
        determinism_check = True
        try:
            again = _normalize_outputs(_run(card, op, points[0], seed, device, knobs))
            _, rel = _errors(again, card_outputs[0])
            determinism_check = rel <= op.numerics.rtol
        except Exception:
            determinism_check = False

        for p, ref_out, card_out in zip(points, ref_outputs, card_outputs, strict=True):
            dtype_short = p.get("dtype", "fp32")
            rtol, atol = op.numerics.tolerance_for(dtype_short)
            abs_err, rel_err = _errors(card_out, ref_out)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            passed = (abs_err <= atol) and (rel_err <= rtol)
            if not passed:
                failing.append({"point": p, "abs_err": abs_err, "rel_err": rel_err,
                                "rtol": rtol, "atol": atol})
        correctness_passed = not failing
    except Exception as e:  # backend missing / compile error / runtime error
        compiled = False
        error = f"{type(e).__name__}: {e}"
        correctness_passed = False
        determinism_check = False

    run_id = _run_id(card, arch, seed, sweep_id)
    result: dict[str, Any] = {
        "impl_card_id": impl_card_id,
        "implements": card.implements,
        "backend": card.backend.name,
        "arch": arch,
        "compiled": compiled,
        "correctness": {
            "passed": correctness_passed,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "failing_shapes": failing,
            "n_points": n_points,
        },
        "determinism_check": determinism_check,
        "knobs_requested": knobs,
        "knobs_applied": knobs_applied,
        "knobs_unapplied": sorted(set(knobs) - set(knobs_applied)),
        "artifacts": {"run_id": run_id},
    }
    if error is not None:
        result["artifacts"]["error"] = error
        result["artifacts"]["traceback"] = traceback.format_exc(limit=4)

    if measure_perf and compiled and correctness_passed and device != "cpu":
        result["perf"] = _measure_perf(card, op, points[-1], seed, device, knobs)
    else:
        result["perf"] = {"ms": None, "tflops": None, "achieved_bw_pct": None,
                          "note": "perf not measured (set measure_perf=True on a GPU device)"}
    return result


def _measure_perf(
    card: ImplCard, op: OpSpec, point: dict, seed: int, device: str,
    knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inputs = generate_inputs(op.id, point, seed, device)
    fn = backend_callable(op.id, card.backend.name)
    if knobs:
        call = lambda: _apply_knobs(fn, inputs, knobs)[0]  # noqa: E731
    else:
        call = lambda: fn(**inputs)  # noqa: E731
    ms = benchmark(call)
    note = ("ms measured via do_bench; tflops/bw need an op-specific "
            "FLOP/byte model (open question §11)")
    return {"ms": ms, "tflops": None, "achieved_bw_pct": None, "note": note}


def verify_parity(
    op_id: str,
    archs: list[str] | None = None,
    *,
    shapes: str | list[dict] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Cross-backend parity gate (§5.3). For an op with >=2 cards, check that the
    backends agree with *each other* within ``cross_backend_rtol``.

    Returns {agree, max_pairwise_rel_err, diverging: [...], per_backend: {...}}.
    A card that fails parity cannot publish (§2.4).
    """
    from .registry import cards_for

    op = get_spec(op_id)
    bucket = cards_for(op_id)
    sweep_id = shapes if shapes is not None else op.shape_sweep
    points = _resolve_shapes(sweep_id)
    device = "cpu"

    # Collect per-backend, per-point outputs.
    outputs: dict[str, list[list[torch.Tensor]]] = {}
    errors: dict[str, str] = {}
    for backend_name, card in bucket.items():
        outs = []
        try:
            for p in points:
                outs.append(_normalize_outputs(_run(card, op, p, seed, device)))
            outputs[backend_name] = outs
        except Exception as e:  # backend not runnable here
            errors[backend_name] = f"{type(e).__name__}: {e}"

    backends = list(outputs)
    max_pairwise = 0.0
    diverging: list[dict] = []
    for i in range(len(backends)):
        for j in range(i + 1, len(backends)):
            a, b = backends[i], backends[j]
            for idx, (oa, ob) in enumerate(zip(outputs[a], outputs[b], strict=True)):
                _, rel = _errors(oa, ob)
                max_pairwise = max(max_pairwise, rel)
                if rel > op.numerics.cross_backend_rtol:
                    diverging.append({"pair": [a, b], "point": points[idx], "rel_err": rel})

    return {
        "op_id": op_id,
        "cross_backend_rtol": op.numerics.cross_backend_rtol,
        "agree": not diverging,
        "max_pairwise_rel_err": max_pairwise,
        "diverging": diverging,
        "per_backend_runnable": {k: (k in outputs) for k in bucket},
        "errors": errors,
        "n_points": len(points),
    }


def _run_id(card: ImplCard, arch: str, seed: int, sweep_id: str | list) -> str:
    h = hashlib.sha1(f"{card.id}|{arch}|{seed}|{sweep_id}".encode()).hexdigest()[:12]
    return f"run:{h}"
