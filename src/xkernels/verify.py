"""Deterministic verification harness — the agent's correctness + parity surface.

Implements meta/docs/library.md §5. ``verify`` checks one Implementation Card against
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
from .registry.archs import vendor_of as _vendor_of
from .registry.input_gen import generate_inputs
from .registry.models import ImplCard, OpSpec
from .utils.benchmarking import benchmark

DEFAULT_SEED = 1729


def _as_device(arch: str, fallback: str = "cpu") -> str:
    """Map an arch id to a torch device. GPU archs -> 'cuda' if available.

    Honest about vendor mismatch: ``verify(card, arch='amd_cdna3')`` on an
    NVIDIA box still runs (everything is torch under the hood) but emits a
    warning, so the arch label never silently becomes a lie and an agent can
    tell its requested target wasn't honored.
    """
    if arch in ("any", ""):
        return fallback
    if torch.cuda.is_available():
        requested = _vendor_of(arch)
        if requested in ("amd", "nvidia"):
            detected = "nvidia"  # torch.cuda.is_available() is NVIDIA/ROCm-hipify
            if requested != detected and requested == "amd":
                import warnings
                warnings.warn(
                    f"arch {arch!r} is an AMD target but the available CUDA "
                    f"device is NVIDIA; running anyway (everything is torch). "
                    f"The arch label reflects the REQUESTED target, not the host.",
                    stacklevel=2,
                )
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


def _within_tolerance(
    actual: list[torch.Tensor], expected: list[torch.Tensor], rtol: float, atol: float
) -> bool:
    """Standard combined per-element criterion: ``|a - e| <= atol + rtol * |e|``.

    This is the form numpy ``allclose`` / torch ``assert_close`` / pytest
    ``approx`` all use: ``atol`` absorbs the near-zero regime (where relative
    error is ill-defined) and ``rtol`` scales with magnitude. The previous
    ``abs <= atol AND rel <= rtol`` form made ``atol`` a magnitude-independent
    absolute cap, which false-fails any non-bit-identical backend at moderate
    magnitude (e.g. 1 bf16-ULP at |e|=2 is 0.0156 > any reasonable atol) — and
    was inconsistent with ``verify_parity`` (which is rel-only). See
    meta/docs/usage/ds5-testbed.md follow-up (a) for the dual_rmsnorm case that forced
    this fix.
    """
    for a, e in zip(actual, expected, strict=True):
        af = a.detach().float()
        ef = e.detach().float()
        diff = (af - ef).abs()
        limit = atol + rtol * ef.abs()
        if bool((diff > limit).any().item()):
            return False
    return True


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
            passed = _within_tolerance(card_out, ref_out, rtol, atol)
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
    from .registry.cost_model import arch_peaks, cost_model

    inputs = generate_inputs(op.id, point, seed, device)
    fn = backend_callable(op.id, card.backend.name)
    if knobs:
        call = lambda: _apply_knobs(fn, inputs, knobs)[0]  # noqa: E731
    else:
        call = lambda: fn(**inputs)  # noqa: E731
    ms = benchmark(call)

    # Fill the two DERIVED metrics (tflops, achieved_bw_pct) when an analytical
    # cost model is registered for this op — the roofline signal an agent needs
    # to branch on memory- vs compute-bound WITHOUT leaving for an external
    # profiler. Without a model (or with a zero-ceiling arch like 'any'), the
    # metrics stay None honestly rather than being fabricated.
    tflops: float | None = None
    achieved_bw_pct: float | None = None
    note = "ms measured; no FLOP/byte model for this op (derived metrics None)"
    model = cost_model(op.id, point)
    if model is not None:
        flops, bytes_rw = model
        peaks = arch_peaks(_arch_of(op, card))
        peak_flops = peaks["fp32_tflops"]
        peak_bw = peaks["dram_bw_gbs"]
        if ms > 0 and flops > 0 and peak_flops > 0:
            tflops = round(flops / (ms * 1e-3) / 1e12, 3)
        if ms > 0 and bytes_rw > 0 and peak_bw > 0:
            achieved_bw_pct = round(bytes_rw / (ms * 1e-3) / 1e9 / peak_bw * 100, 2)
        note = (
            f"ms measured; tflops/bw derived from the op cost model "
            f"(flops={flops}, bytes={bytes_rw}) against arch "
            f"{_arch_of(op, card)!r} peaks (fp32={peak_flops}TF, BW={peak_bw}GB/s)."
        )
    return {"ms": ms, "tflops": tflops, "achieved_bw_pct": achieved_bw_pct, "note": note}


def _arch_of(op: OpSpec, card: ImplCard) -> str:
    """The arch to grade perf against: the card's family if concrete, else 'any'.

    A reference card (family 'any') has no meaningful peak ceiling, so its
    derived metrics stay None — only concrete-arch cards get a roofline grade.
    """
    fam = card.arch.family
    return fam if fam != "any" else "any"


def verify_parity(
    op_id: str,
    archs: list[str] | None = None,
    *,
    shapes: str | list[dict] | None = None,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
) -> dict[str, Any]:
    """Cross-backend parity gate (§5.3). For an op with >=2 cards, check that the
    backends agree with *each other* within ``cross_backend_rtol``.

    Args:
        op_id: the Op Spec id.
        archs: optional arch hint used to pick the device when ``device`` is
            None. The first arch whose vendor matches an available GPU wins;
            defaults to CPU when none do (back-compat).
        shapes: a sweep id (str) or explicit point list; defaults to the op's
            mandatory ``shape_sweep``.
        seed: deterministic input seed.
        device: override the device (``'cpu'`` | ``'cuda'``). When None it is
            derived from ``archs``: any GPU-capable arch -> ``'cuda'`` if
            available, else ``'cpu'``. This lets GPU-only cards (e.g. CUTE DSL)
            actually run in parity instead of being recorded as backend errors.

    Returns {agree, max_pairwise_rel_err, diverging, per_backend_runnable,
    n_runnable, inconclusive, errors}.

    ``agree`` is True only when >=2 backends actually ran and all pairs passed.
    When fewer than 2 ran, ``agree`` is None and ``inconclusive`` is True — a
    single runnable backend trivially agrees with itself and must NOT be
    reported as a passed parity gate (the honesty §10 demands).
    """
    from .registry import cards_for

    op = get_spec(op_id)
    bucket = cards_for(op_id)
    sweep_id = shapes if shapes is not None else op.shape_sweep
    points = _resolve_shapes(sweep_id)
    if device is None:
        device = _parity_device(archs)

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

    n_runnable = len(backends)
    inconclusive = n_runnable < 2
    # agree is meaningful only with >=2 runnable backends; else None (not True).
    agree: bool | None = None if inconclusive else (not diverging)
    return {
        "op_id": op_id,
        "cross_backend_rtol": op.numerics.cross_backend_rtol,
        "device": device,
        "agree": agree,
        "inconclusive": inconclusive,
        "n_runnable": n_runnable,
        "max_pairwise_rel_err": max_pairwise,
        "diverging": diverging,
        "per_backend_runnable": {k: (k in outputs) for k in bucket},
        "errors": errors,
        "n_points": len(points),
    }


def _parity_device(archs: list[str] | None) -> str:
    """Pick a parity device from an optional arch list.

    Any GPU-capable arch -> 'cuda' if a CUDA device is available, else 'cpu'.
    The default (no archs, or vendor-agnostic archs) stays 'cpu' so the
    back-compat invariant holds: ``verify_parity`` on a CPU-only box behaves
    exactly as before.
    """
    if not archs:
        return "cpu"
    if any(_vendor_of(a) in ("amd", "nvidia") for a in archs) and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run_id(card: ImplCard, arch: str, seed: int, sweep_id: str | list) -> str:
    h = hashlib.sha1(f"{card.id}|{arch}|{seed}|{sweep_id}".encode()).hexdigest()[:12]
    return f"run:{h}"
