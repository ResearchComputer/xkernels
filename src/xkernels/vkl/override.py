# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Per-target override bodies — the native-ceiling path (docs/brainstorm/04 Ex.2, Axis H).

The portable ``@kernel`` body reaches Triton's ceiling; an override body reaches a
specific (backend, arch) ceiling using MORE NATIVE code (TMA + wgmma + clusters on
sm_90; MFMA on cdna3) — but it builds the SAME math IR, so the oracle property
holds: ``verify`` checks the override card against the SAME auto-reference (the
portable body on torch). The override is *more* native code, not a *different* op.

This module ships the **CPU-doable mechanism** for Phase 2.1:

  * ``check_override_math_ir(spec, override)`` — the load-bearing invariant: an
    override body must build the same math IR (same op kinds, same dtype
    contract) as the portable body. If it builds a *different* computation, it's
    not an override — it's a new op (route to ``author-an-op-spec``). This is the
    gate that makes the oracle property *enforced*, not hoped-for.
  * ``emit_override_card(spec, override)`` — project an override to its own Impl
    Card (one per ``(backend, arch)`` override; ``arch.requires`` carries the
    native features the override needs — tensor_cores/tma/clusters on sm_90,
    matrix_cores/mfma on cdna3).

The **GPU-gated half** (native CUDA/CUTE or HIP/CK codegen, the actual wgmma/MFMA
intrinsics, TMA descriptors, cluster launch) is *not* shipped here: it needs the
target arch's compiler (nvcc 12.x / hipcc), which is environment-blocked on the
current node. The decorator + the invariant check + card emission are the
foundation a GPU pass lands on top of; the override ``body`` is a trace-builder
whose lowering (``lower/cuda.py`` / ``lower/hip.py``) is the future Phase 2.1 GPU work.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Any

from .ir.math import MMA, Load, MathNode, Pointwise, Reduce, Store
from .lower.mathbody import MathBody, build_body
from .reference import _math_decls_for
from .surface import KernelSpec, OverrideBody

# ═══════════════════════════════════════════════════════════════════════════════
# §1  The math-IR-invariant check (the oracle property, enforced)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class OverrideCheck:
    """Verdict on whether an override body respects the oracle property."""

    ok: bool
    reason: str = ""
    portable_signature: tuple[str, ...] = ()  # the op-kind sequence of the portable body
    override_signature: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "portable_signature": list(self.portable_signature),
            "override_signature": list(self.override_signature),
        }


def _op_signature(nodes: tuple[MathNode, ...]) -> tuple[str, ...]:
    """The op-kind sequence of a math IR (Load/MMA/Reduce/Pointwise/Store order).

    Two bodies with the same signature compute the same algebra (same op kinds in
    the same dataflow order). This is the *structural* form of the oracle check:
    the override may spell each node differently (TMA load vs plain load, wgmma
    vs tl.dot) but it must perform the same *computation*. A signature mismatch
    means the override computes something different — not an override, a new op.
    """
    kinds = {"MMA": MMA, "Reduce": Reduce, "Pointwise": Pointwise, "Load": Load, "Store": Store}
    out: list[str] = []
    for node in nodes:
        for name, cls in kinds.items():
            if isinstance(node, cls):
                out.append(name)
                break
    return tuple(out)


def build_override_body(spec: KernelSpec, override: OverrideBody) -> MathBody:
    """Run the override's trace-builder to get its math IR (mirrors the portable)."""
    in_decls, out_decls = _math_decls_for(spec)
    return build_body(override.body, in_decls, out_decls)


def check_override_math_ir(spec: KernelSpec, override: OverrideBody) -> OverrideCheck:
    """The oracle-property gate: the override builds the same math IR as portable.

    Concretely, the override's op-kind signature must match the portable body's
    (same Load/MMA/Reduce/Pointwise/Store sequence). A mismatch means the override
    computes a *different* op — it should be a new Op Spec, not an override
    (route to ``author-an-op-spec``).

    This is the CPU-doable half of Phase 2.1: the invariant is checkable without
    a GPU. The dtype contract (MMA ``accum_dtype`` == numerics.reduce_dtype) is
    enforced by the math IR's own construction (``MathBodyCtx``).
    """
    if override.backend not in ("cuda", "hip"):
        return OverrideCheck(
            ok=False,
            reason=(
                f"override backend {override.backend!r} not a native target "
                f"(cuda/hip); the portable Triton body needs no override"
            ),
        )
    # Build both IRs and compare their op-kind signatures.
    from .reference import trace_ir

    portable = trace_ir(spec)
    if portable is None:
        return OverrideCheck(ok=False, reason="portable body has no @launch (not a trace)")
    try:
        ov = build_override_body(spec, override)
    except Exception as exc:  # noqa: BLE001
        return OverrideCheck(ok=False, reason=f"override body failed to build: {exc}")
    p_sig = _op_signature(portable.ir.nodes)
    o_sig = _op_signature(ov.ir.nodes)
    if p_sig != o_sig:
        return OverrideCheck(
            ok=False,
            reason=(
                f"math-IR signature mismatch: portable {p_sig} vs override {o_sig} "
                f"— the override computes a different op (route to author-an-op-spec)"
            ),
            portable_signature=p_sig,
            override_signature=o_sig,
        )
    return OverrideCheck(ok=True, portable_signature=p_sig, override_signature=o_sig)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Override card emission (one Impl Card per (backend, arch) override)
# ═══════════════════════════════════════════════════════════════════════════════


# The native features an override declares it needs (→ ``arch.requires``). These
# are the contract vocabulary the substrate already knows (impl_card.schema.json);
# the override just names which ones its native body uses.
_ARCH_REQUIRES = {
    ("cuda", "nvidia_sm90"): ["tensor_cores", "tma", "clusters"],
    ("cuda", "nvidia_sm80"): ["tensor_cores"],
    ("hip", "amd_cdna3"): ["matrix_cores", "mfma"],
    ("hip", "amd_cdna2"): ["matrix_cores", "mfma"],
}


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def emit_override_card(
    spec: KernelSpec,
    override: OverrideBody,
    *,
    created: str | None = None,
    knobs: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, Any]:
    """Project an override to its own Impl Card (the native-ceiling card).

    Mirrors ``emit.emit_card`` but for an override: the card's ``backend`` is the
    override's backend (cuda/hip), ``arch.family`` is the override's arch, and
    ``arch.requires`` carries the native features. The card is schema-valid (the
    same fields the portable card uses); provenance records ``derived_from`` the
    portable body (the override's contract is the spec's, its body is native).
    """
    from .emit import _DSL_SOURCE

    card_id = f"{spec.short_name}.{override.backend}@{spec.version}"
    requires = _ARCH_REQUIRES.get((override.backend, override.arch), [])
    wave_size = 32 if override.backend == "cuda" else 64
    scratch_kind = "smem" if override.backend == "cuda" else "lds"
    spec_knobs: dict[str, dict[str, Any]] = {}
    for name, choices in (knobs or {}).items():
        spec_knobs[name] = {
            "type": "int",
            "choices": list(choices),
            "_doc": f"vkl override knob for {override.backend}/{override.arch}",
        }
    return {
        "id": card_id,
        "implements": spec.id,
        "backend": override.backend,
        "arch": {
            "family": override.arch,
            "requires": requires,
            "wave_size": wave_size,
            "scratch": {"kind": scratch_kind, "bytes": 0},
        },
        "specialization_knobs": spec_knobs,
        "perf": {
            "regime": f"native override ({override.provenance_kind}) on {override.arch}",
            "roofline": "compute_bound",
            "measured": [],
        },
        "uses_primitives": requires,
        "supersedes": [],
        "provenance": {
            "authored_by": "dsl",
            "created": created or _now_iso(),
            "source_path": f"{_DSL_SOURCE}:{spec.short_name}#{override.backend}",
            "derived_from": f"{spec.short_name}.triton@{spec.version}",
        },
    }
