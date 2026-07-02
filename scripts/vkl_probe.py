#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 0 kill-experiment for the ``vkl`` DSL  (docs/brainstorm/11 §7).

This is NOT the DSL. It is a hand-driven probe that verifies the
*contract-native thesis* before any lowering is built:

  (A) A ``@kernel``-header-shaped Python object can EMIT an Op Spec dict that the
      existing substrate accepts as-is — real JSON-Schema validation
      (``validate_op_spec``), decidable constraints (``validate_decidable``), and
      ingest (``op_spec_from_doc``). If this holds, the header is genuinely a
      *spelling* of the spec, not a parallel vocabulary (docs/brainstorm/02 §1).
  (B) Three Impl Cards (triton/cuda/hip) for one GEMM pass ``validate_impl_card``
      + ``ImplCard.from_doc`` — proving the multi-target emission target
      (docs/brainstorm/02 Layer 3) lands in fields the schema already knows.
  (C) The SCHEDULE-IR edit preconditions from docs/brainstorm/09 §8 are LOCALLY
      DECIDABLE — i.e. each edit (retile/map_to/add_stage/set_knob) is checkable
      from the edit args + a schedule-IR dict + an arch dict alone, with no
      global reasoning and no running code. This is the bet that makes an LLM
      agent a reliable editor of the IR (docs/brainstorm/09 §0).
  (D) A SCHEMA FINDING: the closed schema (``additionalProperties: false`` on
      ``provenance``, ``authored_by`` enum lacks ``"dsl"``) REJECTS the
      namespaced extensions the plan wants (``tuning_trace``, ``launch.graph``).
      This confirms the schema edits in docs/brainstorm/11 §0 are required, and
      is itself a recorded Phase-0 outcome.

Run directly for a human-readable report:

    python scripts/vkl_probe.py

The pytest wrapper (``tests/test_vkl_probe.py``) asserts every check passes, so
the thesis is CI-enforced the moment it lands. If any check fails, that is the
information Phase 0 exists to produce — execute the scope-correction in
docs/brainstorm/06 §D *before* building the lowering.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# --- substrate imports (the real validators; the probe does not mock them) ----
from xkernels.registry.constraints import (
    UndecidableConstraintError,
    validate_decidable,
)
from xkernels.registry.models import ImplCard, op_spec_from_doc
from xkernels.registry.schemas import (
    have_validator,
    validate_impl_card,
    validate_op_spec,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — a minimal kernel-header (honest seed of future vkl/surface.py).
# These dataclasses deliberately mirror the Op Spec fields 1:1 (library.md §2.1)
# so emit_spec() below is a pure projection, not a translation.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TensorDecl:
    """Mirror of the schema's tensorContract (op_spec.schema.json $defs)."""

    dtype: tuple[str, ...]
    rank: int
    shape_symbols: tuple[str, ...]
    layout: str = "row_major"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": list(self.dtype),
            "rank": self.rank,
            "shape_symbols": list(self.shape_symbols),
            "layout": self.layout,
        }


@dataclass(frozen=True)
class NumericsDecl:
    """Mirror of the schema's numerics block."""

    reference: str
    rtol: float
    atol: float
    reduce_dtype: str | None = None
    cross_backend_rtol: float | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "reference": self.reference,
            "rtol": self.rtol,
            "atol": self.atol,
        }
        if self.reduce_dtype is not None:
            d["reduce_dtype"] = self.reduce_dtype
        if self.cross_backend_rtol is not None:
            d["cross_backend_rtol"] = self.cross_backend_rtol
        if self.notes is not None:
            d["notes"] = self.notes
        return d


@dataclass(frozen=True)
class KernelHeader:
    """The ``@kernel(...)`` declarative header — a spelling of the Op Spec.

    Every field here has a 1:1 target in op_spec.schema.json. emit_spec() is the
    projection. If the projection ever needs a "translation" (not a copy), the
    contract-native thesis (docs/brainstorm/02 §1) is broken — and that break
    shows up as a failed round-trip in check (A).
    """

    id: str
    name: str
    version: str
    kernel: str
    signature: str
    canonical_op: str
    inputs: dict[str, TensorDecl]
    outputs: dict[str, TensorDecl]
    constraints: tuple[str, ...]
    numerics: NumericsDecl
    shape_sweep: str
    fusions: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — emit_spec: header -> schema-valid Op Spec dict (seed of emit.py).
# ═══════════════════════════════════════════════════════════════════════════════


def emit_spec(h: KernelHeader) -> dict[str, Any]:
    """Project a KernelHeader to a schema-valid Op Spec dict.

    Pure function; no side effects. The output MUST be accepted by
    validate_op_spec + validate_decidable + op_spec_from_doc (check A).
    """
    return {
        "id": h.id,
        "name": h.name,
        "version": h.version,
        "kernel": h.kernel,
        "op": {
            "signature": h.signature,
            "canonical_op": h.canonical_op,
            "fusions": list(h.fusions),
        },
        "inputs": {k: v.to_dict() for k, v in h.inputs.items()},
        "outputs": {k: v.to_dict() for k, v in h.outputs.items()},
        "constraints": list(h.constraints),
        "preconditions": list(h.preconditions),
        "numerics": h.numerics.to_dict(),
        "shape_sweep": h.shape_sweep,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — the hand-built example: a multi-target bf16 GEMM.
# This is the op docs/brainstorm/04 Ex.2 sketches; here it is concrete.
# ═══════════════════════════════════════════════════════════════════════════════

_GEMM_ID = "vkl_probe_gemm_bf16@1.0.0"


def gemm_bf16_header() -> KernelHeader:
    """The contract for a dense bf16 GEMM with fp32 accumulation."""
    return KernelHeader(
        id=_GEMM_ID,
        name="vkl probe dense bf16 GEMM",
        version="1.0.0",
        kernel="vkl_probe_gemm",  # dispatch key (probe; no callable registered)
        signature="C = A @ B  (bf16 in, fp32 accumulate)",
        canonical_op="gemm",
        inputs={
            "a": TensorDecl(("bf16", "fp16"), rank=2, shape_symbols=("M", "K")),
            "b": TensorDecl(("bf16", "fp16"), rank=2, shape_symbols=("K", "N")),
        },
        outputs={
            "c": TensorDecl(("bf16", "fp16"), rank=2, shape_symbols=("M", "N")),
        },
        constraints=(
            "dtype(a) == dtype(b)",
            "K % 8 == 0",
        ),
        numerics=NumericsDecl(
            reference="xkernels.ops.gemm.reference:gemm_ref",  # probe; need not resolve
            rtol=1.6e-2,
            atol=1e-2,
            reduce_dtype="fp32",
            cross_backend_rtol=2e-2,
            notes="Accumulate in fp32 (bf16/fp16 inputs).",
        ),
        shape_sweep="vkl_probe_gemm_bf16",  # probe; sweep file need not exist for check (A)
    )


def gemm_bf16_cards() -> list[dict[str, Any]]:
    """Three Impl Cards for the GEMM: triton (portable) + cuda (sm_90) + hip (cdna3).

    Each card uses the EXACT arch vocabulary the schema knows
    (arch.family/requires/wave_size/scratch.kind), proving the multi-target
    @targets block (docs/brainstorm/02 Layer 3) projects into existing fields.
    The cuda card carries the sm_90 matrix-engine requirements a per-target
    override would declare (tensor_cores + tma + clusters); the hip card carries
    the CDNA3 ones (matrix_cores + mfma).
    """
    impl = _GEMM_ID
    base_knobs = {
        "BLOCK_M": {"type": "int", "choices": [64, 128, 256]},
        "BLOCK_N": {"type": "int", "choices": [64, 128, 256]},
        "BLOCK_K": {"type": "int", "choices": [32, 64]},
    }
    return [
        {  # portable triton card (arch: any) — the auto-reference companion
            "id": "vkl_probe_gemm_bf16.triton@1.0.0",
            "implements": impl,
            "backend": "triton",
            "arch": {
                "family": "any",
                "requires": [],
                "wave_size": 0,
                "scratch": {"kind": "registers", "bytes": 0},
            },
            "specialization_knobs": {
                **base_knobs,
                "num_stages": {"type": "int", "choices": [2, 3, 4]},
                "num_warps": {"type": "int", "choices": [4, 8]},
            },
            "perf": {
                "regime": "portable baseline; ceiling reached via per-target overrides",
                "roofline": "compute_bound",
                "measured": [],
            },
            "uses_primitives": ["mma.portable", "stage_async.global_to_register"],
            "supersedes": [],
            "provenance": {
                "authored_by": "agent",  # "dsl" rejected today — see check (D)
                "skill_used": ["tile-a-gemm"],
                "created": "2026-06-30T00:00:00Z",
                "source_path": "scripts/vkl_probe.py",
            },
        },
        {  # native NVIDIA sm_90 ceiling card (per-target override target)
            "id": "vkl_probe_gemm_bf16.cuda@1.0.0",
            "implements": impl,
            "backend": "cuda",
            "arch": {
                "family": "nvidia_sm90",
                "requires": ["tensor_cores", "tma", "clusters"],
                "wave_size": 32,
                "scratch": {"kind": "smem", "bytes": 196608},
            },
            "specialization_knobs": {
                "BLOCK_M": {"type": "int", "choices": [128, 256]},
                "BLOCK_N": {"type": "int", "choices": [128, 256]},
                "stages": {"type": "int", "choices": [3, 4]},
            },
            "perf": {
                "regime": "sm_90 ceiling: TMA descriptors + thread-block clusters + wgmma",
                "roofline": "compute_bound",
                "measured": [],
            },
            "uses_primitives": ["wgmma.sm90", "tma_descriptor", "cluster"],
            "supersedes": [],
            "provenance": {
                "authored_by": "agent",
                "skill_used": ["map-to-matrix-cores", "tune-for-cdna"],
                "created": "2026-06-30T00:00:00Z",
                "source_path": "scripts/vkl_probe.py",
            },
        },
        {  # native AMD CDNA3 ceiling card (per-target override target)
            "id": "vkl_probe_gemm_bf16.hip@1.0.0",
            "implements": impl,
            "backend": "hip",
            "arch": {
                "family": "amd_cdna3",
                "requires": ["matrix_cores", "mfma"],
                "wave_size": 64,
                "scratch": {"kind": "lds", "bytes": 65536},
            },
            "specialization_knobs": {
                "BLOCK_M": {"type": "int", "choices": [128, 256]},
                "waves_per_eu": {"type": "int", "choices": [1, 2]},
                "stages": {"type": "int", "choices": [3, 4]},
            },
            "perf": {
                "regime": "CDNA3 ceiling: MFMA + global->LDS DMA, 64-wide wavefronts",
                "roofline": "compute_bound",
                "measured": [],
            },
            "uses_primitives": ["mfma.cdna3", "global_to_lds_dma"],
            "supersedes": [],
            "provenance": {
                "authored_by": "agent",
                "skill_used": ["map-to-matrix-cores", "tune-for-cdna"],
                "created": "2026-06-30T00:00:00Z",
                "source_path": "scripts/vkl_probe.py",
            },
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — minimal arch database (seed of vkl/archdb.py).
# Numbers from registry/cost_model.py arch peaks + docs/brainstorm/10 §4.1.
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_DB: dict[str, dict[str, Any]] = {
    "nvidia_sm90": {
        "wave_size": 32,
        "scratch_bytes": 228 * 1024,  # shared mem per CTA, H100
        "native_shapes": {"wgmma": {"m": 64, "k": 16}},
        "legal_instructions": {"fma", "wmma", "wgmma"},
    },
    "amd_cdna3": {
        "wave_size": 64,
        "scratch_bytes": 64 * 1024,  # LDS per workgroup, MI300A
        "native_shapes": {"mfma": {"m": 32, "k": 16}},
        "legal_instructions": {"fma", "mfma"},
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5 — minimal schedule-IR shape + edit gate (seed of vkl/ir + vkl/gate).
# The gate mirrors docs/brainstorm/10 §5: each rule traces to a §10 anti-goal.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Schedule:
    """A minimal editable schedule IR (docs/brainstorm/10 §2 shape).

    Only the fields the gate needs to decide the probe's edits. Phase 1 grows
    this into the full node set; here it is just enough to test decidability.
    """

    tiles: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {shape, level}
    maps: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {op, level, instr}
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {space, depth}
    knobs: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> {value, choices}

    def scratch_total(self) -> int:
        return sum(s["tile_bytes"] * s["depth"] for s in self.stages.values())


def _ok() -> tuple[bool, str]:
    return True, "ok"


def _reject(reason: str) -> tuple[bool, str]:
    return False, reason


def gate_retile(sched: Schedule, arch: str, tile_id: str, new_shape: list[int]) -> tuple[bool, str]:
    """Tile shape must be divisible by the target's L5 native shape (docs/brainstorm/08 §2.1).

    Stateful: the constraint only bites once an L5 matrix-engine map is present
    in the schedule. This is the load-bearing property of edit decidability —
    a gate is a pure function of *(edit args, current IR, arch)*, so the caller
    must hand it the IR state where the constraint applies.
    """
    db = ARCH_DB[arch]
    native = None
    for m in sched.maps.values():
        if m.get("level") == "L5" and m.get("instruction") in db["native_shapes"]:
            native = db["native_shapes"][m["instruction"]]
            break
    if native is None:
        return _ok()  # no L5 engine mapped yet → no divisibility constraint
    if len(new_shape) < 1:
        return _reject(f"tile {tile_id!r}: need >=1 dim, got {new_shape}")
    m_dim = native["m"]
    # Convention: shape[0] is the output M dim; L5 native m must divide it.
    if new_shape[0] % m_dim != 0:
        return _reject(
            f"tile {tile_id!r} M={new_shape[0]} not divisible by L5 native m={m_dim}"
        )
    return _ok()


def gate_map_to(
    sched: Schedule, arch: str, map_id: str, instruction: str, instr_shape: tuple[int, ...]
) -> tuple[bool, str]:
    """Instruction must be legal for the arch; shape must divide the L2 tile."""
    db = ARCH_DB[arch]
    if instruction not in db["legal_instructions"]:
        return _reject(
            f"instruction {instruction!r} not legal for {arch} "
            f"(have {sorted(db['legal_instructions'])})"
        )
    native = db["native_shapes"].get(instruction)
    if native is None:
        return _ok()  # scalar fma: no native shape constraint
    # Find the L2 tile this map feeds; check divisibility.
    # (In the probe the map feeds the output tile of the GEMM.)
    m_dim = native["m"]
    for t in sched.tiles.values():
        if t.get("level") == "L2" and t["shape"][0] % m_dim != 0:
            return _reject(
                f"L2 tile M={t['shape'][0]} not divisible by {instruction} m={m_dim}"
            )
    return _ok()


def gate_add_stage(
    sched: Schedule, arch: str, stage_id: str, depth: int, tile_bytes: int
) -> tuple[bool, str]:
    """Pipeline depth must fit the arch scratch budget (docs/brainstorm/10 §5)."""
    budget = ARCH_DB[arch]["scratch_bytes"]
    tentative = sched.scratch_total() + tile_bytes * depth
    if tentative > budget:
        return _reject(
            f"stage {stage_id!r}: scratch {tentative} B > {arch} budget {budget} B"
        )
    return _ok()


def gate_set_knob(sched: Schedule, name: str, value: int) -> tuple[bool, str]:
    """Knob value must be inside the declared specialization space (docs/brainstorm/10 §5)."""
    knob = sched.knobs.get(name)
    if knob is None:
        return _reject(f"undeclared knob {name!r}")
    if value not in knob["choices"]:
        return _reject(f"knob {name!r}: {value} not in declared choices {knob['choices']}")
    return _ok()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6 — the check runners. Each returns a list of (name, passed, detail).
# ═══════════════════════════════════════════════════════════════════════════════


def check_roundtrip_spec() -> list[tuple[str, bool, str]]:
    """(A) header -> emit -> validate -> ingest -> field-equal."""
    out: list[tuple[str, bool, str]] = []
    h = gemm_bf16_header()
    doc = emit_spec(h)

    # (A1) real JSON-Schema validation against the shipped op_spec schema.
    try:
        validate_op_spec(doc)
        out.append(("A1 spec schema-valid", True, "validate_op_spec accepted the emitted dict"))
    except Exception as e:  # noqa: BLE001
        out.append(("A1 spec schema-valid", False, f"{type(e).__name__}: {e}"))
        return out  # later steps assume validity

    # (A2) every constraint is in the decidable subset (reject-before-compile, §2.4).
    for c in h.constraints:
        try:
            validate_decidable(c)
            out.append((f"A2 decidable {c!r}", True, "validate_decidable ok"))
        except UndecidableConstraintError as e:
            out.append((f"A2 decidable {c!r}", False, str(e)))

    # (A3) ingest via the real dataclass constructor; field-equality vs the header.
    try:
        spec = op_spec_from_doc(doc)
    except Exception as e:  # noqa: BLE001
        out.append(("A3 op_spec_from_doc", False, f"{type(e).__name__}: {e}"))
        return out
    ok = (
        spec.id == h.id
        and spec.kernel == h.kernel
        and spec.canonical_op == h.canonical_op
        and spec.constraints == h.constraints
        and spec.numerics.reduce_dtype == h.numerics.reduce_dtype
        and spec.numerics.cross_backend_rtol == h.numerics.cross_backend_rtol
    )
    out.append((
        "A3 ingest field-equal",
        ok,
        f"id={spec.id} kernel={spec.kernel} constraints={spec.constraints}",
    ))

    # (A4) byte-identical round-trip: emit -> JSON -> parse -> ingest.doc -> JSON.
    emitted = json.dumps(doc, sort_keys=True, indent=2)
    retrip = json.dumps(spec.doc, sort_keys=True, indent=2)
    out.append(("A4 byte-identical round-trip", emitted == retrip,
                "emit(doc) == op_spec_from_doc(doc).doc"))
    return out


def check_cards_ingest() -> list[tuple[str, bool, str]]:
    """(B) three Impl Cards pass validate_impl_card + ImplCard.from_doc."""
    out: list[tuple[str, bool, str]] = []
    for card in gemm_bf16_cards():
        label = card["id"]
        try:
            validate_impl_card(card)
        except Exception as e:  # noqa: BLE001
            out.append((f"B {label} schema", False, f"{type(e).__name__}: {e}"))
            continue
        try:
            c = ImplCard.from_doc(card)
        except Exception as e:  # noqa: BLE001
            out.append((f"B {label} from_doc", False, f"{type(e).__name__}: {e}"))
            continue
        out.append((
            f"B {label}",
            True,
            f"backend={c.backend.name} arch={c.arch.family} "
            f"requires={c.arch.requires} wave={c.arch.wave_size} "
            f"scratch={c.arch.scratch.get('kind')}",
        ))
    return out


def check_edit_decidability() -> list[tuple[str, bool, str]]:
    """(C) the docs/brainstorm/09 §8 edit trace is locally decidable.

    Each edit is checked by a pure function of (edit args, schedule dict, arch).
    We assert both the ACCEPTED edits (preconditions pass) and the REJECTED edits
    (preconditions fail with a reason) — the reasons are the training signal.
    """
    out: list[tuple[str, bool, str]] = []

    # Starting schedule for the portable GEMM on sm_90 (pre-override).
    sched = Schedule(
        tiles={
            "out": {"shape": [128, 128], "level": "L2"},
            "k": {"shape": [128, 64], "level": "L2"},
        },
        maps={},
        stages={"a": {"space": "scratch", "depth": 2, "tile_bytes": 32 * 1024}},
        knobs={
            "num_stages": {"value": 2, "choices": [2, 3, 4]},
            "BLOCK_M": {"value": 128, "choices": [128, 256]},
        },
    )
    arch = "nvidia_sm90"

    # The post-map schedule: the wgmma L5 map is now present. This is the IR
    # state AFTER step C2, and it's what makes the retile divisibility gate
    # (which is stateful) actually bite. Edits are pure functions of the IR —
    # so we hand each gate the IR state where its precondition applies.
    sched_post_map = Schedule(
        tiles=sched.tiles,
        maps={"mma0": {"op": "mma", "level": "L5", "instruction": "wgmma",
                       "instr_shape": (64, 128, 16)}},
        stages=sched.stages,
        knobs=sched.knobs,
    )

    # --- accepted edits (the 09 §8 winning trace) -----------------------------
    ok, why = gate_add_stage(sched, arch, "b", depth=3, tile_bytes=32 * 1024)
    # 2 existing (a) + new (b): 32K*2 + 32K*3 = 160K < 228K budget → ok
    out.append(("C1 add_stage(depth=3) fits", ok, why))

    ok, why = gate_map_to(sched, arch, "mma0", "wgmma", (64, 128, 16))
    # wgmma legal on sm_90; out tile M=128 % 64 == 0 → ok
    out.append(("C2 map_to(wgmma) legal", ok, why))

    ok, why = gate_retile(sched_post_map, arch, "out", [256, 128])
    # wgmma mapped; M=256 % 64 == 0 → ok (divisibility now bites, correctly)
    out.append(("C3 retile(M=256) divisible", ok, why))

    ok, why = gate_set_knob(sched, "num_stages", 3)
    out.append(("C4 set_knob(num_stages=3) in choices", ok, why))

    # --- rejected edits (the dead-ends the trace records) ---------------------
    ok, why = gate_retile(sched_post_map, arch, "out", [96, 128])
    out.append((
        "C5 retile(M=96) REJECTED (not divisible by 64)",
        (not ok) and "m=64" in why,
        why,
    ))

    sched_bad_knob = Schedule(knobs={"num_stages": {"value": 2, "choices": [2, 3, 4]}})
    ok, why = gate_set_knob(sched_bad_knob, "num_stages", 5)
    out.append((
        "C6 set_knob(num_stages=5) REJECTED (out of choices)",
        (not ok) and "5" in why,
        why,
    ))

    ok, why = gate_map_to(sched, arch, "mma0", "mfma", (32, 128, 16))
    out.append((
        "C7 map_to(mfma) on sm_90 REJECTED (wrong vendor)",
        (not ok) and "mfma" in why,
        why,
    ))

    # Scratch overflow: fill the budget then add one more stage.
    sched_full = Schedule(stages={
        "big": {"space": "scratch", "depth": 4, "tile_bytes": 60 * 1024},  # 240K > 228K
    })
    ok, why = gate_add_stage(sched_full, arch, "x", depth=1, tile_bytes=4 * 1024)
    out.append(("C8 add_stage REJECTED (scratch overflow)", (not ok) and "budget" in why, why))

    return out


def check_schema_finding() -> list[tuple[str, bool, str]]:
    """(D) the namespaced schema extensions Phase 1 requires.

    **Phase 0 found** the closed schema rejected ``authored_by="dsl"``,
    ``provenance.tuning_trace``, and ``launch.graph`` (additionalProperties:false
    on provenance + the card). **Phase 1 landed those schema edits**
    (impl_card.schema.json: 'dsl' enum + tuning_trace property + launch object),
    so this check now asserts the extensions are ACCEPTED — the finding is
    resolved and the vkl emitter can use them.
    """
    out: list[tuple[str, bool, str]] = []
    base = gemm_bf16_cards()[1]  # the cuda card

    # D1: authored_by="dsl" is now ACCEPTED (Phase 1 added it to the enum).
    d1 = {**base, "provenance": {**base["provenance"], "authored_by": "dsl"}}
    accepted = True
    msg = "accepted"
    try:
        validate_impl_card(d1)
    except Exception as e:  # noqa: BLE001
        accepted = False
        msg = f"{type(e).__name__}: {e}"
    out.append((
        "D1 authored_by='dsl' accepted (Phase 1 enum edit landed)",
        accepted,
        msg,
    ))

    # D2: provenance.tuning_trace is now ACCEPTED (Phase 1 added the property).
    d2 = {**base, "provenance": {**base["provenance"], "tuning_trace": [
        {"step": 1, "edit": "retile", "target": "t0", "args": {"shape": [128, 128]},
         "check": "ok", "predicted": {}, "measured": {}},
    ]}}
    accepted = True
    msg = "accepted"
    try:
        validate_impl_card(d2)
    except Exception as e:  # noqa: BLE001
        accepted = False
        msg = f"{type(e).__name__}: {e}"
    out.append((
        "D2 provenance.tuning_trace accepted (Phase 1 property landed)",
        accepted,
        msg,
    ))

    # D3: top-level launch.graph is now ACCEPTED (Phase 1 added the launch object).
    d3 = {**base, "launch": {"graph": True, "nodes": ["rmsnorm", "gemm"]}}
    accepted = True
    msg = "accepted"
    try:
        validate_impl_card(d3)
    except Exception as e:  # noqa: BLE001
        accepted = False
        msg = f"{type(e).__name__}: {e}"
    out.append((
        "D3 top-level launch.graph accepted (Phase 1 launch field landed)",
        accepted,
        msg,
    ))

    # D4 (positive, unchanged): specialization_knobs allows namespaced _doc.
    d4 = {**base, "specialization_knobs": {
        **base["specialization_knobs"], "BLOCK_M": {
            "type": "int", "choices": [128, 256], "_doc": "seeded by vkl probe"
        }}}
    accepted = True
    msg = "accepted"
    try:
        validate_impl_card(d4)
    except Exception as e:  # noqa: BLE001
        accepted = False
        msg = f"{type(e).__name__}: {e}"
    out.append((
        "D4 specialization_knobs._doc allowed (namespaced precedent)",
        accepted,
        msg,
    ))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Report runner
# ═══════════════════════════════════════════════════════════════════════════════

CHECKS = [
    ("A. Round-trip (header -> spec -> ingest)", check_roundtrip_spec),
    ("B. Three cards ingest (triton/cuda/hip)", check_cards_ingest),
    ("C. Edit preconditions locally decidable", check_edit_decidability),
    ("D. Schema-extension finding (documented)", check_schema_finding),
]


def run_all() -> list[tuple[str, list[tuple[str, bool, str]]]]:
    return [(name, fn()) for name, fn in CHECKS]


def main() -> int:
    print("=" * 78)
    print("vkl Phase 0 probe — contract-native thesis check (docs/brainstorm/11 §7)")
    print(f"jsonschema validator active: {have_validator()}  (must be True for real checks)")
    print("=" * 78)
    exit_code = 0
    for section_name, results in run_all():
        print(f"\n## {section_name}")
        for name, passed, detail in results:
            flag = "PASS" if passed else "FAIL"
            if not passed:
                exit_code = 1
            print(f"  [{flag}] {name}")
            print(f"         {detail}")
    print("\n" + "=" * 78)
    status = "ALL CHECKS PASSED" if exit_code == 0 else "SOME CHECKS FAILED (see 06 §D)"
    print(status)
    print("=" * 78)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
