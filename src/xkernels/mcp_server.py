"""MCP server exposing the agent-native surfaces as tools (meta/docs/library.md §8.1).

This is the highest-leverage interop move: any MCP-speaking coding agent (Claude
Code, Cursor, Codex, Cline, …) can call ``verify`` / ``verify_parity`` and
inherit the correctness + parity guarantee even if it ignores our skills and
writes a kernel from scratch.

Run with an MCP-compatible client, or directly for a stdio session::

    python -m xkernels.mcp_server

Requires the optional ``mcp`` package. When absent, ``server_factory`` raises a
clear error instead of failing at import, so the rest of the library is usable
without it.
"""
from __future__ import annotations

import json
from typing import Any

from .registry import archs as _ARCHS
from .retrieval import find_impl
from .verify import verify, verify_parity

# ═══════════════════════════════════════════════════════════════════════════════
# Phase B: the vkl agent surface (docs/brainstorm/09) — schedule as source of truth
# ═══════════════════════════════════════════════════════════════════════════════
# These tools are the MCP realization of the doc-09 agent-editable-IR thesis.
# An agent's full tuning loop, stateless across calls:
#   load_schedule -> check_edit -> apply_edit -> (read_cost) -> measure -> record_trace
# The agent carries its state as an ``applied_edits`` list; the server replays from
# the spec each call (the schedule is a deterministic function of spec + edits, so
# there is no hidden server-side state to drift). The schedule IR is the source of
# truth; the binding the launcher reads (``resolve_binding``) is the projection.


def _vkl_spec(spec_id: str):
    """Resolve a spec id (``gemm_bf16`` or ``gemm_bf16@1.0.0``) to its KernelSpec."""
    from .vkl import examples as _ex

    for attr in dir(_ex):
        if not attr.endswith("_spec"):
            continue
        spec = getattr(_ex, attr)
        if (
            getattr(spec, "kernel", None) == spec_id
            or getattr(spec, "id", None) == spec_id
            or str(getattr(spec, "id", "")).startswith(spec_id + "@")
        ):
            return spec
    raise KeyError(
        f"no vkl spec matching {spec_id!r} (imported examples: "
        f"{[a[:-5] for a in dir(_ex) if a.endswith('_spec')]})"
    )


_EDIT_KW_TO_CLS = None  # built lazily (avoids importing edits at module load)


def _parse_edit(d: dict[str, Any]):
    """Parse an MCP edit dict into the matching edit dataclass (edits.py)."""
    global _EDIT_KW_TO_CLS
    if _EDIT_KW_TO_CLS is None:
        from .vkl import AddStage, MapTo_, Retile, SetKnob, SetMapPolicy

        _EDIT_KW_TO_CLS = {
            "set_knob": SetKnob,
            "retile": Retile,
            "map_to": MapTo_,
            "add_stage": AddStage,
            "set_map_policy": SetMapPolicy,
        }
    kind = d.get("kind")
    cls = _EDIT_KW_TO_CLS.get(kind)
    if cls is None:
        raise ValueError(f"unknown edit kind {kind!r} (have {list(_EDIT_KW_TO_CLS)})")
    fields = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k != "kind"}
    return cls(**fields)


def _replay(spec, arch: str, applied_edits: list[dict[str, Any]]):
    """Build the schedule from spec + arch, then apply each edit in order."""
    from .vkl import schedule_from_spec

    sched = schedule_from_spec(spec, arch=arch)
    for d in applied_edits or []:
        sched = _parse_edit(d).apply(sched)
    return sched


def _serialize_schedule(sched) -> dict[str, Any]:
    """Project a ScheduleIR to a JSON-safe view an agent reasons over.

    Nodes keep their id + kind + the key fields an edit targets; ``binding`` is
    the flat dict the launcher reads (``resolve_binding``); ``precision`` is the
    MMA's input_precision policy (None = dtype-default). This is the contract an
    MCP agent sees — every field is a primitive, so it crosses JSON cleanly.

    Phase C: when the schedule carries a ``profile`` side-table (attached by
    ``vkl_annotate_profile``), each annotated node gets an INLINE ``profile``
    field (the normalized on-device metrics) and the top-level view carries a
    ``profile`` summary keyed by node id — so an agent reading a node sees its
    annotation without a second lookup, and a diagnose skill reads the route off
    the ``MapTo`` node directly (issue #74).
    """
    _ensure_ir_types()
    from .vkl import precision_of, resolve_binding

    profile_summary = {nid: pm.to_dict() for nid, pm in sched.profile.items()}
    nodes = []
    for n in sched.nodes:
        if isinstance(n, _Tile):
            nd = {"id": n.id, "kind": "Tile", "shape": list(n.shape), "level": n.level}
        elif isinstance(n, _MapTo):
            nd = {
                "id": n.id, "kind": "MapTo", "op_ref": n.op_ref, "level": n.level,
                "instruction": n.instruction, "precision": n.precision,
            }
        elif isinstance(n, _Stage):
            nd = {
                "id": n.id, "kind": "Stage", "producer_ref": n.producer_ref,
                "space": n.space, "depth": n.depth,
            }
        elif isinstance(n, _Knob):
            nd = {
                "id": n.name, "kind": "Knob", "value": n.value, "choices": list(n.choices),
            }
        else:
            continue
        if nd["id"] in profile_summary:
            nd["profile"] = profile_summary[nd["id"]]
        nodes.append(nd)
    view: dict[str, Any] = {
        "nodes": nodes,
        "knobs": {
            k: {"value": v.value, "choices": list(v.choices)} for k, v in sched.knobs.items()
        },
        "binding": resolve_binding(sched),
        "precision": precision_of(sched),
    }
    if profile_summary:
        view["profile"] = profile_summary
    return view


def _legal_edits(sched, arch: str, *, include_rejected: bool = False) -> list[dict[str, Any]]:
    """Enumerate small, locally-decidable next edits for the current schedule.

    This is intentionally conservative. It does not try to invent a full tiling
    plan; it exposes the closed edit space already declared by the schedule:
    alternate knob bindings, alternate MMA precision policies, and legal native
    instruction remaps. Agents can still propose arbitrary edits through
    ``vkl_check_edit``; this list is the low-entropy starting set.
    """
    from .vkl import archdb as _archdb
    from .vkl.edits import is_ok
    from .vkl.ir.schedule import PRECISION_POLICIES
    from .vkl.ir.schedule import MapTo as _MapTo

    out: list[dict[str, Any]] = []

    def consider(edit_dict: dict[str, Any]) -> None:
        edit = _parse_edit(edit_dict)
        r = edit.check(sched, arch)
        if is_ok(r):
            out.append({"edit": edit_dict})
        elif include_rejected:
            out.append({"edit": edit_dict, "reject_reason": r.reason})  # type: ignore[attr-defined]

    for knob in sched.knobs.values():
        for value in knob.choices:
            if value != knob.value:
                consider({"kind": "set_knob", "name": knob.name, "value": value})

    for node in sched.maps():
        if not isinstance(node, _MapTo):
            continue
        for precision in PRECISION_POLICIES:
            if precision != node.precision:
                consider({"kind": "set_map_policy", "map_id": node.id, "precision": precision})
        for instruction in _archdb.legal_instructions(arch):
            if instruction != node.instruction:
                consider({
                    "kind": "map_to",
                    "map_id": node.id,
                    "op_ref": node.op_ref,
                    "level": node.level,
                    "instruction": instruction,
                    "precision": node.precision,
                })
    return out


def _schedule_cost(spec, sched, arch: str, point: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a JSON-safe closed-form cost summary for a schedule."""
    from .vkl import cost as _cost

    view = _serialize_schedule(sched)
    binding = view["binding"]
    config = {k: int(v) for k, v in binding.items() if isinstance(v, int)}
    pattern = getattr(getattr(spec, "launch", None), "pattern", "direct")
    dtype = (
        str(point["dtype"])
        if point and "dtype" in point
        else next(iter(spec.inputs.values())).dtype[0]
    )
    maps = [n for n in view["nodes"] if n["kind"] == "MapTo"]
    instruction = (maps[0].get("instruction") if maps else None) or "fma"
    cost: dict[str, Any] = {
        "pattern": pattern,
        "dtype": dtype,
        "dtype_source": "point" if point and "dtype" in point else "first_declared_input_dtype",
        "instruction": instruction,
        "scratch_bytes": _cost.predict_scratch(pattern, config, dtype, arch),
        "overflows_scratch": _cost.overflows_scratch(pattern, config, dtype, arch),
        "occupancy": _cost.occupancy(pattern, config, dtype, arch).to_dict(),
    }
    if point is not None:
        rf = _cost.roofline(spec.id, point, arch, instruction=instruction)
        if rf is not None:
            cost["roofline"] = rf.to_dict()
    view["cost"] = cost
    view["legal_edits"] = _legal_edits(sched, arch)
    return view


# Lazy imports of the IR node types (avoid importing vkl at module load — it pulls
# torch, which the MCP server's find_impl/verify tools don't need).
_Tile = _MapTo = _Stage = _Knob = None  # type: ignore[assignment]


def _ensure_ir_types() -> None:
    global _Tile, _MapTo, _Stage, _Knob
    if _Tile is None:
        from .vkl import Knob, MapTo, Stage, Tile

        _Tile, _MapTo, _Stage, _Knob = Tile, MapTo, Stage, Knob


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "find_impl",
            "description": (
                "Ranked retrieval of kernel implementations over the contract "
                "(meta/docs/library.md §3). Returns candidates with `applicable` + "
                "`reject_reasons`. Two-stage: filter Op Specs by decidable "
                "constraints, then Impl Cards by arch."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "canonical_op": {
                        "type": "string",
                        "description": "gemm|norm|reduce|attention|...",
                    },
                    "input_specs": {"type": "object"},
                    "target_arch": {
                        "type": "string",
                        # Derived from the single source of truth (registry.archs)
                        # so the MCP schema never drifts from the contract enum.
                        "description": f"{'|'.join(sorted(_ARCHS.ALL_ARCHS))}|any",
                    },
                    "available_features": {"type": "array", "items": {"type": "string"}},
                    "required_fusions": {"type": "array", "items": {"type": "string"}},
                    "objective": {"type": "string", "description": "throughput|latency|memory"},
                },
                "required": ["canonical_op"],
            },
        },
        {
            "name": "verify",
            "description": (
                "Verify one Implementation Card against its Op Spec's single "
                "backend-neutral reference + tolerances (§5.2). Returns a "
                "structured blob: compiled, correctness{passed,max_abs_err,"
                "max_rel_err,failing_shapes}, determinism_check, perf."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "impl_card_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "knobs": {"type": "object"},
                    "shapes": {"description": "sweep id (str) or point list"},
                    "seed": {"type": "integer"},
                    "measure_perf": {"type": "boolean"},
                },
                "required": ["impl_card_id"],
            },
        },
        {
            "name": "verify_parity",
            "description": (
                "Cross-backend parity gate (§5.3): do >=2 backends of an op agree "
                "with each other within cross_backend_rtol? A card that breaks "
                "parity cannot publish."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "op_id": {"type": "string"},
                    "archs": {"type": "array", "items": {"type": "string"}},
                    "shapes": {},
                    "seed": {"type": "integer"},
                },
                "required": ["op_id"],
            },
        },
        {
            "name": "record_measurement",
            "description": (
                "Write back a (op,arch,shape,dtype) -> knobs -> perf measurement "
                "into the card's perf.measured (§6.2 compounding loop). The "
                "measurement must cite a reproducible run id. NOTE: external "
                "(untrusted) callers are read+verify only; write-back requires a "
                "server-side-rerun source (§11)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "impl_card_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "shape": {"type": "object"},
                    "dtype": {"type": "string"},
                    "knobs": {"type": "object"},
                    "tflops": {"type": "number"},
                    "achieved_bw_pct": {"type": "number"},
                    "ms": {"type": "number"},
                    "source": {"type": "string", "description": "reproducible run id"},
                },
                "required": ["impl_card_id", "arch", "shape", "dtype", "source"],
            },
        },
        {
            "name": "vkl_load_schedule",
            "description": (
                "Build the structured schedule IR for a DSL-authored kernel + arch "
                "(docs/brainstorm/09). Returns the nodes (Tile/MapTo/Stage/Knob), the "
                "knobs, the binding the launcher reads, and the MMA precision policy. "
                "This is the read-out half of the schedule-IR source-of-truth: the "
                "agent edits this view by name, then check_edit/apply_edit reach silicon."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {
                        "type": "string",
                        "description": "gemm_bf16 | gemm_bf16@1.0.0 | ...",
                    },
                    "arch": {"type": "string"},
                },
                "required": ["spec_id"],
            },
        },
        {
            "name": "vkl_validate_kernel",
            "description": (
                "CPU-decidable VKL contract preflight. Validates emitted schema "
                "artifacts, decidable constraints, math-IR trace shape, launch "
                "compatibility, reduction dtype invariants, and declared Triton knobs. "
                "Complements verify/verify_parity, which remain the publish gate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                },
                "required": ["spec_id"],
            },
        },
        {
            "name": "vkl_list_legal_edits",
            "description": (
                "Enumerate conservative locally-legal next schedule edits from the "
                "current replayed schedule: alternate knob bindings, MMA precision "
                "policies, and legal native instruction remaps. Use this before "
                "guessing edits."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {"type": "array", "items": {"type": "object"}},
                    "include_rejected": {"type": "boolean"},
                },
                "required": ["spec_id", "arch"],
            },
        },
        {
            "name": "vkl_check_edit",
            "description": (
                "Locally-decidable precondition for a schedule edit (docs/brainstorm/10 §5). "
                "No code runs — a pure function of (edit + current IR + arch). Returns "
                "{ok: bool, reason?}. The agent calls this BEFORE apply_edit to predict "
                "legality; reject reasons are the training signal that skips dead-ends. "
                "Stateless: pass the full applied_edits list to replay the current IR."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {
                        "type": "array", "items": {"type": "object"},
                        "description": "prior edits, in order (the agent's replayed state)",
                    },
                    "edit": {
                        "type": "object",
                        "description": (
                            "{kind: set_knob|retile|map_to|add_stage|set_map_policy, ...}. "
                            "set_map_policy is the MMA input_precision lever (tf32/ieee)."
                        ),
                    },
                },
                "required": ["spec_id", "arch", "edit"],
            },
        },
        {
            "name": "vkl_apply_edit",
            "description": (
                "Apply a schedule edit (check first, then return the new IR snapshot). "
                "Returns the new serialized schedule (nodes + binding + precision) and "
                "the appended applied_edits list the agent carries forward. Immutable: "
                "each call is a new snapshot (the tuning_trace form)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {"type": "array", "items": {"type": "object"}},
                    "edit": {"type": "object"},
                },
                "required": ["spec_id", "arch", "edit"],
            },
        },
        {
            "name": "vkl_read_cost",
            "description": (
                "Surface the cost-relevant summary of a schedule (the read-in projection): "
                "the binding + tile shapes + arch-native MMA instruction. Phase C: when "
                "the schedule carries a `profile` side-table (attached by "
                "vkl_annotate_profile), read_cost surfaces the per-node on-device "
                "annotations inline — so the agent sees both the PREDICTION (closed-form "
                "cost) and the MEASUREMENT (profile) on the same nodes it reasons over. "
                "(Full roofline / occupancy / scratch annotation per node without a "
                "profile is still the closed-form prediction below.)"
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {"type": "array", "items": {"type": "object"}},
                    "point": {
                        "type": "object",
                        "description": (
                            "optional concrete sweep point; when present, read_cost "
                            "adds roofline using this shape/dtype"
                        ),
                    },
                },
                "required": ["spec_id", "arch"],
            },
        },
        {
            "name": "vkl_annotate_profile",
            "description": (
                "Phase C (issue #74): attach an on-device profile (ncu / rocprof) to the "
                "schedule's node ids. Parses the raw profiler text-table into the "
                "normalized §10 vocabulary (bottleneck / dominant_stall / achieved_bw / "
                "compute_throughput / tensor_pipe_util / occupancy), then keys it to the "
                "MapTo node (the heavy op the diagnose skills route on) and the Stage/Tile "
                "nodes (the load pipeline). Returns the annotated schedule view + the "
                "causal route (which diagnose skill the dominant stall reason points to). "
                "This is the bridge from the decidable cost PREDICTION to the measured "
                "cost; the profile text itself is produced on a GPU (bristen sm_80 for ncu, "
                "beverin gfx942 for rocprof)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {"type": "array", "items": {"type": "object"}},
                    "profiler": {
                        "type": "string",
                        "description": "ncu | rocprof (alias: omniperf, nsight)",
                    },
                    "profile_text": {
                        "type": "string",
                        "description": (
                            "the raw profiler text-table output (ncu .report.txt or "
                            "rocprof .analyze.txt). The parser is format-tolerant; "
                            "absent metrics degrade to null."
                        ),
                    },
                    "point": {
                        "type": "object",
                        "description": "optional sweep point (roofline cross-check)",
                    },
                },
                "required": ["spec_id", "arch", "profiler", "profile_text"],
            },
        },
        {
            "name": "vkl_route_from_profile",
            "description": (
                "Phase C convenience: return ONLY the causal routing decision from a "
                "profile — the diagnose skill the dominant stall reason points to, plus "
                "the MapTo node id it was read off. This is the one-line consumer a "
                "diagnose skill calls instead of reading a separate profile file (issue #74 "
                "criterion 3). Routes by the dominant stall reason (causal), not the "
                "throughput ratio alone: memory_latency -> diagnose-memory-bound; "
                "dependency/vgpr/scratch -> diagnose-low-occupancy; compute-bound with an "
                "idle matrix engine -> map-to-matrix-cores."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "spec_id": {"type": "string"},
                    "arch": {"type": "string"},
                    "applied_edits": {"type": "array", "items": {"type": "object"}},
                    "profiler": {"type": "string"},
                    "profile_text": {"type": "string"},
                },
                "required": ["spec_id", "arch", "profiler", "profile_text"],
            },
        },
        {
            "name": "get_op_spec",
            "description": "Return the full Op Spec document for an op id.",
            "inputSchema": {"type": "object", "properties": {"op_id": {"type": "string"}},
                            "required": ["op_id"]},
        },
        {
            "name": "get_impl_card",
            "description": "Return the full Implementation Card document.",
            "inputSchema": {"type": "object", "properties": {"impl_card_id": {"type": "string"}},
                            "required": ["impl_card_id"]},
        },
        {
            "name": "record_outcome",
            "description": (
                "Record a skill outcome (§7.3 governance loop): success/partial/fail "
                "+ iterations + failure_mode. Rolls into the skill's metrics so "
                "authoring gets more reliable and cheaper per task. Integrated-runtime "
                "write only (external callers are read + verify only)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string"},
                    "version": {"type": "string"},
                    "task_signature": {"type": "string"},
                    "result": {"type": "string", "enum": ["success", "partial", "fail"]},
                    "iterations": {"type": "integer"},
                    "run_id": {"type": "string"},
                    "failure_mode": {"type": "string"},
                },
                "required": ["skill_id", "version", "task_signature", "result"],
            },
        },
        {
            "name": "skill_metrics",
            "description": (
                "Roll a skill's outcome records into metrics: success_rate, "
                "median_iterations, regression_count, failure_modes (§7.3.1)."
            ),
            "inputSchema": {"type": "object", "properties": {"skill_id": {"type": "string"}},
                            "required": ["skill_id"]},
        },
        {
            "name": "list_skills",
            "description": (
                "List the SKILL.md skills library (§7), optionally filtered by "
                "backend_scope (§7.2). Each entry has the trigger `description`, "
                "the namespaced x-kernel-lib metadata (id, backend_scope, tools), "
                "and the procedure body path."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "description": ("only skills whose backend_scope applies "
                                         "(cuda|hip|agnostic)"),
                    }
                },
            },
        },
        {
            "name": "get_skill",
            "description": (
                "Return a single skill's parsed metadata + procedure body."
            ),
            "inputSchema": {"type": "object", "properties": {"skill_id": {"type": "string"}},
                            "required": ["skill_id"]},
        },
    ]


def _dispatch(name: str, args: dict[str, Any]) -> Any:
    if name == "find_impl":
        return find_impl(**args)
    if name == "verify":
        return verify(**args)
    if name == "verify_parity":
        return verify_parity(**args)
    if name == "get_op_spec":
        from .registry import get_spec
        return get_spec(args["op_id"]).doc
    if name == "vkl_load_schedule":
        _ensure_ir_types()
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args.get("arch", "any"), [])
        return _serialize_schedule(sched)
    if name == "vkl_validate_kernel":
        from .vkl import validate_kernel
        spec = _vkl_spec(args["spec_id"])
        return validate_kernel(spec, arch=args.get("arch", "any")).to_dict()
    if name == "vkl_list_legal_edits":
        _ensure_ir_types()
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args["arch"], args.get("applied_edits", []))
        return {
            "legal_edits": _legal_edits(
                sched,
                args["arch"],
                include_rejected=bool(args.get("include_rejected", False)),
            )
        }
    if name == "vkl_check_edit":
        _ensure_ir_types()
        from .vkl import Ok
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args["arch"], args.get("applied_edits", []))
        edit = _parse_edit(args["edit"])
        r = edit.check(sched, args["arch"])
        return {"ok": isinstance(r, Ok),
                "reason": None if isinstance(r, Ok) else r.reason}
    if name == "vkl_apply_edit":
        _ensure_ir_types()
        from .vkl import Ok
        spec = _vkl_spec(args["spec_id"])
        prior = args.get("applied_edits", [])
        sched = _replay(spec, args["arch"], prior)
        edit = _parse_edit(args["edit"])
        r = edit.check(sched, args["arch"])
        if not isinstance(r, Ok):
            return {"applied": False, "reason": r.reason}
        sched2 = edit.apply(sched)
        return {
            "applied": True,
            "schedule": _serialize_schedule(sched2),
            "applied_edits": prior + [args["edit"]],
        }
    if name == "vkl_read_cost":
        _ensure_ir_types()
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args["arch"], args.get("applied_edits", []))
        return _schedule_cost(spec, sched, args["arch"], args.get("point"))
    if name == "vkl_annotate_profile":
        _ensure_ir_types()
        from .vkl import profile as _profile
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args["arch"], args.get("applied_edits", []))
        metrics = _profile.parse_profile(
            args["profiler"], args["profile_text"], arch=args["arch"]
        )
        sched = _profile.annotate_schedule(sched, metrics)
        view = _schedule_cost(spec, sched, args["arch"], args.get("point"))
        view["route"] = _profile.route_of(sched) or {
            "node_id": None,
            "skill": _profile.route(metrics),
            "bottleneck": metrics.bottleneck,
            "dominant_stall": metrics.dominant_stall,
            "dominant_stall_pct": metrics.dominant_stall_pct,
        }
        return view
    if name == "vkl_route_from_profile":
        _ensure_ir_types()
        from .vkl import profile as _profile
        spec = _vkl_spec(args["spec_id"])
        sched = _replay(spec, args["arch"], args.get("applied_edits", []))
        metrics = _profile.parse_profile(
            args["profiler"], args["profile_text"], arch=args["arch"]
        )
        sched = _profile.annotate_schedule(sched, metrics)
        decision = _profile.route_of(sched)
        if decision is None:
            decision = {
                "node_id": None,
                "skill": _profile.route(metrics),
                "bottleneck": metrics.bottleneck,
                "dominant_stall": metrics.dominant_stall,
                "dominant_stall_pct": metrics.dominant_stall_pct,
            }
        return decision
    if name == "get_impl_card":
        from .registry import get_card
        return get_card(args["impl_card_id"]).doc
    if name == "record_measurement":
        from .registry.writeback import record_measurement
        return record_measurement(**args)
    if name == "record_outcome":
        from .registry.outcomes import record_outcome
        return record_outcome(**args)
    if name == "skill_metrics":
        from .registry.outcomes import skill_metrics
        return skill_metrics(**args)
    if name == "list_skills":
        from .registry import all_skills, skills_for_backend
        backend = (args or {}).get("backend")
        lib = skills_for_backend(backend) if backend else all_skills()
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "backend_scope": list(s.meta.backend_scope) if s.meta else ["agnostic"],
                "tools": list(s.meta.tools) if s.meta else [],
                "triggers": list(s.meta.triggers) if s.meta else [],
                "path": str(s.path),
            }
            for s in lib.values()
        ]
    if name == "get_skill":
        from .registry import get_skill
        s = get_skill(args["skill_id"])
        return {
            "id": s.id,
            "name": s.name,
            "description": s.description,
            "license": s.license,
            "backend_scope": list(s.meta.backend_scope) if s.meta else ["agnostic"],
            "tools": list(s.meta.tools) if s.meta else [],
            "validation_must_pass": list(s.meta.validation_must_pass) if s.meta else [],
            "body": s.body,
            "path": str(s.path),
        }
    raise ValueError(f"unknown tool {name!r}")


def server_factory(server_name: str = "xkernels"):
    """Build and return an MCP Server (requires the `mcp` package)."""
    try:
        from mcp.server import Server  # type: ignore[import-untyped]
        from mcp.types import Tool  # type: ignore[import-untyped]
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError(
            "the `mcp` package is required for the MCP server; "
            "install with `pip install mcp`"
        ) from e

    server = Server(server_name)

    @server.list_tools()
    async def _list() -> list[Tool]:  # pragma: no cover - exercised by MCP clients
        return [Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in _tools()]

    @server.call_tool()
    async def _call(name: str, arguments: dict) -> list:  # pragma: no cover
        result = _dispatch(name, arguments or {})
        from mcp.types import TextContent  # type: ignore[import-untyped]
        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    return server


def main() -> None:  # pragma: no cover - requires a live MCP client
    import asyncio

    server = server_factory()

    async def _run() -> None:
        from mcp.server.stdio import stdio_server  # type: ignore[import-untyped]

        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover
    main()
