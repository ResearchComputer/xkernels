"""MCP server exposing the agent-native surfaces as tools (docs/library.md §8.1).

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


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "find_impl",
            "description": (
                "Ranked retrieval of kernel implementations over the contract "
                "(docs/library.md §3). Returns candidates with `applicable` + "
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
