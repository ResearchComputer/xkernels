# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase B: the vkl agent surface as MCP tools (docs/brainstorm/09).

Exercises the full stateless agent tuning loop over the MCP dispatch:

    vkl_load_schedule -> vkl_check_edit -> vkl_apply_edit -> vkl_read_cost

The agent carries its state as an ``applied_edits`` list; the server replays from
the spec each call (the schedule is a deterministic function of spec + edits).
This is the MCP realization of the doc-09 thesis: an agent edits the schedule IR
by name, and the binding the launcher reads is the projection of those edits.

The load_schedule / serialize / parse-edit helpers are CPU-doable; the tools are
driven here through ``_dispatch`` (the same path the MCP server's call_tool uses).
The gemm_bf16 read-out needs torch (the @kernel body traces via trace_ir); the
end-to-end "edited schedule launches" check is GPU-gated.
"""
from __future__ import annotations

import pytest

from xkernels.mcp_server import _dispatch

# ─── GPU gating ───────────────────────────────────────────────────────────────
try:  # pragma: no cover
    import torch

    _GPU_OK = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    _GPU_OK = False
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  load_schedule: the read-out (spec + arch -> structured schedule)
# ═══════════════════════════════════════════════════════════════════════════════


class TestLoadSchedule:
    def test_gemm_schedule_has_tiles_map_stages_knobs(self):
        view = _dispatch("vkl_load_schedule", {"spec_id": "gemm_bf16", "arch": "nvidia_sm90"})
        kinds = {n["kind"] for n in view["nodes"]}
        assert {"Tile", "MapTo", "Stage", "Knob"} <= kinds
        # the MMA map on sm_90 uses the native matrix engine
        maps = [n for n in view["nodes"] if n["kind"] == "MapTo"]
        assert len(maps) == 1 and maps[0]["instruction"] == "wgmma"
        # default precision is None (dtype-default) until an edit sets it
        assert view["precision"] is None
        # binding carries the declared knob values + no precision key
        assert "BLOCK_M" in view["binding"]
        assert "input_precision" not in view["binding"]

    def test_portable_target_has_no_concrete_instruction(self):
        view = _dispatch("vkl_load_schedule", {"spec_id": "gemm_bf16", "arch": "any"})
        maps = [n for n in view["nodes"] if n["kind"] == "MapTo"]
        assert maps[0]["instruction"] is None

    def test_unknown_spec_raises(self):
        with pytest.raises(KeyError, match="no vkl spec matching"):
            _dispatch("vkl_load_schedule", {"spec_id": "nope", "arch": "nvidia_sm90"})

    def test_validate_kernel_preflight_passes(self):
        r = _dispatch("vkl_validate_kernel", {"spec_id": "gemm_bf16", "arch": "nvidia_sm90"})
        assert r["passed"] is True
        assert r["error_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# §2  the edit loop: check -> apply (stateless replay)
# ═══════════════════════════════════════════════════════════════════════════════


class TestEditLoop:
    def test_check_accepts_tf32_edit(self):
        r = _dispatch("vkl_check_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"},
        })
        assert r["ok"] is True and r["reason"] is None

    def test_check_rejects_bad_precision(self):
        r = _dispatch("vkl_check_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "set_map_policy", "map_id": "mma0", "precision": "bogus"},
        })
        assert r["ok"] is False and "not in" in r["reason"]

    def test_check_rejects_illegal_instruction(self):
        r = _dispatch("vkl_check_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "map_to", "map_id": "mma0", "op_ref": "acc",
                     "level": "L5", "instruction": "my_asm"},
        })
        assert r["ok"] is False and "not legal for" in r["reason"]

    def test_apply_returns_new_schedule_and_appended_edits(self):
        r = _dispatch("vkl_apply_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"},
        })
        assert r["applied"] is True
        assert r["schedule"]["precision"] == "tf32"
        assert r["schedule"]["binding"]["input_precision"] == "tf32"
        assert r["applied_edits"] == [
            {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"}
        ]

    def test_apply_refuses_when_check_fails(self):
        r = _dispatch("vkl_apply_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "set_map_policy", "map_id": "mma0", "precision": "bogus"},
        })
        assert r["applied"] is False and "not in" in r["reason"]

    def test_replay_chains_two_edits(self):
        """Stateless replay: the agent carries applied_edits across two edits."""
        r1 = _dispatch("vkl_apply_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [],
            "edit": {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"},
        })
        r2 = _dispatch("vkl_apply_edit", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": r1["applied_edits"],
            "edit": {"kind": "set_knob", "name": "BLOCK_M", "value": 128},
        })
        sched = r2["schedule"]
        assert sched["precision"] == "tf32"  # first edit preserved
        assert sched["binding"]["BLOCK_M"] == 128  # second edit applied
        assert len(r2["applied_edits"]) == 2

    def test_read_cost_surfaces_binding(self):
        r = _dispatch("vkl_read_cost", {
            "spec_id": "gemm_bf16", "arch": "nvidia_sm90",
            "applied_edits": [
                {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"}
            ],
            "point": {"dtype": "bf16", "M": 128, "N": 128, "K": 128},
        })
        assert r["binding"]["input_precision"] == "tf32"
        assert r["cost"]["pattern"] == "tiled_2d"
        assert r["cost"]["instruction"] == "wgmma"
        assert "scratch_bytes" in r["cost"]
        assert "roofline" in r["cost"]
        assert r["legal_edits"]

    def test_list_legal_edits_surfaces_low_entropy_next_moves(self):
        r = _dispatch("vkl_list_legal_edits", {
            "spec_id": "gemm_bf16",
            "arch": "nvidia_sm90",
            "applied_edits": [],
        })
        edits = [item["edit"] for item in r["legal_edits"]]
        assert {"kind": "set_knob", "name": "BLOCK_M", "value": 128} in edits
        assert {"kind": "set_map_policy", "map_id": "mma0", "precision": "tf32"} in edits
