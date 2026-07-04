# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""CLI coverage for ``vkl implement``."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

from xkernels.vkl.cli import (
    _extract_json,
    _remote_env,
    _remote_gpu_extension_path,
    main,
    parse_target,
    resolve_target,
    run_agent,
)

CANONICAL_TARGET = "xkernels.vkl.examples.gemm_bf16:gemm_bf16"


def test_parse_target_requires_module_and_symbol():
    target = parse_target("xkernels.vkl.examples.gemm_bf16:gemm_bf16")
    assert target.module == "xkernels.vkl.examples.gemm_bf16"
    assert target.symbol == "gemm_bf16"


def test_resolve_target_reports_existing_vkl_kernel():
    info = resolve_target(
        parse_target("xkernels.vkl.examples.gemm_bf16:gemm_bf16"),
        Path.cwd(),
    )
    assert info["importable"] is True
    assert info["exists"] is True
    assert info["vkl_kernel"] is True
    assert info["spec_id"] == "gemm_bf16@1.0.0"
    assert info["validation"]["passed"] is True


def test_implement_dry_run_builds_pi_json_command(capsys):
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--dry-run",
            "--arch",
            "nvidia_sm90",
            "--backend",
            "triton",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["request"]["target"]["raw"] == CANONICAL_TARGET
    cmd = out["agent"]["cmd"]
    assert cmd[0] == "pi"
    assert "--mode" in cmd and "json" in cmd
    assert "--print" in cmd
    assert f"vkl implement {CANONICAL_TARGET}" in cmd
    assert "Request JSON:" in out["prompt"]


class _FakeProc:
    """Minimal subprocess.Popen double whose stdout/stderr are line iterables."""

    def __init__(self, stdout_lines, returncode=0, stderr_lines=()):
        self._out = list(stdout_lines)
        self._err = list(stderr_lines)
        self.returncode = returncode

    @property
    def stdout(self):
        return iter(self._out)

    @property
    def stderr(self):
        return iter(self._err)

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def _ndjson_lines(final_text: str) -> list[str]:
    """A realistic pi ``--mode json`` event stream ending in ``final_text``."""
    assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": final_text}]}
    return [
        json.dumps({"type": "session", "version": 3, "id": "x", "timestamp": "t", "cwd": "/p"})
        + "\n",
        json.dumps({"type": "agent_start"}) + "\n",
        json.dumps({"type": "turn_start"}) + "\n",
        json.dumps({"type": "message_start", "message": {"role": "assistant", "content": []}})
        + "\n",
        json.dumps(
            {
                "type": "message_update",
                "message": assistant_msg,
                "assistantMessageEvent": {"type": "text_delta", "delta": final_text},
            }
        )
        + "\n",
        json.dumps({"type": "message_end", "message": assistant_msg}) + "\n",
        json.dumps({"type": "turn_end", "message": assistant_msg, "toolResults": []}) + "\n",
        json.dumps(
            {
                "type": "agent_end",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "go"}]},
                    assistant_msg,
                ],
            }
        )
        + "\n",
    ]


def test_implement_runs_configured_agent(monkeypatch, capsys):
    # The agent wraps its JSON answer in a ```json fence (a very common shape).
    final_text = '```json\n{"status": "success", "changed_files": ["src/a.py"]}\n```'

    def fake_popen(cmd, *a, **k):
        assert cmd[:1] == ["fake-pi"]
        assert k.get("bufsize") == 1  # streaming, not buffered capture
        return _FakeProc(_ndjson_lines(final_text), returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "fake-pi",
            "--quiet",
            "--no-approve",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["agent"]["ran"] is True
    assert out["agent"]["stdout_json"] == {"status": "success", "changed_files": ["src/a.py"]}
    assert out["agent"]["completed"] is True
    assert out["agent"]["interrupted"] is False
    assert out["agent"]["events"][-1]["type"] == "agent_end"


def test_run_agent_parses_ndjson_stream_and_strips_fence(monkeypatch):
    final_text = '```json\n{"status": "ok", "changed_files": ["a.py"], "blockers": []}\n```'
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: _FakeProc(_ndjson_lines(final_text), returncode=0),
    )
    res = run_agent(["pi", "--mode", "json"], Path.cwd(), echo=False)
    assert res["ran"] is True
    assert res["returncode"] == 0
    assert res["final_text"] == final_text  # raw text preserved for debugging
    assert res["stdout_json"] == {"status": "ok", "changed_files": ["a.py"], "blockers": []}
    # the full NDJSON stream is retained for forensics
    assert res["stdout"].count("\n") >= 7
    assert res["events"][0]["type"] == "session"
    assert res["events"][-1]["type"] == "agent_end"


def test_run_agent_parses_bare_json_without_fence(monkeypatch):
    final_text = '{"status": "ok", "changed_files": []}'
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: _FakeProc(_ndjson_lines(final_text), returncode=0),
    )
    res = run_agent(["pi", "--mode", "json"], Path.cwd(), echo=False)
    assert res["stdout_json"] == {"status": "ok", "changed_files": []}


def test_run_agent_installs_and_restores_sigterm_handler(monkeypatch):
    final_text = '{"status": "ok"}'
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: _FakeProc(_ndjson_lines(final_text), returncode=0),
    )
    real_signal = signal.signal
    calls: list = []

    def spy(signum, handler):
        calls.append((signum, handler))
        return real_signal(signum, handler)  # delegate to the real impl

    monkeypatch.setattr(signal, "signal", spy)
    try:
        res = run_agent(["pi", "--mode", "json"], Path.cwd(), echo=False)
    finally:
        monkeypatch.setattr(signal, "signal", real_signal)
    sigterms = [h for s, h in calls if s == signal.SIGTERM]
    assert len(sigterms) >= 1  # handler registered for the run
    assert callable(sigterms[0])
    assert res["stdout_json"] == {"status": "ok"}


def test_run_agent_emits_partial_result_on_interrupt(monkeypatch):
    # Simulate the effect of the SIGTERM handler: the stdout iterator raises
    # KeyboardInterrupt mid-stream. run_agent must terminate the child and still
    # return the events captured so far (no total loss on kill).
    lines = _ndjson_lines('{"status": "ok"}')
    holder = {}

    class InterruptProc(_FakeProc):
        def __init__(self):
            self.returncode = 130
            self.terminated = False

        @property
        def stdout(self):
            yield from lines[:4]  # emit a few events, then "interrupt"
            raise KeyboardInterrupt

        @property
        def stderr(self):
            return iter(())

        def wait(self, timeout=None):
            return self.returncode

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(
        subprocess, "Popen", lambda cmd, *a, **k: holder.setdefault("p", InterruptProc())
    )
    res = run_agent(["pi", "--mode", "json"], Path.cwd(), echo=False)
    assert res["ran"] is True
    assert res["interrupted"] is True
    assert res["completed"] is False
    assert holder["p"].terminated is True
    assert len(res["events"]) >= 1  # partial events survived
    assert res["events"][-1]["type"] in {"turn_start", "message_start"}


def test_run_agent_handles_missing_executable(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("[Errno 2] No such file or directory: 'pi'")

    monkeypatch.setattr(subprocess, "Popen", boom)
    res = run_agent(["pi", "--mode", "json"], Path.cwd())
    # Must NOT raise: a missing binary is reported as a structured blocker.
    assert res["ran"] is False
    assert res["returncode"] is None
    assert res["stdout_json"] is None
    assert "not found" in res["stderr"]


def test_implement_reports_missing_agent(monkeypatch, capsys):
    def boom(*a, **k):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(subprocess, "Popen", boom)
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "missing-pi",
            "--quiet",
            "--out",
            "stdout",
        ]
    )
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["agent"]["ran"] is False
    assert "not found" in out["agent"]["stderr"]


def test_implement_writes_full_result_to_dist_by_default(monkeypatch, capsys, tmp_path):
    # Default --out dist: the full result blob lands in the gitignored dist/
    # tree, and stdout carries a compact summary (with the result path) so a
    # caller can `cat $(... | jq -r .result_path)` or jq the summary directly.
    payload = {"status": "success", "changed_files": ["src/a.py"], "blockers": []}
    final_text = f"```json\n{json.dumps(payload)}\n```"
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: _FakeProc(_ndjson_lines(final_text), returncode=0),
    )
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "fake-pi",
            "--quiet",
            "--no-approve",
            "--cwd",
            str(tmp_path),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)  # stdout is the compact summary, not the blob
    assert summary["ok"] is True
    assert summary["out"] == "dist"
    assert summary["status"] == "success"
    assert summary["changed_files"] == ["src/a.py"]
    assert summary["blockers"] == []
    assert summary["result_path"].startswith("dist/vkl-implement/")

    full_path = tmp_path / summary["result_path"]
    assert full_path.exists()
    full = json.loads(full_path.read_text())
    assert full["agent"]["stdout_json"] == {
        "status": "success",
        "changed_files": ["src/a.py"],
        "blockers": [],
    }
    assert full["result_path"] == summary["result_path"]
    assert full["out"] == "dist"


def test_implement_stdout_out_emits_legacy_full_blob(monkeypatch, capsys):
    final_text = '{"status": "ok", "changed_files": []}'
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda cmd, *a, **k: _FakeProc(_ndjson_lines(final_text), returncode=0),
    )
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "fake-pi",
            "--quiet",
            "--no-approve",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["out"] == "stdout"
    assert out["agent"]["stdout_json"] == {"status": "ok", "changed_files": []}
    assert "result_path" not in out


def test_extract_json_handles_fences_prose_and_garbage():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('```\n{"a": 2}\n```') == {"a": 2}
    assert _extract_json('{"a": 3}') == {"a": 3}
    assert _extract_json(None) is None
    assert _extract_json("") is None
    # prose around the fenced block -> fallback regex extraction
    assert _extract_json('Sure! Here:\n```json\n{"a": 4}\n```\nthanks') == {"a": 4}
    # genuinely unparseable -> None (no crash)
    assert _extract_json("no json here at all") is None



# ---------------------------------------------------------------------------
# --remote: pi runs LOCALLY; a pi extension reroutes GPU bash to the remote.
#   The agent process and all file ops stay local; only GPU-bearing bash
#   (verify/pytest/profiling/...) is rerouted via rcc. So:
#     - the pi command is NOT wrapped in rcc; `pi -e <ext>` is prepended instead
#     - no push/pull happens in Python (the extension does it per-command)
#     - XKL_REMOTE_* env configures the extension for the child pi process
# ---------------------------------------------------------------------------


def test_remote_gpu_extension_path_resolves_to_repo_extensions():
    ext = _remote_gpu_extension_path()
    assert ext.name == "remote-gpu.ts"
    # lives under <repo>/extensions/, shipped with the project
    assert ext.parent.name == "extensions"
    assert ext.exists(), f"extension must exist at {ext}"


def test_remote_env_maps_flags_to_xkl_env(monkeypatch):
    # All --remote knobs flow to the extension via XKL_REMOTE_* env.
    ns = argparse.Namespace(
        remote="beverin",
        rcc_cmd="rcc",
        no_push=True,
        pull=False,
        pull_paths="dist/, jobs/",
        push_best_effort=True,
        remote_patterns=r"\bmybench\b",
    )
    env = _remote_env(ns)
    assert env["XKL_REMOTE_PROFILE"] == "beverin"
    assert env["XKL_REMOTE_RCC_CMD"] == "rcc"
    assert env["XKL_REMOTE_NO_PUSH"] == "1"
    assert env["XKL_REMOTE_NO_PULL"] == "1"  # pull off by default
    assert env["XKL_REMOTE_PULL_PATHS"] == "dist/, jobs/"
    assert env["XKL_REMOTE_PUSH_BEST_EFFORT"] == "1"
    assert env["XKL_REMOTE_PATTERNS"] == r"\bmybench\b"


def test_remote_env_pull_flag_enables_pull():
    ns = argparse.Namespace(
        remote="ds5", rcc_cmd="rcc", no_push=False, pull=True,
        pull_paths="dist/", push_best_effort=False, remote_patterns="",
    )
    env = _remote_env(ns)
    assert env["XKL_REMOTE_NO_PULL"] == "0"  # --pull enables it


def test_implement_remote_dry_run_loads_extension_not_wraps_in_rcc(capsys):
    # Dry-run: pi runs locally with -e <ext>; the command is NOT wrapped in rcc,
    # and no push/pull happens. The extension is configured via env.
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--dry-run",
            "--remote",
            "beverin",
            "--arch",
            "amd_cdna3",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["remote"] == {
        "enabled": True,
        "profile": "beverin",
        "extension": str(_remote_gpu_extension_path()),
    }
    assert out["request"]["remote"] == {"profile": "beverin"}
    cmd = out["agent"]["cmd"]
    # pi first, then -e <ext>, then the rest — NOT wrapped in rcc
    assert cmd[0] == "pi"
    assert cmd[1] == "-e"
    assert cmd[2] == str(_remote_gpu_extension_path())
    assert not any(c == "rcc" for c in cmd), "pi must not be wrapped in rcc"
    # the prompt tells the agent it runs locally with remote GPU routing
    assert "YOU RUN LOCALLY" in out["prompt"]


def test_implement_remote_runs_local_pi_with_extension_env(monkeypatch, capsys):
    # Live run: pi runs locally; the child gets XKL_REMOTE_* env so the
    # extension reroutes GPU bash. No rcc push/pull is invoked by Python.
    final_text = '```json\n{"status": "success", "changed_files": ["src/a.py"]}\n```'
    captured_env: dict[str, str] = {}

    def fake_popen(cmd, *a, **k):
        captured_env.update(k.get("env") or {})
        # The command is local pi with -e; assert it is NOT an rcc-wrapped argv.
        assert cmd[0] == "fake-pi"
        assert "-e" in cmd
        assert str(_remote_gpu_extension_path()) in cmd
        return _FakeProc(_ndjson_lines(final_text), returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, *a, **k: run_calls.append(list(cmd)) or SimpleNamespace(
            returncode=0, stdout="", stderr=""
        )
    )

    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "fake-pi",
            "--remote",
            "sgs-gpu07",
            "--rcc-cmd",
            "rcc",
            "--quiet",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    # Python never invokes rcc push/pull — the extension does that per-command.
    assert run_calls == []
    # The extension is configured via env passed to the child pi process.
    assert captured_env.get("XKL_REMOTE_PROFILE") == "sgs-gpu07"
    assert captured_env.get("XKL_REMOTE_RCC_CMD") == "rcc"
    out = json.loads(capsys.readouterr().out)
    assert out["remote"]["profile"] == "sgs-gpu07"
    assert out["agent"]["stdout_json"] == {"status": "success", "changed_files": ["src/a.py"]}


def test_implement_without_remote_is_unchanged(monkeypatch, capsys):
    # No --remote: plain local pi, no -e, no XKL_REMOTE_* env, no rcc anywhere.
    final_text = '{"status": "ok", "changed_files": []}'
    captured_env: dict[str, str] = {}

    def fake_popen(cmd, *a, **k):
        captured_env.update(k.get("env") or {})
        assert cmd[0] == "fake-pi"
        assert "-e" not in cmd
        return _FakeProc(_ndjson_lines(final_text), returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    rc = main(
        [
            "implement",
            CANONICAL_TARGET,
            "--agent-cmd",
            "fake-pi",
            "--quiet",
            "--out",
            "stdout",
        ]
    )
    assert rc == 0
    assert captured_env == {}  # no XKL_REMOTE_* env injected
    out = json.loads(capsys.readouterr().out)
    assert out["remote"] == {"enabled": False, "profile": None}
