# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Command line entry point for VKL agent workflows."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TargetRef:
    """Parsed ``module_or_path:symbol`` target for ``vkl implement``."""

    raw: str
    module: str
    symbol: str

    @property
    def short_name(self) -> str:
        return f"{self.module}:{self.symbol}"


def parse_target(raw: str) -> TargetRef:
    """Parse a target in ``module:function`` or ``path.py:function`` form."""
    module, sep, symbol = raw.partition(":")
    if not sep or not module or not symbol:
        raise ValueError(f"target must be '<module-or-path>:<symbol>', got {raw!r}")
    return TargetRef(raw=raw, module=module, symbol=symbol)


def _ensure_import_paths(cwd: Path) -> None:
    """Make repo root and ``src`` importable for local target probing."""
    for p in (cwd, cwd / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _module_path_from_ref(module_ref: str, cwd: Path) -> Path | None:
    """Map path-like and ``src.foo.bar`` refs to a concrete Python file if present."""
    pathish = module_ref.endswith(".py") or "/" in module_ref or os.sep in module_ref
    if pathish:
        path = Path(module_ref)
        if not path.is_absolute():
            path = cwd / path
        return path if path.exists() else None

    # User-facing examples often use ``src.foo.bar:fn`` to mean
    # ``./src/foo/bar.py:fn`` even though ``src`` is not a package.
    candidate = cwd / Path(*module_ref.split(".")).with_suffix(".py")
    if candidate.exists():
        return candidate
    return None


def _import_from_path(path: Path) -> Any:
    module_name = f"_vkl_target_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_target(target: TargetRef, cwd: Path) -> dict[str, Any]:
    """Best-effort target probe.

    A missing target is not fatal for ``implement``: the whole point may be to ask
    the agent to create it. When import succeeds and the symbol is a VKL kernel,
    include its spec and preflight result in the prompt.
    """
    _ensure_import_paths(cwd)
    out: dict[str, Any] = {
        "raw": target.raw,
        "module": target.module,
        "symbol": target.symbol,
        "importable": False,
        "exists": False,
    }
    try:
        path = _module_path_from_ref(target.module, cwd)
        module = (
            _import_from_path(path) if path is not None else importlib.import_module(target.module)
        )
        out["importable"] = True
        out["module_file"] = str(getattr(module, "__file__", "")) or None
        obj = getattr(module, target.symbol)
        out["exists"] = True
    except Exception as exc:
        out["probe_error"] = f"{type(exc).__name__}: {exc}"
        return out

    spec = getattr(obj, "_vkl_spec", None)
    if spec is None:
        out["vkl_kernel"] = False
        return out

    out.update(
        {
            "vkl_kernel": True,
            "spec_id": spec.id,
            "kernel": spec.kernel,
            "canonical_op": spec.canonical_op,
            "launch": getattr(spec.launch, "pattern", None),
            "targets": sorted(spec.targets),
        }
    )
    try:
        from .gate import validate_kernel

        out["validation"] = validate_kernel(spec).to_dict()
    except Exception as exc:  # pragma: no cover - defensive; validation is best-effort context
        out["validation_error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_request(
    args: argparse.Namespace,
    target: TargetRef,
    target_info: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON request embedded in the pi prompt."""
    return {
        "task": "vkl_implement",
        "target": target_info,
        "requested_backend": args.backend,
        "requested_arch": args.arch,
        "objective": args.objective,
        "verify": {
            "run_validate_kernel": True,
            "run_verify": args.verify,
            "run_verify_parity": args.verify_parity,
        },
        "output_contract": {
            "format": "json",
            "required_fields": [
                "status",
                "target",
                "changed_files",
                "validation",
                "commands_run",
                "blockers",
            ],
        },
        "remote": {"profile": args.remote} if args.remote else None,
        "notes": args.note,
    }


def build_prompt(request: dict[str, Any]) -> str:
    """Prompt sent to the coding agent in JSON output mode."""
    payload = json.dumps(request, indent=2, sort_keys=True)
    remote = request.get("remote")
    remote_note = ""
    if remote and remote.get("profile"):
        remote_note = (
            "\n"
            "Remote execution (YOU RUN LOCALLY): pi, your reads/edits/writes, and all\n"
            "non-GPU bash run on the LOCAL tree, which is the source of truth. A pi\n"
            "extension transparently reroutes GPU-bearing bash commands (verify,\n"
            "verify_parity, pytest, rocprof/ncu/nsys profiling, python touching\n"
            "xkernels/torch.cuda/triton) to the remote via rcc: it syncs your local\n"
            "edits first (rcc push), runs the command on the remote GPU, and streams\n"
            "the output back here as if local. You do NOT need to do anything\n"
            "different — just run GPU commands normally and read their stdout. To\n"
            "force a command to run locally, append ` # xk:local`; to force remote,\n"
            f"append ` # xk:remote`. Remote profile: {remote['profile']}.\n"
        )
    return (
        "You are implementing or repairing a VKL kernel in the xkernels repository.\n"
        "Follow AGENTS.md and the VKL contract: the contract is the product, and a "
        "kernel is not done until its DSL preflight passes and any available "
        "verify/verify_parity checks have been run.\n\n"
        f"{remote_note}\n"
        "Request JSON:\n"
        f"{payload}\n\n"
        "Implementation requirements:\n"
        "1. If the target module/function does not exist, create it in the requested location.\n"
        "2. Author or repair a VKL @kernel using the existing xkernels.vkl surface.\n"
        "3. Prefer emitted artifacts and existing helpers over hand-written registry drift.\n"
        "4. Run validate_kernel on the resulting KernelSpec.\n"
        "5. Run verify and verify_parity when the requested environment supports them; "
        "otherwise report the exact blocker.\n"
        "6. Return JSON only. Include status, target, changed_files, validation, "
        "commands_run, and blockers.\n"
    )


def build_agent_command(args: argparse.Namespace, prompt: str) -> list[str]:
    """Build the ``pi`` command. ``args.agent_cmd`` may include fixed flags."""
    cmd = shlex.split(args.agent_cmd)
    if not cmd:
        raise ValueError("--agent-cmd must not be empty")
    if args.provider:
        cmd += ["--provider", args.provider]
    if args.model:
        cmd += ["--model", args.model]
    if args.approve:
        cmd.append("--approve")
    cmd += list(args.agent_arg)
    cmd += ["--mode", "json", "--print", "--name", f"vkl implement {args.target}", prompt]
    return cmd


def _message_text(message: Any) -> str | None:
    """Concatenate text parts from a pi message (``content[*].text``)."""
    if not isinstance(message, dict):
        return None
    parts: list[str] = []
    for chunk in message.get("content") or []:
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            parts.append(chunk.get("text", ""))
        elif isinstance(chunk, str):
            parts.append(chunk)
    return "".join(parts) or None


def _last_assistant_text(messages: Any) -> str | None:
    """Newest assistant message text from an ``agent_end`` payload."""
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            text = _message_text(msg)
            if text:
                return text
    return None


def _strip_code_fence(text: str) -> str:
    """Strip a single surrounding ```...``` (or ```json) markdown fence."""
    s = text.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _extract_json(text: str | None) -> Any:
    """Best-effort parse of the JSON document the agent was asked to emit.

    pi's ``--mode json`` returns an NDJSON *event stream*; the agent's own
    answer is the text of the final assistant message (often wrapped in a
    ```json fence, sometimes with surrounding prose). This pulls that text and
    parses it, falling back to the first ``{...}`` object if prose sneaks in.
    """
    if not text:
        return None
    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _echo_event(ev: dict[str, Any], stream: Any) -> None:
    """Print a terse live view of one NDJSON event for the operator (stderr)."""
    etype = ev.get("type")
    if etype == "tool_execution_start":
        print(f"[tool:{ev.get('toolName') or '?'}] start", file=stream, flush=True)
    elif etype == "tool_execution_end":
        tag = "error" if ev.get("isError") else "ok"
        print(f"[tool:{ev.get('toolName') or '?'}] {tag}", file=stream, flush=True)
    elif etype == "message_update":
        ame = ev.get("assistantMessageEvent") or {}
        if ame.get("type") == "text_delta" and ame.get("delta"):
            stream.write(ame["delta"])
            stream.flush()
    elif etype == "turn_end":
        stream.write("\n")
        stream.flush()


def _agent_error(cmd: list[str], message: str) -> dict[str, Any]:
    return {
        "ran": False,
        "completed": False,
        "interrupted": False,
        "cmd": cmd,
        "returncode": None,
        "stdout_json": None,
        "final_text": None,
        "events": [],
        "stdout": "",
        "stderr": message,
    }


def _signal_group(proc: Any, sig: int) -> None:
    """Best-effort signal the agent's whole process group (kills grandchildren)."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except Exception:
        pass


def _terminate_tree(proc: Any) -> None:
    """SIGTERM the agent and any spawned children, then the process itself."""
    _signal_group(proc, signal.SIGTERM)
    try:
        proc.terminate()
    except Exception:
        pass


def _kill_tree(proc: Any) -> None:
    """SIGKILL the agent and any spawned children (final escalation)."""
    _signal_group(proc, signal.SIGKILL)
    try:
        proc.kill()
    except Exception:
        pass


def run_agent(
    cmd: list[str], cwd: Path, echo: bool = True, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Run the coding agent and return a machine-readable result wrapper.

    pi's ``--mode json`` emits an NDJSON event stream (one JSON object per line;
    see pi docs ``json.md``). We stream it line-by-line so progress is visible
    immediately and a partial result survives an interrupt, echo a terse live
    view to stderr when ``echo`` is set, and parse the agent's structured answer
    from the final assistant message (the ``agent_end`` event's
    ``messages[-1]``).
    """
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            start_new_session=True,
            env=env,
        )
    except FileNotFoundError as exc:
        return _agent_error(cmd, f"agent executable not found: {exc}")
    except OSError as exc:
        return _agent_error(cmd, f"failed to launch agent: {exc}")

    events: list[dict[str, Any]] = []
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    latest_text: str | None = None
    final_text: str | None = None
    interrupted = False

    def _drain_stderr() -> None:
        try:
            for line in proc.stderr:
                stderr_chunks.append(line)
        except Exception:
            pass

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()
    echo_stream = sys.stderr

    # Treat SIGTERM (e.g. `timeout`) the same as Ctrl-C so a killed run still
    # terminates the child and emits a partial result instead of dying hard.
    def _on_term(*_args: Any) -> None:
        raise KeyboardInterrupt

    installed_term: Any = None
    try:
        installed_term = signal.signal(signal.SIGTERM, _on_term)
    except (ValueError, OSError):
        # Not the main thread, or signals unsupported on this platform.
        pass

    try:
        for line in proc.stdout:
            stdout_chunks.append(line)
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(ev, dict):
                continue
            events.append(ev)
            if echo:
                _echo_event(ev, echo_stream)
            if ev.get("type") == "agent_end":
                text = _last_assistant_text(ev.get("messages") or [])
                if text:
                    final_text = text
            elif ev.get("type") == "message_end":
                msg = ev.get("message") or {}
                if msg.get("role") == "assistant":
                    text = _message_text(msg)
                    if text:
                        latest_text = text
    except KeyboardInterrupt:
        interrupted = True
        _terminate_tree(proc)
    finally:
        if installed_term is not None:
            signal.signal(signal.SIGTERM, installed_term)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.wait()
        drain.join(timeout=5)

    chosen = final_text or latest_text
    parsed = _extract_json(chosen)
    return {
        "ran": True,
        "completed": bool(final_text) and not interrupted,
        "interrupted": interrupted,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout_json": parsed,
        "final_text": chosen,
        "events": events,
        "stdout": "".join(stdout_chunks),
        "stderr": "".join(stderr_chunks),
    }


def _remote_gpu_extension_path() -> Path:
    """Path to the pi extension that reroutes GPU bash to the remote via rcc.

    The extension overrides pi's ``bash`` tool: non-GPU commands (and all
    read/edit/write) run locally on the authoritative tree, while GPU-bearing
    commands are ``rcc push``'d then executed on the remote via ``rcc run -s``
    with their output streaming back as if local. It is a no-op unless
    ``XKL_REMOTE_PROFILE`` is set, so loading it is always safe.
    """
    return Path(__file__).resolve().parents[3] / "extensions" / "remote-gpu.ts"


def _remote_env(args: argparse.Namespace) -> dict[str, str]:
    """Env configuring the remote-gpu extension for a ``--remote`` run."""
    return {
        "XKL_REMOTE_PROFILE": args.remote,
        "XKL_REMOTE_RCC_CMD": args.rcc_cmd,
        "XKL_REMOTE_NO_PUSH": "1" if args.no_push else "0",
        "XKL_REMOTE_NO_PULL": "0" if args.pull else "1",
        "XKL_REMOTE_PULL_PATHS": args.pull_paths,
        "XKL_REMOTE_PUSH_BEST_EFFORT": "1" if args.push_best_effort else "0",
        "XKL_REMOTE_PATTERNS": args.remote_patterns,
    }


def _slug(text: str) -> str:
    """Filename-safe slug for a target like ``module:symbol``."""
    slug = re.sub(r"[^0-9a-zA-Z._-]+", "-", text).strip("-")
    return slug or "target"


def _dist_result_path(cwd: Path, target: TargetRef) -> Path:
    """Timestamped result path under the gitignored ``dist/`` tree."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = _slug(target.short_name)
    return cwd / "dist" / "vkl-implement" / f"{slug}-{ts}.json"


def _summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Compact, pipe-friendly summary for stdout when the full blob goes to dist/."""
    agent = result.get("agent") or {}
    payload = agent.get("stdout_json")
    if isinstance(payload, dict) and payload.get("status"):
        status = payload["status"]
    elif not agent.get("ran"):
        status = "blocked"
    elif agent.get("interrupted"):
        status = "interrupted"
    elif result["ok"]:
        status = "ok"
    else:
        status = "failed"
    summary: dict[str, Any] = {
        "ok": result["ok"],
        "dry_run": result["dry_run"],
        "out": result["out"],
        "result_path": result.get("result_path"),
        "status": status,
    }
    if isinstance(payload, dict):
        for key in ("changed_files", "blockers"):
            if key in payload:
                summary[key] = payload[key]
    return summary


def cmd_implement(args: argparse.Namespace) -> int:
    cwd = Path(args.cwd).resolve()
    try:
        target = parse_target(args.target)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    target_info = resolve_target(target, cwd)
    request = build_request(args, target, target_info)
    prompt = build_prompt(request)
    pi_cmd = build_agent_command(args, prompt)

    remote_profile = args.remote
    echo = not args.quiet
    remote: dict[str, Any] = {"enabled": bool(remote_profile), "profile": remote_profile}

    # pi always runs LOCALLY (it edits the local, authoritative tree and all
    # read/edit/write stay local). With --remote, a pi extension is loaded (-e)
    # that overrides the ``bash`` tool: GPU-bearing commands are rcc-push'd then
    # executed on the remote via ``rcc run -s``, streaming output back as if
    # local; everything else runs locally. The extension is configured by env.
    exec_cmd = list(pi_cmd)
    child_env: dict[str, str] | None = None
    if remote_profile:
        ext = _remote_gpu_extension_path()
        if not ext.exists():
            print(
                json.dumps(
                    {"ok": False, "error": f"remote-gpu extension not found: {ext}"}
                ),
                file=sys.stderr,
            )
            return 2
        # ``-e <ext>`` must follow the agent binary (cmd[0]) and precede the
        # prompt positional. Insert right after the leading binary + any of its
        # fixed flags carried in --agent-cmd.
        exec_cmd = [pi_cmd[0], "-e", str(ext), *pi_cmd[1:]]
        remote["extension"] = str(ext)
        child_env = _remote_env(args)

    result: dict[str, Any] = {
        "ok": True,
        "dry_run": args.dry_run,
        "out": args.out,
        "request": request,
        "remote": remote,
        "agent": {"ran": False, "cmd": exec_cmd},
    }
    if args.include_prompt or args.dry_run:
        result["prompt"] = prompt

    if not args.dry_run:
        result["agent"] = run_agent(exec_cmd, cwd, echo=echo, env=child_env)
        agent = result["agent"]
        result["ok"] = (
            bool(agent.get("ran"))
            and not agent.get("interrupted")
            and agent.get("returncode") == 0
        )

    if args.out == "stdout":
        # Legacy: full result blob on stdout (pipe-friendly).
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        # Default: full result to the gitignored dist/ tree; a compact summary
        # (with the result path) goes to stdout so callers can jq it or cat it.
        result_path = _dist_result_path(cwd, target)
        result["result_path"] = str(result_path.relative_to(cwd))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(_summarize(result), sort_keys=True))
        print(f"result written: {result['result_path']}", file=sys.stderr)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vkl", description="VKL command line tools")
    sub = parser.add_subparsers(dest="command", required=True)

    impl = sub.add_parser(
        "implement",
        help="ask a JSON-mode coding agent to implement or repair a VKL kernel target",
    )
    impl.add_argument(
        "target",
        help="module_or_path:function, e.g. xkernels.vkl.examples.gemm_bf16:gemm_bf16",
    )
    impl.add_argument("--backend", default="triton", help="requested backend/card family")
    impl.add_argument("--arch", default="any", help="target arch hint, e.g. nvidia_sm90")
    impl.add_argument(
        "--objective",
        default="correctness_first",
        help="agent optimization objective",
    )
    impl.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    impl.add_argument("--verify-parity", action=argparse.BooleanOptionalAction, default=True)
    impl.add_argument("--note", action="append", default=[], help="extra instruction for the agent")
    impl.add_argument("--cwd", default=".", help="repository working directory")
    impl.add_argument("--agent-cmd", default="pi", help="agent executable plus fixed leading args")
    impl.add_argument(
        "--agent-arg",
        action="append",
        default=[],
        help="extra raw arg passed to agent",
    )
    impl.add_argument("--provider", help="pi provider name")
    impl.add_argument("--model", help="pi model id/pattern")
    impl.add_argument("--approve", action=argparse.BooleanOptionalAction, default=True)
    impl.add_argument("--dry-run", action="store_true", help="print request and pi command only")
    impl.add_argument("--include-prompt", action="store_true", help="include prompt in JSON output")
    impl.add_argument(
        "--out",
        choices=("dist", "stdout"),
        default="dist",
        help="where to write the full result JSON (default: dist/, which is gitignored)",
    )
    impl.add_argument(
        "--quiet",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="suppress live NDJSON event echo to stderr during the agent run",
    )
    impl.add_argument(
        "--remote",
        metavar="PROFILE",
        help=(
            "run pi LOCALLY but route GPU-bearing bash commands to the remote via "
            "rcc. The agent edits the local tree (source of truth); a pi extension "
            "reroutes verify/verify_parity/pytest/profiling/python-touching-xkernels "
            "commands to the remote GPU (rcc push, then rcc run -s), streaming "
            "output back as if local. VALUE is an rcc profile name "
            "(beverin/bristen/ds5/sgs-gpu07)."
        ),
    )
    impl.add_argument(
        "--no-push",
        action="store_true",
        help="with --remote, the extension skips the pre-command rcc push "
        "(only safe if the remote tree is already in sync)",
    )
    impl.add_argument(
        "--pull",
        action="store_true",
        help="with --remote, enable a best-effort post-command rcc pull of "
        "--pull-paths (default: off — verify results come back via stdout)",
    )
    impl.add_argument(
        "--pull-paths",
        default="",
        help="with --remote and --pull, comma/space-separated remote subpaths to "
        "pull after each GPU command (e.g. 'dist/,jobs/')",
    )
    impl.add_argument(
        "--push-best-effort",
        action="store_true",
        help="with --remote, a failed pre-command push warns and proceeds instead "
        "of failing the command (risk: a stale remote tree)",
    )
    impl.add_argument(
        "--remote-patterns",
        default="",
        help="extra newline-separated regexes that force a command onto the remote "
        "(appended to the built-in GPU detection set)",
    )
    impl.add_argument(
        "--rcc-cmd",
        default="rcc",
        help="rcc executable plus fixed leading args (default: rcc)",
    )
    impl.set_defaults(func=cmd_implement)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
