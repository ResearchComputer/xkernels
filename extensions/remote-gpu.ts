/**
 * remote-gpu — run pi LOCALLY, route GPU-bearing bash to a remote via rcc.
 *
 * Mental model: the agent (pi) and all its file reads/edits/writes plus
 * non-GPU bash run on the LOCAL tree, which stays the source of truth. Only
 * GPU-bearing bash commands (verify, verify_parity, pytest, rocprof/ncu/nsys
 * profiling, python touching xkernels/torch.cuda/triton, ...) are rerouted to
 * the remote: the extension `rcc push`-es the local edits first, then runs the
 * command on the remote via `rcc --profile <p> run -s '<cmd>'`, streaming the
 * output back to the agent as if it had run locally. The remote uses the GPU;
 * the local box needs no GPU.
 *
 * Config (env), all optional — the extension is a NO-OP unless
 * XKL_REMOTE_PROFILE is set, so loading it (`pi -e .../remote-gpu.ts`) is
 * always safe:
 *   XKL_REMOTE_PROFILE        rcc profile name (enables rerouting)
 *   XKL_REMOTE_RCC_CMD        rcc executable          (default: "rcc")
 *   XKL_REMOTE_NO_PUSH=1      skip the pre-command rcc push
 *   XKL_REMOTE_NO_PULL=1      skip the post-command rcc pull (default: 1)
 *   XKL_REMOTE_PULL_PATHS     space/comma-separated remote subpaths to pull
 *                             after each GPU command (e.g. "dist/ jobs/")
 *   XKL_REMOTE_PUSH_BEST_EFFORT=1  warn-and-proceed on push failure instead
 *                                  of failing the command
 *   XKL_REMOTE_PATTERNS       extra newline-separated regexes appended to the
 *                             built-in GPU detection set
 *
 * Per-command escape hatches (append anywhere in the command):
 *   ` # xk:local`   force this command to run locally
 *   ` # xk:remote`  force this command onto the remote
 *
 * Usage:
 *   XKL_REMOTE_PROFILE=beverin pi -e ./extensions/remote-gpu.ts
 *   # or via vkl:  vkl implement <target> --remote beverin --arch amd_cdna3
 */

import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createBashTool, type BashOperations } from "@earendil-works/pi-coding-agent";
import { buildDetector } from "./_gpu-detect.ts";

const PROFILE = process.env.XKL_REMOTE_PROFILE || "";
const RCC = process.env.XKL_REMOTE_RCC_CMD || "rcc";
const NO_PUSH = process.env.XKL_REMOTE_NO_PUSH === "1";
const NO_PULL = process.env.XKL_REMOTE_NO_PULL === "1";
const PULL_PATHS = (process.env.XKL_REMOTE_PULL_PATHS || "")
	.split(/[,\s]+/)
	.map((s) => s.trim())
	.filter(Boolean);
const PUSH_BEST_EFFORT = process.env.XKL_REMOTE_PUSH_BEST_EFFORT === "1";

const { isGpuCommand } = buildDetector(process.env.XKL_REMOTE_PATTERNS || "");

interface RccResult {
	code: number;
	stdout: string;
	stderr: string;
}

/** Run rcc with captured output (used for push / pull). */
function runRcc(args: string[], opts?: { cwd?: string }): Promise<RccResult> {
	return new Promise((resolve, reject) => {
		const child = spawn(RCC, args, {
			cwd: opts?.cwd ?? process.cwd(),
			stdio: ["ignore", "pipe", "pipe"],
		});
		const out: Buffer[] = [];
		const err: Buffer[] = [];
		child.stdout.on("data", (d) => out.push(d));
		child.stderr.on("data", (d) => err.push(d));
		child.on("error", reject);
		child.on("close", (code) =>
			resolve({
				code: code ?? -1,
				stdout: Buffer.concat(out).toString(),
				stderr: Buffer.concat(err).toString(),
			}),
		);
	});
}

/**
 * BashOperations that run each command on the remote via `rcc run -s`.
 *
 * `rcc --profile <p> run -s <script>` inserts <script> verbatim into a remote
 * `bash -lc` that `cd`s into the profile's remote_dir, so the agent's command
 * (pipes, quotes, &&, newlines) runs unchanged on the remote GPU box; its exit
 * code propagates as rcc's exit code. Output streams back over rcc's SSH
 * ControlMaster, so pi's bash tool sees it exactly like a local run.
 */
function remoteBashOps(): BashOperations {
	return {
		exec: (command, cwd, { onData, signal, timeout }) =>
			new Promise((resolve, reject) => {
				const child = spawn(RCC, ["--profile", PROFILE, "run", "-s", command], {
					cwd,
					stdio: ["ignore", "pipe", "pipe"],
				});
				let timedOut = false;
				const timer = timeout
					? setTimeout(() => {
							timedOut = true;
							child.kill("SIGKILL");
						}, timeout * 1000)
					: undefined;
				child.stdout.on("data", onData);
				child.stderr.on("data", onData);
				child.on("error", (e) => {
					if (timer) clearTimeout(timer);
					reject(e);
				});
				const onAbort = () => child.kill("SIGKILL");
				signal?.addEventListener("abort", onAbort, { once: true });
				child.on("close", (code) => {
					if (timer) clearTimeout(timer);
					signal?.removeEventListener("abort", onAbort);
					if (signal?.aborted) reject(new Error("aborted"));
					else if (timedOut) reject(new Error(`timeout:${timeout}`));
					else resolve({ exitCode: code });
				});
			}),
	};
}

function notice(msg: string): void {
	// TUI/RPC have richer status; json/print modes have none. stderr works everywhere.
	process.stderr.write(`[remote-gpu] ${msg}\n`);
}

export default function (pi: ExtensionAPI) {
	if (!PROFILE) return; // no-op unless configured

	const localCwd = process.cwd();
	const localBash = createBashTool(localCwd);
	const remoteBash = createBashTool(localCwd, { operations: remoteBashOps() });
	let routed = false;

	pi.on("session_start", async () => {
		notice(`enabled — GPU bash reroutes to rcc profile '${PROFILE}'`);
	});

	// Override the bash tool: GPU -> remote (push, run, pull); else local.
	pi.registerTool({
		...localBash,
		description: `${localBash.description} (GPU commands auto-route to remote '${PROFILE}' via rcc)`,
		async execute(toolCallId, params, signal, onUpdate) {
			const cmd: string = params.command;
			if (!isGpuCommand(cmd)) {
				return localBash.execute(toolCallId, params, signal, onUpdate);
			}

			if (!routed) {
				notice(`first GPU command rerouted to '${PROFILE}': ${cmd.slice(0, 80)}${cmd.length > 80 ? "…" : ""}`);
				routed = true;
			}

			// Sync local edits -> remote so the GPU sees the latest source.
			// Failure is fatal by default (a stale remote gives wrong/failed
			// verify); --push-best-effort downgrades it to a warning.
			if (!NO_PUSH) {
				const r = await runRcc(["--profile", PROFILE, "push"], { cwd: localCwd });
				if (r.code !== 0) {
					const msg = `rcc push failed (rc=${r.code}); remote tree may be stale.\n${(r.stderr || r.stdout).trim()}`;
					if (!PUSH_BEST_EFFORT) throw new Error(msg);
					notice(msg);
				}
			}

			const result = await remoteBash.execute(toolCallId, params, signal, onUpdate);

			// Best-effort scoped pull of GPU-produced artifacts (logs, dist/).
			// Off by default — verify/verify_parity results come back via stdout.
			if (!NO_PULL && PULL_PATHS.length) {
				await runRcc(["--profile", PROFILE, "pull", ...PULL_PATHS], { cwd: localCwd }).catch(
					(e) => notice(`pull failed (ignored): ${String(e)}`),
				);
			}
			return result;
		},
	});
}
