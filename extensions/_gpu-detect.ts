/**
 * GPU-command detection for remote-gpu — pure, no pi imports, so it is unit
 * testable with jiti in isolation. Remote routing routes a bash command onto
 * the remote GPU box iff isGpuCommand(cmd) is true.
 *
 * Config knobs (env) are parsed here so the same surface is testable:
 *   XKL_REMOTE_PATTERNS  extra newline-separated regexes (appended to defaults)
 */

export const REMOTE_MARKER = "xk:remote";
export const LOCAL_MARKER = "xk:local";

/**
 * Commands that need a GPU. Conservative on purpose: git/ls/cat/ruff/rg/echo
 * never match. A CPU-only `python ... xkernels` may match and get rerouted
 * (harmless — just a push + ssh round-trip); force it local with `# xk:local`.
 */
export function defaultGpuPatterns(): RegExp[] {
	return [
		/\bverify(?:_parity)?\s*\(/, // verify(...) / verify_parity(...)
		/\bverify_parity\b/,
		/\bpytest\b/,
		/\brocprof(?:-compute)?\b/,
		/\bomniperf\b/,
		/\bncu\b/,
		/\bnsys\b/,
		/\bhipify\w*/,
		/\bhipcc\b/,
		/\btorch\.cuda\b/,
		/(?:^|[\s;&|(])python[0-9.]*(?:\s+-c)?\b[\s\S]*\b(?:xkernels|triton)\b/,
	];
}

/** Parse XKL_REMOTE_PATTERNS (newline-separated) into RegExps, skipping bad ones. */
export function parseExtraPatterns(raw: string): RegExp[] {
	return (raw || "")
		.split("\n")
		.map((s) => s.trim())
		.filter(Boolean)
		.map((src) => {
			try {
				return new RegExp(src);
			} catch {
				return new RegExp(src.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
			}
		});
}

export function buildDetector(extraRaw = ""): {
	patterns: RegExp[];
	isGpuCommand: (cmd: string) => boolean;
} {
	const patterns = [...defaultGpuPatterns(), ...parseExtraPatterns(extraRaw)];
	return {
		patterns,
		isGpuCommand: (cmd: string): boolean => {
			if (cmd.includes(LOCAL_MARKER)) return false;
			if (cmd.includes(REMOTE_MARKER)) return true;
			return patterns.some((re) => re.test(cmd));
		},
	};
}
