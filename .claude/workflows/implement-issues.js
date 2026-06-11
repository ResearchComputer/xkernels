// implement-issues — scan open GitHub issues, implement each in an isolated
// worktree (TDD), validate locally AND on the MI300A (beverin) cluster, then push
// a branch, open a DRAFT PR, request review from a GitHub user, and comment the
// findings back on the issue.
//
// Invoke with the Workflow tool:
//   Workflow({ name: 'implement-issues' })                      // all open `enhancement` issues
//   Workflow({ name: 'implement-issues', args: { issues: [17, 20] } })
//   Workflow({ scriptPath: '.claude/workflows/implement-issues.js', args: { dryRun: true } })
//
// args (all optional):
//   issues:    number[]  explicit issue allow-list (overrides the label filter)
//   label:     string    issue label to scan when no explicit list (default 'enhancement')
//   reviewer:  string    GitHub login to request review from (default 'xzyao-agent')
//   draft:     boolean    open PRs as draft (default true)
//   onDevice:  boolean    require MI300A/beverin validation before shipping (default true)
//   dryRun:    boolean    scan + report the worklist only; do NOT implement or open PRs

export const meta = {
  name: 'implement-issues',
  description: 'Scan open GitHub issues, implement each in an isolated worktree (TDD + on-device validation), push a branch, open a draft PR, and request review',
  whenToUse: 'When you want to fan out one implementing agent per open issue and produce a reviewed draft PR per issue, with MI300A validation',
  phases: [
    { title: 'Scan', detail: 'gh issue list → drop issues already covered by a PR' },
    { title: 'Implement', detail: 'one worktree-isolated agent per issue: TDD → local + beverin validation → push → draft PR → request review → comment' },
  ],
}

const REPO = 'ResearchComputer/kernels'
const MAIN_REPO = '/home/xiayao/Documents/research/kernels'
const VENV = MAIN_REPO + '/.venv'

const REVIEWER = (args && args.reviewer) || 'xzyao-agent'
const LABEL = (args && args.label) || 'enhancement'
const DRAFT = !(args && args.draft === false) // default true
const ON_DEVICE = !(args && args.onDevice === false) // default true (decision: always validate on beverin)
const DRY_RUN = !!(args && args.dryRun)
const ONLY = args && Array.isArray(args.issues) ? args.issues : null

const SCAN_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          number: { type: 'integer' },
          title: { type: 'string' },
          reason: { type: 'string', description: 'one line: why this issue needs work' },
        },
        required: ['number', 'title', 'reason'],
      },
    },
    skipped: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          number: { type: 'integer' },
          reason: { type: 'string', description: 'which PR already addresses it' },
        },
        required: ['number', 'reason'],
      },
    },
  },
  required: ['issues', 'skipped'],
}

const RESULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    issue: { type: 'integer' },
    branch: { type: ['string', 'null'] },
    prUrl: { type: ['string', 'null'] },
    shipped: { type: 'boolean', description: 'true iff a PR was opened' },
    testsPassed: { type: 'boolean', description: 'ruff + CPU/interpreter pytest gate' },
    onDevicePassed: { type: ['boolean', 'null'], description: 'beverin/MI300A correctness; null if not run' },
    summary: { type: 'string', description: 'what shipped, in 1-3 sentences' },
    notes: { type: 'string', description: 'caveats, negative results, follow-ups' },
  },
  required: ['issue', 'shipped', 'testsPassed', 'summary', 'notes'],
}

function scanPrompt() {
  const base =
    `Identify the GitHub issues in ${REPO} that need implementation. Use the gh CLI.\n\n` +
    `Run:\n` +
    (ONLY
      ? `  For each of these explicit issue numbers ${JSON.stringify(ONLY)}: gh issue view <N> --repo ${REPO} --json number,title,body\n`
      : `  gh issue list --repo ${REPO} --state open --label '${LABEL}' --json number,title,body,assignees --limit 100\n`) +
    `  gh pr list --repo ${REPO} --state all --json number,state,title,body,closingIssuesReferences,headRefName --limit 300\n\n` +
    `An issue is "already addressed" (skip it) if ANY open-or-merged PR references it via:\n` +
    `  - closingIssuesReferences containing the issue number, OR\n` +
    `  - a headRefName containing 'issue-<N>', OR\n` +
    `  - the PR title or body mentioning '#<N>' or '(issue #<N>)'.\n` +
    `Be conservative: when in doubt that a PR already covers an issue, put it in 'skipped', not 'issues'.\n\n`
  const rule = ONLY
    ? `Return ALL of the explicit issue numbers in 'issues' (do not drop them even if a PR exists — but note any existing PR in their 'reason'). Leave 'skipped' empty.\n`
    : `Return open '${LABEL}' issues that are NOT already addressed in 'issues'; put already-addressed ones in 'skipped' with the covering PR number.\n`
  return base + rule + `Return JSON only — do not implement anything.`
}

function implementPrompt(issue) {
  const slugHint = `<short-kebab-slug>`
  return `You are implementing GitHub issue #${issue.number} of ${REPO} end-to-end.

TITLE: ${issue.title}

You are running in your OWN isolated git clone/worktree — your edits do not affect other agents. Your CWD is the repo root of this worktree. The main checkout (with the prepared Python env) is at ${MAIN_REPO}.

== CONVENTIONS (read first) ==
- Global + repo instructions: ${MAIN_REPO}/CLAUDE.md, /home/xiayao/.claude/CLAUDE.md, and docs/ in this repo.
- Kernels are Triton (@triton.jit) targeting AMD MI300A (gfx942). Python package 'xkernels' lives under src/. Tests live in tests/.
- PRs are squash-merged. Use conventional commit titles: 'feat(<area>): ...', 'fix(<area>): ...', or 'bench(<area>): ...'.
- TDD: write a failing test FIRST, then the minimal code to make it pass.
- Be honest about negative results (see docs/issue-12-* and docs/issue-20-* for the precedent: a correct optimization that the hardware does not reward is shipped opt-in/off and documented, not hidden).

== STEP 1 — Understand ==
Run: gh issue view ${issue.number} --repo ${REPO} --comments
Read the source files it references. Write down a concrete mini-spec: acceptance criteria, shapes/dtypes, and how you will test it.

== STEP 2 — Branch off latest main ==
  git fetch origin main
  git switch -c <type>/issue-${issue.number}-${slugHint} origin/main
(type = feat for new kernels/features, fix for bug fixes, bench for benchmarks/characterization.)

== STEP 3 — TDD implementation ==
Add failing test(s) under tests/ encoding the acceptance criteria, then implement the minimal kernel/fix under src/xkernels/... to pass them. Match surrounding style; keep files focused.

== STEP 4 — Local validation gate (MUST pass before continuing) ==
Use the prepared venv at ${VENV} (the fresh worktree has no .venv of its own):
  ${VENV}/bin/python -c 'import numpy' || VIRTUAL_ENV=${VENV} uv pip install numpy   # Triton interpreter needs numpy
  ${VENV}/bin/ruff check .
  PYTHONPATH=$PWD/src TRITON_INTERPRET=1 ${VENV}/bin/python -m pytest tests -q
All must pass. If anything fails, fix it and rerun. Set testsPassed accordingly.

== STEP 5 — On-device validation on beverin / MI300A (${ON_DEVICE ? 'REQUIRED' : 'optional'}) ==
${ON_DEVICE
      ? `This is mandatory before opening the PR. First READ ${'/home/xiayao/Documents/xzyao/skills/clusters/cscs/README.md'} and the existing slurm/*_beverin.sbatch scripts in this repo (working templates: partition mi300, account a-infra02, --environment=tokenspeed-rocm-aiter-myofi).
  - rsync THIS worktree to a UNIQUE per-issue scratch path so concurrent agents do not clobber each other, e.g. /capstor/scratch/cscs/xyao/kernels-issue-${issue.number}
  - Write or adapt a slurm/<name>_beverin.sbatch for your kernel/test; submit it with REPO pointing at your unique scratch path: sbatch --export=ALL,REPO=<scratch-path> slurm/<name>_beverin.sbatch
  - Poll the job's .out log until the job leaves the queue; parse correctness (bf16 atol/rtol ~ 2e-2, i.e. max|err| within tolerance) and any perf numbers.
  - GATE: set onDevicePassed=true only if on-device correctness passes. If the cluster is unreachable or it fails, set onDevicePassed=false and record exactly what happened in notes — still open a DRAFT PR documenting the state rather than claiming success.`
      : `Skipped (onDevice=false). Set onDevicePassed=null.`}

== STEP 6 — Commit & push ==
  git add -A
  git commit   (conventional title + body; end the message with:
                Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>)
  git push -u origin <branch>

== STEP 7 — Open PR (${DRAFT ? 'DRAFT' : 'ready'}) + request review ==
  gh pr create --repo ${REPO} ${DRAFT ? '--draft ' : ''}--base main --head <branch> \\
    --title '<conventional title>' \\
    --reviewer ${REVIEWER} \\
    --body '<summary; Test Plan with the local AND beverin results; "Addresses #${issue.number}"; end the body with:
            🤖 Generated with [Claude Code](https://claude.com/claude-code)>'
If the on-device step failed or the optimization did not help, say so plainly in the body and keep the feature opt-in/off by default.

== STEP 8 — Comment findings on the issue ==
  gh issue comment ${issue.number} --repo ${REPO} --body '<concise findings: what shipped, local + on-device numbers, caveats, and the PR link>'

Return the structured result (issue number, branch, prUrl, shipped, testsPassed, onDevicePassed, summary, notes).`
}

// ---- run ----

phase('Scan')
const scan = await agent(scanPrompt(), { schema: SCAN_SCHEMA, label: 'scan-issues' })
let issues = (scan && scan.issues) || []
if (scan && scan.skipped && scan.skipped.length) {
  log(`Skipped (already addressed by a PR): ${scan.skipped.map((s) => '#' + s.number).join(', ')}`)
}
if (!issues.length) {
  log('No issues need work — nothing to implement.')
  return { implemented: [], scanned: scan }
}
log(`Worklist (${issues.length}): ${issues.map((i) => '#' + i.number).join(', ')}`)

if (DRY_RUN) {
  log('dryRun=true — reporting the worklist only; not implementing or opening PRs.')
  return { dryRun: true, worklist: issues, skipped: (scan && scan.skipped) || [] }
}

if (ON_DEVICE) {
  log('On-device beverin validation is ON — each agent queues a SLURM job on the single MI300A; these serialize on the cluster.')
}

phase('Implement')
const results = await parallel(
  issues.map((issue) => () =>
    agent(implementPrompt(issue), {
      label: `impl:#${issue.number}`,
      phase: 'Implement',
      isolation: 'worktree',
      schema: RESULT_SCHEMA,
    })
  )
)

const ok = results.filter(Boolean)
const shipped = ok.filter((r) => r.shipped)
log(`Done. ${shipped.length}/${issues.length} opened a PR.`)
return { results: ok, skipped: (scan && scan.skipped) || [] }
