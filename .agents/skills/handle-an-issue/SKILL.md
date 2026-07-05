---
name: handle-an-issue
description: >
  Drive a GitHub issue to resolution: read it with `gh` (body AND comments), classify
  it, route to the matching kernel skill (or answer directly if it is a question/doc),
  act, then CLOSE it with evidence if the acceptance condition is met or COMMENT and
  leave it OPEN if it is only partial / blocked / needs-info. This is the issue-driven
  ENTRY/DISPATCH skill — it does not do kernel math itself; it picks the right skill
  (author-an-op-spec / port-cuda-to-hip / tune-for-cdna / diagnose-wrong-results /
  establish-parity / autotune-knob-sweep / the profile skills) and enforces the close-
  vs-comment discipline that prevents "resolved but still open" and "closed without
  proof". Fires whenever an agent is handed an issue by number or URL ("handle #42",
  "look at this issue", a bare URL to /issues/N). CPU-satisfiable for the routing +
  non-kernel cases; kernel cases inherit their routed skill's GPU gate.
license: Apache-2.0
x-kernel-lib:
  id: handle-an-issue@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "an agent is handed an issue by number or URL ('handle #42', 'work on issue 17', a bare github.com/.../issues/N link)"
      - "a task says 'triage the open issues' or 'go through the issue backlog'"
      - "an issue is referenced from a card's provenance.issue or a PR body and the agent needs to act on it"
    preconditions:
      - "`gh` is installed and authenticated (`gh auth status` is OK)"
      - "the issue is OPEN (a CLOSED issue is review-only — read it, do not reopen unless the task explicitly says so)"
      - "shell is in the repo root so `gh` resolves ResearchComputer/xkernels, OR `-R ResearchComputer/xkernels` is passed"
  inputs_required:
    - "the issue number or URL"
    - "(optional) `-R <owner/repo>` if not running from the repo root"
  tools:
    - "gh (issue view / comment / close)"
    - find_impl
    - get_op_spec
    - verify
    - verify_parity
  validation:
    must_pass:
      - "the FULL thread was read — `gh issue view <N> --comments`, not just the body — because later comments routinely supersede the original ask"
      - "the issue's ask is re-stated as a CHECKABLE acceptance condition BEFORE any work begins (close-vs-comment is decided against this, not by feel)"
      - "every kernel change referenced in the resolution passes `verify` + `verify_parity` — the hard rule, AGENTS.md — and that result is quoted in the close comment"
      - "the issue is CLOSED only if the acceptance condition is fully met; partial / blocked / needs-info → COMMENT + stay OPEN (never close-as-unresolved)"
      - "the close or comment carries EVIDENCE (verify/verify_parity result, PR link `closes #N`, cards/files changed), never a bare 'done'"
      - "the issue number is linked back: PR body contains `closes #N`, branch name references #N, and any new/changed card's `provenance.issue` records it"
    # NOTE: this skill's own gate (read → route → close-or-comment) is
    # CPU-satisfiable; the kernel work it dispatches inherits the routed skill's
    # GPU gate. A non-kernel issue (question / doc / wontfix) closes on CPU.
  references:
    - "AGENTS.md (the hard rule: verify + verify_parity before any kernel is 'done')"
    - ".agents/skills/README.md (the Seeded-skills table — the routing source this skill dispatches against)"
    - ".agents/skills/author-an-op-spec/SKILL.md, .agents/skills/author-a-kernel-with-dsl/SKILL.md (coverage-gap / 'add op X' issues)"
    - ".agents/skills/port-cuda-to-hip/SKILL.md, .agents/skills/port-across-arch/SKILL.md (missing-backend / missing-arch issues)"
    - ".agents/skills/diagnose-wrong-results/SKILL.md (crash / wrong-results issues — runs BEFORE any perf skill)"
    - ".agents/skills/use-rocprof-compute/SKILL.md, .agents/skills/use-nsight-compute/SKILL.md + diagnose-low-occupancy / diagnose-memory-bound / tune-for-cdna / map-to-matrix-cores / autotune-knob-sweep (slow-on-arch issues)"
    - ".agents/skills/establish-parity/SKILL.md (cross-backend numeric drift issues)"
    - ".agents/skills/mixed-precision-convert/SKILL.md, add-epilogue-fusion/SKILL.md, fuse-elementwise-chain/SKILL.md (precision / fusion requests)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-07-03T00:00:00Z"
    supersedes: []
---

## Why this skill exists

Issues are this repo's task-delivery format. An agent handed `#N` needs a
deterministic procedure, not intuition: **read → classify → route → act → close-
or-comment**, where the last step is governed by an explicit acceptance condition
and the hard rule (AGENTS.md: every kernel change passes `verify` +
`verify_parity`). Without this skill, three failure modes recur:

1. **Acting on the title/body only.** The body is stale half the time; the real,
   refined ask lives in a later comment ("actually just bf16", "skip the fp32
   point", "this is really about the AMD card"). `gh issue view <N>` WITHOUT
   `--comments` hides exactly that.
2. **Closing without proof.** "Done, closing" with no verify result and no PR
   link is not resolution — it is a claim. The close comment must carry evidence.
3. **Leaving resolved work open / closing partial work.** Resolved-and-verified
   → close. Partial / blocked / needs-info → comment and STAY OPEN. Agents invert
   both: they close half-done work to "look productive" and leave fully-done work
   open "for review".

This skill is the entry/dispatch layer: it does no kernel math itself. It picks
the right kernel skill, enforces the read-the-whole-thread and close-with-
evidence disciplines, and routes non-kernel issues (questions, doc requests,
duplicates) to a direct answer + close.

## Procedure

1. **Read the FULL thread.** Not just the body — the comments:
   ```bash
   gh issue view <N> --comments                       # human-readable, full thread
   gh issue view <N> --json title,body,state,labels,  # machine-readable for routing
       assignees,comments
   ```
   Later comments routinely supersede the body. If the thread is long, scan the
   comments in order and note the *latest* statement of the ask — that is the
   live spec, not the original body. Check `state`: if it is already CLOSED, this
   is review-only (read it, do not act unless the task says "reopen and fix").

2. **Re-state the ask as a CHECKABLE acceptance condition, before any work.**
   Issues are often ambiguous ("make the norm faster", "add attention"). Pin the
   done-criteria as a predicate you can eval at the end, e.g.:
   - "add op X" → `get_spec('X@..')` no longer raises AND
     `verify('X.reference@..', arch='any').correctness.passed` is True.
   - "port to AMD" → `find_impl(..., target_arch='amd_cdna3')` returns an
     applicable card AND `verify` passes on that arch.
   - "fix the crash on shape S" → `verify` passes on S in the process order that
     used to crash.
   - "is X supported?" → a one-line answer; no code change.

   Write this down. Step 5 (close vs comment) is decided against it, not by vibe.

3. **Classify and route.** Map the issue to exactly one bucket and CONFIRM the
   routed skill's preconditions before doing the work — do not start a perf tune
   on a card whose `verify` fails (that is a `diagnose-wrong-results` case):

   | issue intent | confirm first | route to |
   |---|---|---|
   | "add op X" / coverage gap / no spec | `find_impl` returns no card, `get_spec` raises | `author-an-op-spec` (or `author-a-kernel-with-dsl` if math-IR-expressible) |
   | CUDA card exists, AMD missing | `find_impl` → `missing_backend` | `port-cuda-to-hip` then `tune-for-cdna` |
   | card exists on one arch, missing on a sibling | `find_impl` → `missing_arch` | `port-across-arch` |
   | crash / illegal-memory-access / wrong results on GPU | `verify().correctness.passed == False` | `diagnose-wrong-results` (BEFORE any perf skill) |
   | "slow on arch Y" / misses perf regime | `verify` PASSES, `perf.ms` bad | profile (`use-rocprof-compute` / `use-nsight-compute`) → `diagnose-low-occupancy` / `diagnose-memory-bound` → fix (`tune-for-cdna` / `map-to-matrix-cores` / `diagnose-memory-bound`) |
   | declared knobs, no measured winner for target | card has non-empty `specialization_knobs` | `autotune-knob-sweep` |
   | backends disagree numerically | `verify_parity` diverges | `establish-parity` |
   | take fp32 → bf16/fp16/fp8 | fp32 card exists | `mixed-precision-convert` |
   | fuse a chained/bias/norm epilogue | profile shows a short op after a heavy kernel | `add-epilogue-fusion` / `fuse-elementwise-chain` |
   | question / doc / usage / duplicate / wontfix | no code change implied | answer inline (step 5, no kernel skill) |

   Load the routed skill's `SKILL.md` with the `read` tool and follow its
   procedure — its `validation.must_pass` becomes this issue's gate.

4. **Act — and link the issue into the work.** Branch as `#N-<slug>`, and when
   you open a PR put `closes #N` (or `fixes #N`) in the body so GitHub wires the
   resolution. Record the issue number in any new/changed card's
   `provenance.issue`. For kernel changes, the hard rule applies unchanged:
   `verify` + `verify_parity` must pass before the work is "done" — capture their
   output (pass/fail + `max_abs_err` / `agree`) for the close comment.

5. **Decide close vs comment against the acceptance condition from step 2:**
   - **RESOLVED (acceptance condition met AND, for kernel work, verify +
     verify_parity green)** → close WITH a tight evidence comment. Do not leave
     resolved work open "for review":
     ```bash
     gh issue close <N> --reason completed \
       --comment "Done in #<PR>. verify('<op>.<backend>@<ver>', arch='<a>') passes (max_abs_err=<e>); verify_parity agrees (rtol=<r>). Cards: <list>."
     ```
   - **PARTIAL / BLOCKED / NEEDS-INFO / no-GPU** → COMMENT with what you did,
     what is left, and exactly what you need (GPU access on arch X, a shape, a
     tolerance source, a decision). STAY OPEN:
     ```bash
     gh issue comment <N> --body "Contract layer landed (reference card verify-passes on CPU, <files>). Blocked on a gfx942 node to compile/verify the HIP card — reassigning when beverin is free. Acceptance condition: <predicate>."
     ```
     The honest no-GPU branch (see `author-an-op-spec`) lands here: a CPU-verified
     reference card is real progress, not a resolution — comment it, keep open
     until the GPU gate fires.
   - **INVALID / DUPLICATE / WONTFIX** → close with reason and a one-line
     explanation; do not leave it to rot open:
     ```bash
     gh issue close <N> --reason not_planned --comment "Duplicate of #<M>."   # or --reason completed for a question that was answered
     ```

6. **(Triage many issues) loop steps 1–5 per issue.** For a backlog sweep:
   ```bash
   gh issue list --state open --limit 200 --json number,title,labels
   ```
   Classify each from title+labels first, then `gh issue view <N> --comments` only
   for the ones you will actually act on (full-thread reads are expensive at
   scale). Triage is allowed to COMMENT ("routing this to port-cuda-to-hip,
   confirmed CUDA card exists, AMD missing") without resolving — that is a valid
   outcome, not a no-op.

## Pitfalls

- **Reading the body without `--comments`.** The single most common miss. The
  body states the original ask; a later comment usually refines, narrows, or
  reverses it. `gh issue view <N>` alone is a partial read — always append
  `--comments`.
- **Starting the routed skill before confirming its preconditions.** A "make it
  faster" issue on a card whose `verify` FAILS is a `diagnose-wrong-results` case,
  not a tune case. The diagnose/perf split is hard: perf skills REQUIRE
  `verify().correctness.passed == true`. Check it first, route second.
- **Closing partial work.** If the acceptance condition is not fully met —
  especially the all-GPU-gates-must-fire reality of kernel work — COMMENT and
  stay open. Closing half-done work to "look done" is the cardinal sin here; the
  next agent sees CLOSED and assumes it shipped.
- **Closing without evidence.** "Fixed, closing" is a claim. The close comment
  must quote the `verify` / `verify_parity` result (pass + `max_abs_err` / agree
  + `rtol`) and/or link the PR. No proof, no close.
- **Leaving fully-resolved work open "for review".** The mirror error. If the
  acceptance condition is met and the gates are green, CLOSE — with the evidence
  comment. An open issue with a merged `closes #N` PR is just noise.
- **Treating a question as a work item.** Some issues just need an answer
  ("does this support grouped GEMM?"). Answer in a comment and
  `--reason completed` (or `not_planned` if the answer is "no, and we won't").
  Do not spin up a kernel skill for a question.
- **Forgetting the back-link.** A PR without `closes #N`, a card without
  `provenance.issue`, a branch named `fix-stuff` instead of `#42-fix-stuff` —
  each severs the issue→resolution trace. Wire the number in at branch/PR/card
  time, not as an afterthought.
- **Reopening by accident.** `gh issue close` on an already-closed issue is a
  no-op; `gh issue reopen` is a deliberate act. If the task is "look at #N" and
  #N is closed, read it and stop — do not reopen unless told to.
- **Acting across the wrong repo.** If not in the repo root, `gh` may resolve a
  fork or fail. Pass `-R ResearchComputer/xkernels` explicitly, or `cd` to the
  repo root first (`gh repo set-default` to pin it for the session).
