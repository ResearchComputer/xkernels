# Skills

Kernel-authoring / porting / tuning **skills** — reusable procedural knowledge in
the open [SKILL.md](https://github.com/kapilt/skill) format (a folder with a
`SKILL.md`: YAML frontmatter + markdown body). Consumable by any
skills-compatible agent (Claude Code, Codex, Gemini CLI, Cursor, Cline, …).

Per `meta/docs/library.md` §7: **cards are nouns (what exists); skills are verbs (how
to make/improve them).** Every skill is narrow on purpose (§7.2) — broad skills
mis-fire.

## Authoring surface (two roads to the same contract)

Every NEW op needs an Op Spec + a backend-neutral reference + Impl Cards. There
are two ways to produce them — **peers, not rivals** (the contract is
interchangeable; the DSL is never a gatekeeper, `meta/docs/library.md` §10):

- **DSL path** — [`author-a-kernel-with-dsl`](author-a-kernel-with-dsl/SKILL.md):
  one `@kernel` body builds the math IR that lowers to BOTH the auto-reference
  (torch, bit-exact) AND a generated Triton kernel. The spec + reference + cards
  are **emitted**. Faster, less boilerplate, but only for ops expressible as a
  fixed DAG of pointwise / reduce / MMA (gemm / norm / reduce / activation).
- **Hand path** — [`author-an-op-spec`](author-an-op-spec/SKILL.md): write the
  spec + reference + cards by hand across the eight artifacts. The fallback when
  the op needs data-dependent control flow, masking, scatter/gather, collectives,
  or a dequant the math IR can't spell — and always a valid choice.

Both gates are CPU-satisfiable (the reference passes `verify` on torch with no
GPU), so either is a productive first step on a CPU-only box.

## Frontmatter

Standard fields every agent reads: `name`, `description`, `license`. The
`description` is the trigger-selection field. Our library-specific metadata lives
in a namespaced `x-kernel-lib` block that non-standard consumers ignore (§7.1).

## Seeded skills

| Skill | Scope | When |
|---|---|---|
| [`handle-an-issue`](handle-an-issue/SKILL.md) | agnostic | the issue-driven ENTRY/DISPATCH skill — read an issue with `gh` (body AND comments), classify it, route to the matching kernel skill (or answer directly), then CLOSE with evidence if the acceptance condition is met or COMMENT + stay open if partial/blocked. Fires whenever an agent is handed an issue by number or URL |
| [`author-a-kernel-with-dsl`](author-a-kernel-with-dsl/SKILL.md) | agnostic | author a NEW op's whole contract (spec + reference + cards) from ONE `@kernel` source in the **vkl DSL** — the fast-path twin of `author-an-op-spec` for ops the math IR can express (gemm / norm / reduce / activation) |
| [`author-an-op-spec`](author-an-op-spec/SKILL.md) | agnostic | author the contract BY HAND — the fallback when the op is not math-IR-expressible (attention masking, scatter/gather, collectives), or the gateway when you prefer the eight-artifact path |
| [`tile-a-gemm`](tile-a-gemm/SKILL.md) | cuda, hip | build a tiled GEMM from primitives (the workhorse) |
| [`autotune-knob-sweep`](autotune-knob-sweep/SKILL.md) | agnostic | search the declared knob space, record the winner |
| [`port-cuda-to-hip`](port-cuda-to-hip/SKILL.md) | cuda→hip | functional port of a CUDA card to a HIP card |
| [`tune-for-cdna`](tune-for-cdna/SKILL.md) | hip | make a correct HIP card fast on CDNA |
| [`diagnose-wrong-results`](diagnose-wrong-results/SKILL.md) | cuda, hip | restore a kernel that CRASHES or fails verify on GPU after passing on the interpreter — runs BEFORE the perf-diagnose skills (which all require `verify().correctness.passed == true`) |
| [`diagnose-memory-bound`](diagnose-memory-bound/SKILL.md) | cuda, hip | fix a correct-but-bandwidth-limited kernel |
| [`diagnose-low-occupancy`](diagnose-low-occupancy/SKILL.md) | cuda, hip | fix a correct-but-latency/occupancy-limited kernel |
| [`establish-parity`](establish-parity/SKILL.md) | agnostic | cross-backend parity gate + divergence localization |
| [`use-rocprof-compute`](use-rocprof-compute/SKILL.md) | hip | run AMD ROCm Compute Profiler (Omniperf) on beverin to get the occupancy/stall/roofline profile the diagnose-* skills branch on |
| [`use-nsight-compute`](use-nsight-compute/SKILL.md) | cuda | run NVIDIA Nsight Compute (ncu) / Systems (nsys) on bristen (A100/sm_80) to get the occupancy/stall/roofline profile the diagnose-* skills branch on (the NVIDIA twin of use-rocprof-compute) |

## Evolution (governance loop, §7.3)

Every agent run emits a **skill outcome record**
`{skill_id, version, task_signature, result, iterations, final_tflops_vs_regime,
failure_mode?, run_id}`. These feed a continuous loop: score → promote → revise
→ split/merge → deprecate, with a frozen replay set gating every change so the
library can only get better, never silently worse. (Outcome-record infrastructure
is tracked separately; the skills here are the content.)
