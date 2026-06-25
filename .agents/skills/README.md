# Skills

Kernel-authoring / porting / tuning **skills** — reusable procedural knowledge in
the open [SKILL.md](https://github.com/kapilt/skill) format (a folder with a
`SKILL.md`: YAML frontmatter + markdown body). Consumable by any
skills-compatible agent (Claude Code, Codex, Gemini CLI, Cursor, Cline, …).

Per `docs/library.md` §7: **cards are nouns (what exists); skills are verbs (how
to make/improve them).** Every skill is narrow on purpose (§7.2) — broad skills
mis-fire.

## Frontmatter

Standard fields every agent reads: `name`, `description`, `license`. The
`description` is the trigger-selection field. Our library-specific metadata lives
in a namespaced `x-kernel-lib` block that non-standard consumers ignore (§7.1).

## Seeded skills

| Skill | Scope | When |
|---|---|---|
| [`tile-a-gemm`](tile-a-gemm/SKILL.md) | cuda, hip | build a tiled GEMM from primitives (the workhorse) |
| [`autotune-knob-sweep`](autotune-knob-sweep/SKILL.md) | agnostic | search the declared knob space, record the winner |
| [`port-cuda-to-hip`](port-cuda-to-hip/SKILL.md) | cuda→hip | functional port of a CUDA card to a HIP card |
| [`tune-for-cdna`](tune-for-cdna/SKILL.md) | hip | make a correct HIP card fast on CDNA |
| [`diagnose-memory-bound`](diagnose-memory-bound/SKILL.md) | cuda, hip | fix a correct-but-bandwidth-limited kernel |
| [`establish-parity`](establish-parity/SKILL.md) | agnostic | cross-backend parity gate + divergence localization |

## Evolution (governance loop, §7.3)

Every agent run emits a **skill outcome record**
`{skill_id, version, task_signature, result, iterations, final_tflops_vs_regime,
failure_mode?, run_id}`. These feed a continuous loop: score → promote → revise
→ split/merge → deprecate, with a frozen replay set gating every change so the
library can only get better, never silently worse. (Outcome-record infrastructure
is tracked separately; the skills here are the content.)
