# xkernels documentation

xkernels is an **agent-native cross-platform GPU kernel library**: every op is a
backend-agnostic **Op Spec** plus one or more backend-specific **Implementation
Cards**, validated by a deterministic **harness**. This directory is the coherent
source of truth for the design, the usage, the per-kernel performance, and the
shared knowledge base.

```
meta/docs/
├── README.md               ← you are here (the index)
├── library.md              ← the design contract (read this if extending the substrate)
├── adding-a-kernel.md      ← the card-driven contribution checklist
├── design/                 ← how the vkl authoring language works (as shipped)
│   ├── README.md           ← the index
│   └── vkl.md              ← surface, two-layer IR, lowering, schedule spine,
│                            edit gate, cost model, MCP agent surface, graphs
├── usage/                  ← how to use the library
│   ├── README.md
│   ├── clusters.md         ← beverin (MI300A) + bristen (A100): bench + profile
│   └── ds5-testbed.md      ← GB10 (sm_121) CUTE-DSL testbed
└── kernels/                ← per-kernel-family optimizations & performance
    ├── README.md
    ├── gemm.md             ← bf16 dense (#17), fp8 blockscale (#38 portable, #41 mfma)
    ├── moe.md              ← INT4 W4A16, MXFP4 (#43), EP (#26), fused combine (#20)
    ├── attention.md        ← sparse MLA (#32), DSA indexer (#27), mxfp4 gather (#28)
    ├── mhc.md              ← prenorm gemm (#36), pre/post (#44), V4 perf pass (#39)
    ├── comm.md             ← hierarchical all-reduce (#12)
    └── norm.md             ← dual_rmsnorm, fused_ffn
meta/wiki/                  ← the shared knowledge base (benchmark/profile campaign,
                              roofline, gotchas, CUTE-DSL authoring reference)
```

## Where to start

| You want to… | Read |
|---|---|
| Understand the design / contract | [`library.md`](library.md) |
| Understand how the vkl authoring language works | [`design/vkl.md`](design/vkl.md) |
| Use or call a kernel | [`usage/README.md`](usage/README.md) |
| Add a new kernel | [`adding-a-kernel.md`](adding-a-kernel.md) |
| See what a kernel achieves + how | [`kernels/`](kernels/) (the family READMEs) |
| Run benchmarks / profiles on a cluster | [`usage/clusters.md`](usage/clusters.md), [`usage/ds5-testbed.md`](usage/ds5-testbed.md) |
| Learn the lessons (roofline, gotchas, CUTE authoring) | [`../wiki/`](../wiki/) |

## The three layers

1. **The contract** ([`library.md`](library.md) + [`adding-a-kernel.md`](adding-a-kernel.md))
   — the backend-agnostic Op Spec / Implementation Card model, the verification +
   parity harness, and the compounding loop. This is the bottom tier of
   consumption: "just read the JSON". It is the one thing that must stay stable;
   every skill, source file, and doc cross-references it by section number
   (`library.md §5`, `§10`, …).

2. **The language** ([`design/`](design/)) — how the `vkl` authoring DSL works as
   shipped: the `@kernel`/`@targets`/`@launch` surface, the frozen-math /
   editable-schedule two-layer IR, the lowering to torch reference + Triton (+
   native override), the schedule-IR spine that makes it the agent's source of
   truth, the edit gate, the cost model, and the MCP agent surface. The rationale
   + open RFC questions live in [`../../docs/brainstorm/`](../../docs/brainstorm/).

3. **Usage** ([`usage/`](usage/)) — the user/contributor surface: how to install,
   call, override, and contribute kernels, plus the cluster testbed runbooks.

4. **Per-kernel performance** ([`kernels/`](kernels/)) — the deep dives, one doc
   per kernel *family* (not one per issue). Each gives the math, the strategy, the
   measured performance tables, and the honest lessons — positive **and** negative
   results. The cross-cutting benchmark/profile campaign that produced most of the
   numbers is in [`../wiki/`](../wiki/).

## Provenance of the kernel docs

The `kernels/` docs consolidate the older per-issue write-ups
(`issue-12-…` … `issue-44-…`) into one coherent doc per family. Issue numbers are
kept inline as historical anchors (e.g. "issue #41") but no longer name separate
files. Each measured number still cites its on-device job id or reproducible
`meta/benchmarks/` / `tests/` path.

## Related locations

- `registry/` — the machine-readable artifacts: `ops/` (Op Specs), `impls/` (Impl
  Cards), `shape_sweeps/`, `schema/` (JSON Schemas).
- `src/xkernels/` — the runtime: kernels by type, the registry loader, retrieval,
  verify, the MCP server.
- `.agents/skills/` — the SKILL.md authoring/porting/tuning playbooks.
- `meta/benchmarks/` — per-op benchmark sweeps (see its README).
- `scripts/` — the rcc cluster toolkit + SLURM jobs.
- `tests/` — unit + integration tests.

## Document drift

`python meta/docs/check_document_drift.py` checks that the top-level public APIs
are documented and indexes test/benchmark/SLURM coverage (auto-writes
`DRIFT_CHECK_REPORT.txt`).
