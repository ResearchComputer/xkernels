# Design — how the language works

This directory is the authoritative reference for **how the shipped `vkl`
(Vibe Kernel Language) works**, written from the code. It is the counterpart to
[`docs/brainstorm/`](../../../docs/brainstorm/) — the brainstorm set is the
exploratory RFC (the *why* and the *what-if*); the docs here are the realized
design (the *what landed* and *where it lives*).

```
meta/docs/design/
├── README.md          ← you are here (the index)
└── vkl.md             ← how vkl works: surface, IR, lowering, schedule spine,
                         edit gate, cost model, MCP agent surface, graphs
```

## What vkl is (in one paragraph)

`vkl` is a **contract-native authoring DSL** that sits *above* the existing
substrate ([`../library.md`](../library.md)). One `@kernel` source emits the Op
Spec + the backend-neutral reference + the per-backend Impl Cards mechanically,
so they cannot drift; the same math IR lowers to a generated Triton kernel
*and* a torch CPU reference *and* (via per-target override bodies) native
CUDA/HIP. The contract stays the product; `vkl` is one producer among others —
never a gatekeeper (the §10 anti-goals hold). The IR is split into a **frozen
math oracle** (the WHAT) and an **editable schedule** (the HOW over a 6-level
hardware hierarchy), and the schedule is the source of truth that an agent edits
by name through the MCP surface to push a kernel toward the architecture ceiling.

## Where to start

| You want to… | Read |
|---|---|
| Understand how a `@kernel` becomes a verified kernel | [`vkl.md`](vkl.md) |
| The authoring surface (`@kernel` / `@targets` / `@launch`) | [`vkl.md` §2](vkl.md) |
| The two-layer IR (frozen math + editable schedule) | [`vkl.md` §3](vkl.md) |
| The lowering (mathbody → torch ref + Triton, the override path) | [`vkl.md` §4](vkl.md) |
| The schedule-IR spine (agent's source of truth, Phase A) | [`vkl.md` §5](vkl.md) |
| The edit gate + the MCP agent surface (Phase B) | [`vkl.md` §6–§7](vkl.md) |
| The cost model (scratch / roofline / occupancy) | [`vkl.md` §8](vkl.md) |
| Graph capture (`@graph`) | [`vkl.md` §9](vkl.md) |
| The rationale + open questions (the RFC) | [`../../../docs/brainstorm/`](../../../docs/brainstorm/) |
| The contract `vkl` emits into | [`../library.md`](../library.md) |

## Phasing status (what is shipped)

The brainstorm's [README](../../../docs/brainstorm/README.md) tracks the RFC
phases. As of this writing the **schedule-IR spine** (Phase A: schedule as the
source of truth in both directions) and the **MCP agent surface** (Phase B) are
shipped — these close the doc-09 "agent-editable IR" thesis end-to-end on the
Triton backend, with `input_precision` as the one concrete lever wired from an
agent edit through to what `tl.dot` compiles to. The remaining tracks (profile
feedback onto schedule nodes; HIP/MFMA codegen; trace compounding) are filed as
GitHub issues rather than open RFC questions.
