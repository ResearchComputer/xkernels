# vkl — how the Vibe Kernel Language works

`vkl` (the **Vibe Kernel Language**) is `xkernels`' contract-native authoring
DSL. This document is the authoritative reference for how it works **as shipped**,
grounded in the code under [`src/xkernels/vkl/`](../../../src/xkernels/vkl/). For
the design rationale and open questions, read the
[brainstorm](../../../docs/brainstorm/); for the contract `vkl` emits into, read
[`../library.md`](../library.md).

> **The hard rule (unchanged).** The contract is the product; `vkl` is one
> producer among others, never a gatekeeper. A DSL-authored op must still pass
> `verify` + `verify_parity` like any hand-written card. Portability lives in the
> contract, not the source — wave size is never hardcoded to 32.

## 1. The pipeline at a glance

```
            @kernel  src  (one source: the math + the launch pattern)
                │
        ┌───────┴────────┐
        ▼                ▼
   math IR (frozen)   schedule IR (editable)        ← the two-layer IR
        │                │
        ├─► torch CPU reference (the auto-oracle)
        ├─► generated Triton kernel   ─┐
        ├─► native CUDA override body  ├─► Impl Cards (per backend) ──► verify / verify_parity
        └─► HIP override body  (Phase D)┘
                                         ▲
   agent loop (MCP):  validate_kernel ─► load_schedule ─► list_legal_edits
                       ─► check_edit ─► apply_edit ─► read_cost ─► measure
```

One `@kernel` source produces the Op Spec, the backend-neutral reference, and the
Impl Cards **mechanically** — they cannot drift, because they are all derived
from the same math IR. The schedule IR is what an agent edits to push the kernel
toward the architecture ceiling.

## 2. The authoring surface (`surface.py`)

Three decorators declare a kernel:

| Decorator | Role |
|---|---|
| `@kernel(id=…, kernel=…, inputs=…, outputs=…, numerics=…)` | the header → Op Spec identity + the body |
| `@targets(triton=Target(arch=…, knobs=…), cuda=Target(…), …)` | per-backend specialization (the search space) |
| `@launch(Launch.tiled_2d() \| Launch.rowwise() \| Launch.elementwise())` | the device lowering pattern (the grid / parallelism) |

A worked example (abridged from [`examples/gemm_bf16.py`](../../../src/xkernels/vkl/examples/gemm_bf16.py)):

```python
@launch(Launch.tiled_2d())   # 2D grid over (M, N); the MMA's K dim -> the K-loop
@targets(triton=Target(      # the per-backend specialization search space
    backend="triton", arch="any", roofline="compute_bound", scratch_kind="smem",
    knobs={
        "BLOCK_M": (64, 128, 256), "BLOCK_N": (64, 128, 256), "BLOCK_K": (32, 64),
        "num_warps": (4, 8), "num_stages": (2, 3, 4),
    },
))
@kernel(
    id="gemm_bf16@1.0.0", kernel="gemm_bf16", canonical_op="gemm",
    inputs={"a": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("M", "K")),
            "b": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("K", "N"))},
    outputs={"out": TensorDecl(rank=2, dtype=(fp32, "bf16"), symbols=("M", "N"))},
    numerics=Numerics(reduce_dtype=fp32, rtol=2e-2, atol=1e-1, …),
    shape_sweep="gemm_bf16",
)
def gemm_bf16(ctx):
    """The body IS the computation — built as math IR via the ctx builder."""
    a = ctx.load("a")                       # [M, K]
    b = ctx.load("b")                       # [K, N]
    acc = ctx.mma(a, b, accum_dtype=fp32)   # [M, N] fp32  (the heavy op)
    out = acc.cast(ctx.out_dtype())         # cast to the output dtype
    ctx.store("out", out)

# A per-target override body reaches the vendor ceiling the portable Triton
# kernel cannot. Its math IR MUST match the trace (oracle property, checked).
@gemm_bf16.target("cuda", arch="nvidia_sm121")
def gemm_bf16_cuda(ctx):
    a, b = ctx.load("a"), ctx.load("b")
    ctx.store("out", ctx.mma(a, b, accum_dtype=fp32).cast(ctx.out_dtype()))
```

The body is **build-mode math-IR construction** (the `ctx` builder), not
executable Python — it lowers to `torch.matmul` (the reference) and Triton's
tiled `tl.dot` (the device). `spec_of(gemm_bf16)` recovers the `KernelSpec`
everything else derives from. The
three **launch patterns** select the lowering:

| Pattern | Grid | Math shape | Examples |
|---|---|---|---|
| `tiled_2d` | `(M/BM, N/BN)` programs, K-loop | an MMA + optional pointwise epilogue | `gemm_bf16` |
| `rowwise` | `(T,)` programs, one row each | one or more `Reduce` over the row | `rmsnorm`, `dual_rmsnorm`, `quant_fp8`, `softmax` |
| `elementwise` | flat | pure pointwise (+ optional gather/slice addressing) | `silu_and_mul`, `apply_rope`, `paged_kv_gather` |

A body with no `@launch` is a **direct torch computation** — its reference is the
body itself; it does not lower to a device kernel.

## 3. The two-layer IR

### 3.1 The frozen math IR — the oracle (`ir/math.py`)

The math IR is the **WHAT**: the pure computation, independent of how it is
scheduled on hardware. It is frozen — an edit can never reach it, so the
auto-reference cannot drift (the oracle property, by construction).

```python
MathNode = Load | Reduce | MMA | Pointwise | Store | Gather | Slice | Concat | Unsqueeze
```

- **`MMA`** (`out += a @ b`) — the only "heavy" op; what a `MapTo(L5, …)` schedules
  onto a matrix engine. `accum_dtype` must equal `numerics.reduce_dtype` (gate-checked).
- **`Reduce`** (`sum | max | rsqrt` over one axis) — the row-reduce workhorse.
- **`Pointwise`** (cast, scale, bias, mul, add, activation, …) — numerically exact;
  fusion of pointwise chains is always safe.
- **`Gather | Slice | Concat | Unsqueeze`** — the data-selection/addressing nodes
  (`paged_kv_gather`, `apply_rope`). *(Top-k / sort / RNG are out of scope by
  design — they are not math-IR-expressible.)*

`trace_ir(spec)` ([`reference.py`](../../../src/xkernels/vkl/reference.py)) returns
the `MathBody` (the math IR + in/out decls) for any trace body. The same `MathBody`
is what every lowering consumes.

### 3.2 The editable schedule IR — the HOW (`ir/schedule.py`)

The schedule IR is the **HOW**: how the math is mapped onto the 6-level hardware
hierarchy. This is the layer an agent edits.

```
L0 device | L1 cluster | L2 CTA | L3 wave/warp | L4 lane | L5 matrix engine
Spaces: register | scratch (smem/LDS) | dsmem | global | descriptor
```

```python
ScheduleNode = Tile | MapTo | Stage | CopyAtom | Reduce | Knob
```

| Node | Meaning |
|---|---|
| `Tile(id, shape, level)` | a tile of data. `shape` is concrete ints **or symbolic knob names** (e.g. `("BLOCK_M", "BLOCK_N")`), resolved at emit. Output/streaming tiles live at L2. |
| `MapTo(id, op_ref, level, instruction, instr_shape, precision)` | map a math node onto a level + (optional) instruction. `instruction="wgmma"/"mfma"` → L5 matrix engine; `"fma"`/`None` → L4. `precision` is the MMA's `input_precision` policy (§5). |
| `Stage(id, producer_ref, space, depth)` | a pipeline stage buffering a Tile in a memory space. `depth` is a concrete int or a knob name (resolved at emit). |
| `CopyAtom(id, src, dst, width, swizzle)` | a vectorized copy primitive between spaces. |
| `Reduce(id, op_ref, level)` | schedule a math `Reduce` onto L3 (within-wave) / L2 / L0. |
| `Knob(name, value, choices)` | a declared specialization point + its current binding. |

Every field that names hardware is `str | None` or a closed enum — never a
free-form literal. `instruction="wgmma"` is legal; `instruction="my_asm"` is
rejected at the gate. There is no syntax for "32 lanes"; wave size is bound by
the target (`archdb.py`), never remembered by a human.

`ScheduleIR` is a frozen, indexed bag of nodes; edits return a **new** `ScheduleIR`
(`with_node`), so the `tuning_trace` is a chain of immutable snapshots.

## 4. The lowering (`lower/`)

### 4.1 `mathbody.py` — one body, two outputs

The math IR lowers to **both** the torch CPU reference *and* the generated Triton
device kernel from the *same* `MathBody`:

- **Reference path:** the math IR evaluates on torch in `fp32` (with TF32
  disabled), so the reference is a true-fp32 oracle a GPU kernel must match
  bit-for-bit for fp32, not just within a loose tolerance.
- **Triton path:** `_TritonGen` emits a `@triton.jit` kernel per launch pattern.
  `launch(body, inputs, out_dtype, pattern=…, schedule=…, **knobs)` is the single
  host entry; it dispatches to `_launch_tiled_2d` / `_launch_rowwise` /
  `_launch_elementwise`.

`launch()` accepts either a flat `**knobs` binding (the substrate's
`verify(knobs=…)` autotune path) **or** a structured `schedule=` (the agent path,
§5) — both converge on the same resolved binding.

### 4.2 `triton.py` — dispatch + substrate wiring

`lower_to_triton(spec)` returns a host launcher whose signature matches the spec's
input order (so it is callable like a hand-written kernel). `register_dsl(spec)`
registers it with the dispatch registry and auto-wires the seeded input generator,
so `verify("gemm_bf16.triton@1.0.0", …)` runs the DSL kernel with zero JSON
hand-editing.

### 4.3 The override path (`override.py`, `lower/cuda.py`)

A per-target **override body** reaches the vendor ceiling that the portable
Triton kernel cannot. `check_override_math_ir` enforces that the override's math
IR is equivalent to the auto-trace (the oracle property holds — an override may
*tile differently* but may not *compute differently*); `emit_override_card`
emits the native card. Today the live native path is CUDA on GB10 (`sm_121`);
HIP/MFMA codegen is the open Phase D track.

### 4.4 The one MMA-level policy lever: `input_precision`

The Triton codegen responds to one MMA-level policy: `input_precision` on the
`MapTo` node. The default (`None`) is dtype-aware — bf16/fp8 use tensor cores,
fp32 uses `input_precision="ieee"` (true fp32, bit-faithful to the reference). A
`"tf32"` policy flips an fp32 GEMM to sm_80+ tensor cores (~10× faster, ~1e-3
precision). This is the one lever currently wired end-to-end from an agent edit
to what `tl.dot` compiles to (§5–§7).

## 5. The schedule-IR spine (Phase A) — the agent's source of truth

The schedule IR is the source of truth **in both directions**. This is the
closure of the [agent-editable-IR thesis](../../../docs/brainstorm/09-agent-editable-ir.md):

**Read-out** — `schedule_from_spec(spec, arch)` ([`schedule.py`](../../../src/xkernels/vkl/schedule.py))
builds a *structured* `ScheduleIR` from the spec + arch, not the knob-only bag
the old `schedule_from_card` produced:

- `tiled_2d` → output + two streaming `Tile`s, one L5 `MapTo` for the MMA, two
  scratch `Stage`s (depth tracking the `num_stages` knob *by name*), and the
  declared `Knob`s.
- `rowwise` → a wave-level (L3) `Reduce` node per math Reduce + Knobs.
- `elementwise` → Knobs.

The `MapTo.instruction` is the arch-native matrix engine (`wgmma` on `sm_90`,
`mfma` on `cdna3`, `None` on the portable `any` target — Triton picks at runtime).

**Read-in** — `resolve_binding(sched)` projects the (edited) schedule to the flat
`{name: value}` dict the launcher reads: knob values, resolved stage depths, and
the MMA `precision` flattened to `input_precision` (omitted when `None`).

**Convergence** — the agent path and the substrate path meet at the same launcher:

```
agent:   load_schedule → check_edit → apply_edit → resolve_binding ─┐
                                                                    ▼
substrate:  verify(knobs=…)  ──────────────────────────────►  launch()
                                                                    │
                                              _launch_tiled_2d pops input_precision
                                                    ▼
                                        _get_kernel(precision=…) → _TritonGen(precision=…)
                                                    ▼
                                        tl.dot(a, b, input_precision="tf32")
```

`launch(schedule=<edited>)` merges `resolve_binding(schedule)` with any explicit
`**knobs`; the launcher pops `input_precision` out of the binding (it is a codegen
setting, not a launch meta) and threads it to the codegen. So an agent's
`SetMapPolicy("tf32")` edit **changes what compiles**.

## 6. The edit gate (`edits.py`, `gate.py`)

Each edit primitive is a frozen dataclass with two methods:

```python
def check(self, ir: ScheduleIR, arch: str) -> Result   # locally decidable; no code runs
def apply(self, ir: ScheduleIR) -> ScheduleIR          # returns a NEW frozen IR
```

`Result` is `Ok()` or `Reject(reason)`. The reject reasons are **training signal**
— the agent reads them to skip dead-ends before applying.

| Primitive | What it does | The load-bearing check |
|---|---|---|
| `SetKnob(name, value)` | bind a declared knob | `value ∈ choices` |
| `Retile(tile_id, shape)` | resize a tile | **STATEFUL**: the divisibility check bites only once an L5 `MapTo` is present (the Phase 0 finding) |
| `MapTo_(map_id, op_ref, level, instruction, …)` | map an op onto a level + instruction | instruction legal for arch; `precision ∈ PRECISION_POLICIES`; concrete L2 M divisible by native m (symbolic dims deferred to emit) |
| `AddStage(stage_id, producer_ref, depth, tile_bytes)` | add a pipeline stage | scratch fits the arch budget |
| `SetMapPolicy(map_id, precision)` | tweak the MMA `input_precision` in one field | target node is an L5 `MapTo`; `precision ∈ PRECISION_POLICIES` |

The oracle property holds by construction: edits operate on `ScheduleIR` only;
the math IR lives on the `KernelSpec`. An edit literally cannot reach the
reference.

There are now two CPU-decidable gates in
[`gate.py`](../../../src/xkernels/vkl/gate.py), and they serve different layers:

| Gate | Input | Catches |
|---|---|---|
| `validate_kernel(spec, arch=...)` | a `KernelSpec` | emitted schema drift, undecidable constraints, trace failures, missing/duplicate output stores, `accum_dtype != numerics.reduce_dtype`, launch/node incompatibility, and declared Triton knobs the launcher does not consume |
| `run_gate(edits, schedule, arch)` | an edit sequence | illegal knob values, illegal instructions for the arch, tile/native-shape mismatch, scratch-budget overflow, and other schedule-edit preconditions |

`validate_kernel` is not a substitute for `verify` / `verify_parity`; it is the
cheap preflight that says the DSL contract is internally coherent before any GPU
compile. `verify` remains the proof that a concrete backend card matches the
reference on the shape sweep.

## 7. The MCP agent surface (Phase B, `mcp_server.py`)

The agent loop is exposed as **stateless** MCP tools — the agent carries its
state as an `applied_edits` list, and the server replays from the spec each call
(the schedule is a deterministic function of `spec + arch + edits`, so there is
no hidden server-side state to drift):

```
vkl_validate_kernel(spec_id, arch)                        → {passed, issues[], counts}
vkl_load_schedule(spec_id, arch)                          → structured schedule view
vkl_list_legal_edits(spec_id, arch, applied_edits)        → low-entropy next edits
vkl_check_edit(spec_id, arch, applied_edits, edit)        → {ok, reason?}
vkl_apply_edit(spec_id, arch, applied_edits, edit)        → {applied, schedule, applied_edits[]}
vkl_read_cost(spec_id, arch, applied_edits, point?)       → schedule + cost + legal_edits
```

The serialized schedule view is JSON-safe: nodes (id/kind/key fields), the knob
table, the resolved `binding`, and the MMA `precision`. Every field is a
primitive, so it crosses JSON cleanly to any MCP-speaking agent.

`vkl_list_legal_edits` enumerates conservative next moves from the current
schedule: alternate declared knob bindings, legal MMA `input_precision` policies,
and legal native-instruction remaps for the target arch. It does not replace
`vkl_check_edit`; it gives the agent a smaller search frontier before it proposes
custom edits.

`vkl_read_cost` returns the same schedule view plus a closed-form summary:
`scratch_bytes`, `overflows_scratch`, occupancy, the selected instruction, and,
when given a concrete `point`, a roofline prediction. The agent then runs the
edited kernel via the existing `verify` tool to measure — closing the loop.
*(Per-node cost annotation from ncu/rocprof — Phase C, [`profile.py`](../../../src/xkernels/vkl/profile.py): the plumbing landed in #74 and is CPU-tested; the live-profile confirmation is the remaining GPU gate. See §8.)*

### 7.1 CLI wrapper: `vkl implement`

The packaged CLI exposes the same agent-native flow from a terminal:

```bash
vkl implement "src.gemm.gemm_fp16:gemm" --arch nvidia_sm90 --backend triton
```

The target is `module_or_path:symbol`. If the module/symbol already imports and
is a VKL `@kernel`, the CLI includes its `KernelSpec` and `validate_kernel`
result in the request. If it does not exist yet, the missing-target probe is
included as context and the agent is instructed to create it.

By default the CLI invokes the local Pi coding agent in JSON mode:

```bash
pi --mode json --print --approve --name "vkl implement <target>" "<prompt>"
```

Useful flags:

```bash
vkl implement "src.gemm.gemm_fp16:gemm" --dry-run
vkl implement "xkernels.vkl.examples.gemm_bf16:gemm_bf16" --dry-run --include-prompt
vkl implement "src.gemm.gemm_fp16:gemm" --agent-cmd "pi --offline" --model google/gemini-pro
vkl implement "src.gemm.gemm_fp16:gemm" --no-verify-parity --note "only scaffold the DSL op"
```

The CLI itself returns a JSON wrapper with the request, the exact Pi command, and
the Pi JSON output when the agent runs. This keeps the top-level orchestration
machine-readable while still letting Pi own the code-editing loop.

## 8. The cost model (`cost.py`, `archdb.py`)

The cost model is closed-form and **decidable without running code** — so the
agent predicts before it measures (the bet that makes an LLM editor reliable).

| Function | What it predicts | Use |
|---|---|---|
| `predict_scratch` / `overflows_scratch` | shared-memory / LDS bytes for a tile×depth | prevents launch crashes (the `AddStage` gate) |
| `roofline(workload, arch)` | compute vs DRAM-bound, vs the vendor ceiling | 60–80% cold accuracy; routes to `diagnose-memory-bound` vs `tune` |
| `occupancy(workload, arch)` | active waves from smem/warp/register pressure | smem/warp closed-form, register pressure profile-calibrated |
| `roofline_gate` | is the achieved bw ≥ 70% of the vendor ceiling? | the publish bar for a "memory-bound" claim |

`archdb.py` is the architecture facts table: `legal_instructions(arch)`,
`native_shape(arch, instr)` (the MMA `m`/`k`), `scratch_budget(arch)`,
`instr_peak(arch, instr)`. These are what the gate's hardware-naming checks read.

**Phase C (profile feedback, [`profile.py`](../../../src/xkernels/vkl/profile.py),
issue #74)** bridges the closed-form PREDICTION above to the on-device
MEASUREMENT. `parse_ncu_report` / `parse_rocprof_compute` map a profiler's raw
text tables onto the cross-vendor §10 vocabulary (`ProfileMetrics`:
bottleneck / `dominant_stall` / `achieved_bw_pct` / `compute_throughput_pct` /
`tensor_pipe_util_pct` / `occupancy_fraction`); `route(metrics)` is the causal
diagnose-skill decision (the dominant stall reason, not the throughput ratio);
`annotate_schedule` keys one kernel-level profile to the schedule's node ids
(the `MapTo` node carries the bottleneck + stall; the `Stage`/`Tile` nodes carry
the load-pipeline bandwidth). `vkl_annotate_profile` / `vkl_route_from_profile`
expose it over MCP, and `diagnose-low-occupancy` reads `route_of(sched)` off the
`MapTo` node when an annotation is present. The plumbing is CPU-tested against
synthetic fixtures modeled on the profilers' real on-device numbers; confirming
the parsers against a LIVE `report.txt` / `analyze.txt` is the remaining GPU gate
(bristen sm_80 for ncu, beverin gfx942 for rocprof).

## 9. Graph capture (`graph.py`)

`@graph` captures a composition of kernels into one instantiated **CUDA/HIP
graph**, killing the per-launch overhead on the fused/short chains that dominate
real workloads. The graph IR has parameter nodes (so one captured graph serves
many shapes) and conditional nodes (a data-dependent branch). Measured 5–6.5×
over sequential on launch-bound chains. See [`examples/gemm_chain.py`](../../../src/xkernels/vkl/examples/gemm_chain.py).

## 10. File map

| Concern | File |
|---|---|
| Authoring surface (`@kernel`/`@targets`/`@launch`) | `vkl/surface.py` |
| Frozen math IR | `vkl/ir/math.py` |
| Editable schedule IR | `vkl/ir/schedule.py` |
| Math-IR → torch reference + Triton codegen | `vkl/lower/mathbody.py` |
| Triton dispatch + substrate wiring | `vkl/lower/triton.py` |
| Native CUDA override codegen | `vkl/lower/cuda.py`, `vkl/override.py` |
| **Schedule-IR spine (read-out + read-in)** | **`vkl/schedule.py`** |
| **Phase C: profile feedback onto nodes** | **`vkl/profile.py`** |
| Edit primitives + the gate | `vkl/edits.py`, `vkl/gate.py` |
| Cost model + arch facts | `vkl/cost.py`, `vkl/archdb.py` |
| Graph capture | `vkl/graph.py` |
| Knob sweep / autotune | `vkl/sweep.py` |
| Emit Op Spec / reference card / Impl Card JSON | `vkl/emit.py`, `vkl/artifacts.py` |
| Auto-reference registration | `vkl/auto.py`, `vkl/reference.py` |
| **MCP agent surface** | `src/xkernels/mcp_server.py` |
| Authored ops (the examples) | `vkl/examples/*.py` |

## 11. Relationship to the rest of the docs

- **The contract** ([`../library.md`](../library.md)) — `vkl` emits into this; it
  does not change it. The Op Spec / Impl Card / harness model is the bottom tier.
- **The RFC** ([`docs/brainstorm/`](../../../docs/brainstorm/)) — the *why*: `02`
  the thesis, `08` the programming model, `09` the agent-editable IR, `10` the
  concrete data structures, `11` the implementation plan.
- **Authoring a kernel** ([`../adding-a-kernel.md`](../adding-a-kernel.md)) — the
  card-driven checklist; the DSL fast-path is the `author-a-kernel-with-dsl` skill.
- **Open tracks** (filed as GitHub issues, not RFC questions): profile feedback
  onto schedule nodes (C) — **plumbing landed in #74**
  (`vkl/profile.py` + the `vkl_annotate_profile` / `vkl_route_from_profile` MCP
  tools), awaiting the live ncu/rocprof profile that confirms the parsers on a
  GPU; HIP/MFMA codegen + the H1/H2 edit-frequency count (D); persisted
  `tuning_trace` records carrying `{edit, predicted, measured, rationale}` for
  cross-task compounding (E).
