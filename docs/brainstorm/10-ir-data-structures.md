# 10 — The IR, concretely: node schemas, diffs, cost formulas, lowering

> `09` argued *why* the IR is agent-editable. This doc says *what the nodes are*,
> *what an edit looks like*, *what the cost model computes*, and *how lowering
> works* — concretely enough to implement. Everything here is grounded in the
> actual substrate: it reuses `registry/cost_model.py`, `registry/models.py`,
> `registry/archs.py`, and emits documents that `ImplCard.from_doc` ingests.

## 0. The hard constraint that shapes everything

The IR's output is **JSON that the existing substrate already validates.** That
means every IR object must lower to fields the schema knows (`arch.family`,
`arch.requires`, `arch.wave_size`, `arch.scratch.kind`, `specialization_knobs`,
`backend`, `perf.roofline`). The IR does not invent new contract vocabulary; it
*produces* the existing vocabulary from a richer editable representation. This is
`05` §5.3 ("must not rename the contract vocabulary") at the data-structure level.

Concretely, the mapping the emitter must hit:

| IR object | Emits into |
|---|---|
| `MapTo(level=L5, instruction=wgmma)` | `arch.requires: ["tensor_cores"]`, `arch.family: nvidia_sm90` |
| `Stage(space=scratch)` | `arch.scratch.kind: smem` (NVIDIA) / `lds` (AMD) |
| `Knob(name, choices)` | `specialization_knobs: {name: {type, choices}}` |
| the whole schedule | one `ImplCard` doc → `ImplCard.from_doc(...)` |
| a `@graph` | one Impl Card with namespaced `launch.graph` |

If the emitter can't hit that mapping, the IR is wrong, not the schema.

## 1. The math IR (frozen — the correctness oracle)

A small, typed algebra over the *what*. Every node carries a `dtype` and a
`shape` so type-checking is decidable. There are ~9 kinds (pointwise / reduce /
mma + the data-addressing family `Gather`/`Slice`/`Concat`); that's the whole
oracle:

```python
# src/xkernels/vkl/ir/math.py  (frozen dataclasses — the agent NEVER edits these)
from dataclasses import dataclass
from typing import Literal, tuple

@dataclass(frozen=True)
class TensorRef:
    name: str            # an input/output of the op, or an intermediate id
    dtype: str           # "fp32" | "bf16" | "fp16" | "fp8e4m3" | ...
    shape: tuple[int, ...]      # concrete, or symbolic dims
    subscript: tuple[str, ...]  # e.g. ("m","k") — Einstein-ish indexing

@dataclass(frozen=True)
class MMA:        # the only "heavy" op; everything else is pointwise/reduce
    a: TensorRef; b: TensorRef
    accum_dtype: str       # MUST equal numerics.reduce_dtype → checked

@dataclass(frozen=True)
class Reduce:
    x: TensorRef; op: Literal["sum","max","rsqrt"]; axis: int
    accum_dtype: str       # MUST equal numerics.reduce_dtype → checked

@dataclass(frozen=True)
class Pointwise:  # cast, scale, bias, activation, residual — unary/binary fn
    fn: str; args: tuple[TensorRef, ...]; out_dtype: str

@dataclass(frozen=True)
class Load:  x: TensorRef
@dataclass(frozen=True)
class Store: ref: TensorRef; val: "MathNode"

MathNode = MMA | Reduce | Pointwise | Load | Store | TensorRef
```

**Why so small:** the math IR only needs to express what the contract's reference
expresses — pure-torch pointwise/reduce/mma, plus the data-**addressing** family
(`Gather`/`Slice`/`Concat`, added 2026-07-02 — `06` A4 case (a)). The
addressing nodes are oracle-safe: their torch lowering is bit-exact with their
device lowering, and the index is an *input* tensor (no data-dependent control
flow). If an op needs data-**selection** (a gather whose index is a value
computed in the kernel, a sort, a top-k, an RNG), the math IR can't express it
→ that op falls back to a hand-written reference (the `06` A4 case-(c) line).
The algebra's *limits are the auto-reference's limits, made explicit.*

**The oracle property:** the CPU lowering of this algebra (§5) *is* the
`numerics.reference`. A schedule edit never touches it. `verify` checks a card
against the reference; with the IR, the reference *is* the math IR lowered to
torch — so the agent literally cannot make the reference drift by editing the
schedule.

## 2. The schedule IR (editable — the HOW, over L0–L5)

The node vocabulary from `09` §2, as frozen dataclasses with two field groups:
**structural** (what the edit changes) and **cost** (the closed-form annotation,
§4). Structural fields are what the author/agent edits; cost fields are derived
and re-annotated each edit.

```python
# src/xkernels/vkl/ir/schedule.py
from dataclasses import dataclass
from typing import Literal

Level = Literal["L0","L1","L2","L3","L4","L5"]
Space = Literal["register","scratch","dsmem","global","descriptor"]

@dataclass(frozen=True)
class Tile:
    id: str
    shape: tuple[int | str, ...]   # ints (concrete) or knob names (symbolic)
    level: Literal["L0","L2"]      # output tile (L0/CTA) or streaming tile (L2)
    # cost: bytes = prod(shape)*dtype_bytes; occupancy contribution = 1 CTA per tile

@dataclass(frozen=True)
class MapTo:
    id: str
    op_ref: str            # → a MathNode id (the WHAT being scheduled)
    level: Level           # L4 (FMA) or L5 (matrix engine) typically
    instruction: str | None  # "wgmma" | "mfma" | "fma" | None (= compiler picks)
    instr_shape: tuple[int,...] | None  # native MMA shape, e.g. (64,128,16)
    # cost: peak_flops = ARCH_INSTR_PEAK[arch][instruction][dtype] (§4.1)

@dataclass(frozen=True)
class Stage:
    id: str
    producer_ref: str      # a Load/Tile it buffers
    space: Space           # register | scratch | dsmem
    depth: int | str       # pipeline stages (int or knob name)
    # cost: scratch_bytes = prod(tile)*depth; throughput = pipe-fill latency

@dataclass(frozen=True)
class CopyAtom:
    id: str
    src: Space; dst: Space         # e.g. global → scratch
    width: int | str               # vectorize lanes-wide (L4); 0 = auto
    swizzle: str | None            # None | "xor" | "pad" | ... (L4 bank-conflict policy)
    # cost: peak_bytes_per_s = ARCH_BW[arch][src,dst] * coalesce_factor(width,swizzle)

@dataclass(frozen=True)
class Reduce:
    id: str
    op_ref: str            # → a math Reduce id
    level: Literal["L3","L2","L0"]  # within-wave / within-CTA / cross-CTA
    # cost: latency = ARCH_REDUCE_LAT[arch][level]

@dataclass(frozen=True)
class Knob:
    name: str
    value: int | str       # current binding; must be ∈ declared choices
    choices: tuple[int,...]  # the declared specialization space

ScheduleNode = Tile | MapTo | Stage | CopyAtom | Reduce | Knob
```

Note the pattern: **every field that names hardware is `str | None` or a closed
enum** — never a free-form literal. `instruction="wgmma"` is legal;
`instruction="my_custom_asm"` is rejected at the check gate. There is no syntax
for "32 lanes"; the wave size is bound by the target, read from `archs.py`.

## 3. The edit primitives + the diff format

Each edit is a frozen dataclass with two methods: `check(ir, arch) -> Result`
(preconditions, §5) and `apply(ir) -> ScheduleIR` (returns a new frozen IR — no
in-place mutation, so the trace is a chain of immutable snapshots). The **diff**
is the JSON serialization of the edit + its measured outcome, which is what goes
into `provenance.tuning_trace`.

```python
# src/xkernels/vkl/edits.py
@dataclass(frozen=True)
class SetKnob:
    name: str; value: int
    def check(self, ir, arch): ...   # value ∈ ir.knobs[name].choices
    def apply(self, ir): ...          # returns ir with that knob bound

@dataclass(frozen=True)
class Retile:
    tile_id: str; shape: tuple[int,...]
    def check(self, ir, arch): ...   # shape divisible by arch L5 native shape (§5)
    def apply(self, ir): ...

@dataclass(frozen=True)
class MapTo_:
    op_ref: str; level: Level; instruction: str; instr_shape: tuple[int,...]
    def check(self, ir, arch): ...   # instruction legal for (arch,dtype,shape)
    def apply(self, ir): ...

@dataclass(frozen=True)
class AddStage:    stage_id: str; depth: int
@dataclass(frozen=True)
class SetCopyAtom: copy_id: str; width: int; swizzle: str | None
@dataclass(frozen=True)
class ReduceLevel: reduce_id: str; level: Literal["L3","L2","L0"]
@dataclass(frozen=True)
class PromoteOverride: target: str; arch: str   # H2 → H1 escape (09 §3)
```

**The trace entry** — this is the compounding artifact (`09` §6 step 7). It must
be JSON-serializable (goes in a namespaced provenance field) and token-compact
(an agent reads a whole trace to skip dead-ends):

```json
{
  "step": 4,
  "edit": "map_to",
  "target": "MapTo_mma_7",
  "args": {"op_ref":"mma_0","level":"L5","instruction":"wgmma","instr_shape":[64,128,16]},
  "check": "ok",
  "predicted": {"tflops": 740, "bottleneck": "compute"},
  "measured": {"ms_before": 0.42, "ms_after": 0.29, "tflops": 741, "roofline_pct": 0.75}
}
```

One line of intent, one line of outcome. An agent reading a 6-step trace reads
~6 of these — well inside a context window — and learns "step 2 (`add_stage` depth=4`)
was rejected for scratch overflow, don't retry." **That is the compounding loop
made of bytes an agent can actually parse**, not prose.

## 4. The cost model — grounded in the substrate's existing one

The substrate already has `registry/cost_model.py`: per-op `(flops, bytes)` and
per-arch ceilings `arch_peaks(arch) -> {fp32_tflops, dram_bw_gbs}`. **The IR's
cost model does not replace this; it composes with it.** The new pieces the IR
adds are (a) **per-instruction ceilings** (wgmma vs FMA — a 15× lever the
substrate's scalar-only `fp32_tflops` hides), and (b) **occupancy**, which the
substrate has zero of today.

### 4.1 Per-instruction peak tables (the new arch data)

Extend `archs.py` / a new `vkl/archdb.py` with matrix-engine ceilings. These are
*table data*, not formulas — same discipline as the existing `_ARCH_PEAKS`:

```python
# src/xkernels/vkl/archdb.py — extends registry/archs.py + cost_model.py
ARCH_INSTR_PEAK = {
  "nvidia_sm90": {        # bf16 peaks, TFLOPS
    "fma":    67.0,       # = arch_peaks("nvidia_sm90")["fp32_tflops"]  (scalar ceiling)
    "wgmma":  989.0,      # H100 SXM dense bf16 tensor-core ceiling
  },
  "amd_cdna3": {
    "fma":    80.0,       # = arch_peaks("amd_cdna3")["fp32_tflops"]
    "mfma":   1300.0,     # MI300 bf16 MFMA ceiling (32x32 instr family)
  },
  # ... per dtype: fp8 ≈ 2× bf16 on both; fp32 MMA on cdna3 via MFMA, etc.
}
ARCH_NATIVE_SHAPE = {     # the L5 shapes that divide L2 tiles (§5 check)
  "nvidia_sm90": {"wgmma": {"m":64, "k":16}},     # BLOCK_M % 64 == 0, BLOCK_K % 16 == 0
  "amd_cdna3":   {"mfma":  {"m":32, "k":16}},     # + the 16x16, 4x4 families
}
ARCH_SCRATCH_BYTES = {    # the L2 budget AddStage/Tile are checked against
  "nvidia_sm90": 228*1024,   # shared mem per CTA (H100)
  "amd_cdna3":   64*1024,    # LDS per workgroup (MI300A)
}
```

The cost of a `MapTo` node is then just a lookup: `peak = ARCH_INSTR_PEAK[arch][node.instruction]`.
This is what lets the agent **predict** that `map_to(mma, L5, wgmma)` is worth
~15× the `map_to(mma, L4, fma)` baseline before measuring anything.

### 4.2 The roofline aggregate (reuse `cost_model.py`)

For a whole schedule, predicted throughput is the standard roofline min, **using
the substrate's `cost_model(op_id, point)` for the workload's `(flops, bytes)`**:

```python
def predict(schedule, op_id, point, arch):
    flops, bytes_rw = cost_model(op_id, point)        # ← SUBSTRATE, reused
    compute_ceiling = sum(node_peak(n, arch) for n in schedule if isinstance(n, MapTo))
    bw_ceiling = arch_peaks(arch)["dram_bw_gbs"] * 1e9 * coalesce_eff(schedule)
    ai = flops / bytes_rw                               # arithmetic intensity
    return min(compute_ceiling, bw_ceiling * ai) / 1e12  # predicted TFLOPS
```

The bottleneck is whichever term wins the min — and that label
(`"compute"` vs `"memory"`) is exactly the `diagnose-memory-bound` /
`diagnose-low-occupancy` routing signal (`09` §7). **The cost model doesn't just
predict a number; it predicts the *skill the agent should fire next*.**

### 4.3 Occupancy (the genuinely new substrate contribution)

This is the part `cost_model.py` doesn't model at all, and it's where most
"correct but slow" kernels die. The IR computes a closed-form occupancy estimate
from register + scratch pressure, per arch:

```python
def occupancy(schedule, arch):
    regs_per_thread = sum(register_footprint(n) for n in schedule)  # from MapTo/Reduce/Stage
    scratch_bytes   = sum(scratch_footprint(n) for n in schedule)
    if arch == "nvidia_sm90":
        # 65536 regs/SM; warps = floor(65536 / (regs_per_thread*32)); ≤64 warps/SM
        warp_limit = min(64, 65536 // (regs_per_thread * 32))
        smem_limit = ARCH_SCRATCH_BYTES[arch] // max(1, scratch_bytes)  # CTAs/SM from smem
        return min(warp_limit, smem_limit) / 64   # fraction of max occupancy
    if arch == "amd_cdna3":
        # 512 VGPRs/SIMD; waves = floor(512 / vgprs); waves_per_eu 1..10
        wave_limit = min(10, 512 // vgprs_per_wave(schedule))
        lds_limit  = ARCH_SCRATCH_BYTES[arch] // max(1, scratch_bytes)
        return min(wave_limit, lds_limit) / 10
```

These formulas are approximations (the real occupancy calculator on each vendor
is more nuanced), but they're the right *shape* — and like §4 of `09`, the
profile feeds back: when `rocprof`/`ncu` report the true achieved waves/SM, that
number overwrites the prediction on the IR node, so the next estimate is
calibrated. **The occupancy model is wrong on a cold start and gets less wrong
every profile** — same compounding discipline as the rest.

## 5. The check gate (edits validated, not trusted)

Implemented as each edit's `check(ir, arch) -> Result`, where `Result` is
`Ok | Reject(reason)`. This is `08` §7's *checked* layer enforced at edit time.
The rules, each traceable to a §10 anti-goal:

| Check | Catches | §10 link |
|---|---|---|
| `BLOCK_M % native_shape.m == 0` | an L2 tile the L5 engine can't consume | reject `warp=32`-style hidden assumptions |
| `Reduce.accum_dtype == numerics.reduce_dtype` | bf16 accumulation when contract says fp32 | numerics drift |
| `Stage.space` legal for arch (`dsmem`/`descriptor` only sm_90+) | AMD card claiming smem-only feature | arch vocab honesty |
| `MapTo.instruction` ∈ `ARCH_INSTR_PEAK[arch]` | "wgmma" on an AMD target | vendor bake-in |
| `Knob.value ∈ Knob.choices` | undeclared specialization | validity surface |
| `sum(scratch) ≤ ARCH_SCRATCH_BYTES[arch]` | scratch overflow (compile-fail saved) | honest knobs |
| **math IR unchanged** after `apply` | an edit that silently changed the computation | the oracle property |

A reject returns a reason string the agent reads — the same `reject_reasons`
pattern `find_impl` uses (§3.2), now at the edit layer. The agent's failed
proposals are themselves training signal.

## 6. Lowering: schedule IR + math IR → target source

The lowering is a dispatch over `MapTo.instruction` per target. The math IR is
the same for every target; the schedule picks the instruction; the lowering emits
the spelling. Concretely, one math `MMA` node + one `MapTo(instruction=wgmma)`
lowers three ways:

| Target | Emission for `MMA` + `MapTo(wgmma)` |
|---|---|
| **triton** | `tl.dot(a, b, acc, allow_tf32=False)` — Triton picks the tensor-core instr |
| **cuda** (sm_90) | `wgmma.mma_async(...)` over a `cute::TmaDescriptor` (CUTE template emission) |
| **hip** (cdna3) | `__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)` (or Composable Kernel call) |

The **CPU lowering** of the same `MMA` is `torch.matmul(a.to(fp32), b.to(fp32))`
— and *that is the reference*. So one math node + N schedule choices → N device
lowerings + 1 reference, all type-checked against the same `MMA.accum_dtype`.
This is the `02`/`05` auto-reference guarantee at the level of individual nodes.

The lowering produces:
1. A compiled callable, registered via `register(kernel, Backend.X)(...)` — the existing `_dispatch.py`, unchanged.
2. An Impl Card JSON doc, ingested by `ImplCard.from_doc(...)` — the existing `models.py`, unchanged.
3. (For a `@graph`) host-side `cudaGraph*` / `hipGraph*` construction code + a card with namespaced `launch.graph`.

## 7. What this concretely reuses vs adds

| Substrate code | Reused by the IR as | Added by the IR |
|---|---|---|
| `registry/cost_model.py` | `predict()`'s `(flops,bytes)` + `arch_peaks` | per-instr peaks, occupancy model |
| `registry/archs.py` | the arch enum + vendor mapping | `ARCH_INSTR_PEAK`, `ARCH_NATIVE_SHAPE`, scratch budgets |
| `registry/models.py` | `ImplCard.from_doc` ingests emitted cards | the emitter itself |
| `registry/constraints.py` | the mini-language the `@kernel` header spells | nothing new |
| `src/xkernels/verify.py`, `retrieval.py` | unchanged consumers | nothing |
| `src/xkernels/_dispatch.py` | `register(...)` for emitted callables | nothing |

**The ratio matters:** most of the IR's correctness substrate is *reused*, not
rebuilt. The new code is the IR data structures (`vkl/ir/`), the edit primitives
(`vkl/edits.py`), the arch database extension (`vkl/archdb.py`), the cost model
extension, and the per-target lowering (`vkl/lower/`). That's a contained
surface — `11` sizes it.
