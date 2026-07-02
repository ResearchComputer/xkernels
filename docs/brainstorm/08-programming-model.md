# 08 — The programming model: tiling, SIMD, and the hardware hierarchy

This is the core design question. Tiling and SIMD are *the* hard part of any GPU
DSL, and the answer determines whether the DSL is "portable but capped" (Triton's
block-level model) or "fast but non-portable" (raw CUTLASS / Composable Kernel).
The v0.2 stance — **portable *and* fast, multi-target, perf day 1** — cannot pick
one; it has to *compose* them. This doc says how.

## 0. The core principle: a layered hierarchy, block-level by default

A GPU is not flat. There is a strict **execution + memory hierarchy**, and a
kernel's performance is *entirely* about how its logical tiling maps onto that
hierarchy. The DSL's programming model makes the hierarchy *explicit and named*,
then sets a default:

> **The portable body is written at the block level** (Triton-style: you think in
> tiles, the compiler maps them down the hierarchy). **A per-target override
> drops to the explicit-hierarchy level** (CUTE/Composable-Kernel-style: you name
> warps, lanes, the matrix engine, the swizzle) — and *that* is how you reach the
> vendor ceiling.

This is not a new idea grafted on; it is the **portable-body / per-target-override
split (`02` Layer 2/3) made concrete at the programming-model level.** The reason
the default *must* be block-level is precisely portability: you cannot write
`warp = 32` code and `wavefront = 64` code at the same level. The reason the
override *must* reach the explicit hierarchy is precisely performance: no
block-level abstraction has ever matched a hand-tuned CUTE/CK kernel on the
ceiling. So the DSL gives you block-level when portability matters and the
hierarchy when the ceiling matters — and the H1/H2 question (`06` A1) is
literally *"how often does reaching the ceiling require dropping levels?"*

## 1. The hierarchy, named

Six logical levels. Each maps to a concrete hardware object on NVIDIA and on AMD.
The §4.1 table from `library.md` is the bottom half of this.

| Logical level | What the author names | NVIDIA | AMD (CDNA) |
|---|---|---|---|
| **L0 device** | the launch grid | grid of CTAs | grid of workgroups |
| **L1 cluster** *(sm_90+)* | CTA-cooperation group | thread-block cluster (CGA), distributed shared mem | — *(not in CDNA3)* |
| **L2 CTA** | thread-block, owns a scratchpad | CTA (≤1024 threads) + shared memory | workgroup (≤1024 threads) + LDS |
| **L3 wave** | the SIMT execution group | **warp = 32 lanes** | **wavefront = 64 lanes** |
| **L4 lane** | one SIMT thread; *vectorizes within itself* | thread; 32-bit lanes, vectorizable to 128-bit loads | workitem; VGPR/SGPR split, vector ISA |
| **L5 matrix engine** | the FMA accelerator | tensor cores: wmma (per-warp) / **wgmma** (warp-group, sm_90) | **MFMA** (per-wavefront / half-wavefront) |

The two things that *change across vendors* are **L3 (32 vs 64)** and **L5
(instruction family + issuing unit)**. Everything in the programming model that
claims portability must operate *above* L3 (i.e., at L0–L2), and everything that
claims the ceiling must operate *at* L3–L5. **The default lives above L3; the
override lives at L3–L5.** That single rule is the whole model.

## 2. Each concept, expressed at the right level

### 2.1 Tiling (a *logical* concept → maps to L0/L2/L5)

Tiling happens at three different granularities, and conflating them is the
classic bug:

- **Output tiling (L0/L2)** — partition the output (`M×N`) into `BLOCK_M×BLOCK_N`
  tiles; each CTA owns one (or several). *Logical, fully portable.* This is the
  `@launch(grid=...)` and the outer two loop bounds.
- **Reduction/streaming tiling (L2)** — the K-loop sliced into `BLOCK_K` chunks
  that stream through the scratchpad. *Logical, portable; the pipeline depth
  `num_stages` is the knob.*
- **Matrix-engine tiling (L5)** — the MMA's native shape (`wgmma m=64,k=16`;
  `MFMA 16×16`, `32×32`, …). *Hardware-specific; must divide the L2 tile.* This is
  where `BLOCK_M % 64 == 0` constraints come from on sm_90, and the author must
  not assume them — the DSL checks `BLOCK_M` against the target's L5 shape.

**Expression:** tiles are first-class objects with a *layout*, not just shapes.
In the default body, the layout is inferred (Triton-style). In the override, the
layout is named:

```python
# default body (portable, L0/L2 only): tiles are just shapes
acc = zeros((BLOCK_M, BLOCK_N), fp32)              # layout inferred
for k in range(0, K, BLOCK_K):
    a_tile = stage_async(load(a[mi:mi+BLOCK_M, k:k+BLOCK_K]), into=scratch)
    acc = mma(a_tile, b_tile, acc, accum=fp32)     # L5 shape chosen by target

# override body (sm_90, L1/L2/L5): layout + ownership named
a_desc = ctx.tma_descriptor(a, tile=(BLOCK_M, BLOCK_K))   # L5-grade bulk copy
with ctx.cluster(2):                                      # L1
    acc = wgmma(a_desc, b_smem_tile, acc,                 # L5 native instr
                mma_shape=(64, BLOCK_N//4, 16))           # L5 shape pinned
```

### 2.2 SIMD / vectorization (a *within-lane* concept → L4)

A subtle but load-bearing distinction the model must name **separately** from L3:

- **L3 wave (32/64)** = the SIMT *lockstep group*. "Warp" vs "wavefront."
- **L4 vectorize** = how many elements *one lane* loads/operates per instruction
  (e.g. `float4` = 128-bit = 8 bf16). This is the actual "SIMD width."

Conflating these is a recurring bug source (people say "SIMD" meaning both).
The DSL uses **wave** for L3 and **vectorize** for L4, always.

**Expression:** the emitter *auto-vectorizes* loads/stores up to the arch's
natural width (128-bit on both vendors; wider on AMD for some paths) in the
default body — the author never says it. In the override, the author can pin it
for a specific copy atom (CUTE `Copy_Atom` style), which is occasionally needed
to dodge bank conflicts or hit a TMA shape. **Default: auto; override: optional
pin.** This is one of the cleanest wins of "block-level default" — vectorization
is a thing the compiler should own except when it's a ceiling lever.

### 2.3 Memory spaces (L2/L4/L5 → named, not magic)

Where a value lives is performance-defining and currently implicit per-kernel.
The model names **five spaces** and binds the portability vocabulary to them:

| Space | Lives in | Shared across | Default for |
|---|---|---|---|
| `register` | per-lane registers (VGPR) | one lane | accumulators (`acc`) |
| `scratch` / `smem` / `lds` | per-CTA scratchpad | the CTA's warps | staged tiles (`a_tile`) |
| `dsmem` | distributed shared mem (sm_90) | a cluster's CTAs | cluster-shared tiles (override only) |
| `global` | DRAM | everyone | inputs/outputs |
| `descriptor` | a TMA descriptor (sm_90) | a CTA | bulk-copy source (override only) |

`scratch` is the *abstract* name; its `kind` (`smem` on NVIDIA, `lds` on AMD) is
bound by the target — exactly the `arch.scratch.kind` field already in the Impl
Card schema. **The default body says `scratch`; the emitted card says `smem` or
`lds`.** No rename of the contract vocabulary (`05` §5.3).

### 2.4 Reductions (L3 + L2 → a primitive, per-target body)

Reductions decompose by level, and the *implementation* is the part that changes
across vendors — which is why `wave_reduce` is a primitive, not inline code:

- **L3 lane reduction** (within a wave): warp-shuffle tree on NVIDIA
  (`__shfl_xor`), DPP/permute-lane on AMD. *Different instructions, same math.*
- **L2 cross-wave reduction** (within a CTA): through scratchpad, or a
  cooperative warp-shuffle. *Mostly shared structure.*
- **L0 cross-CTA reduction** (split-K, MoE reduce): global atomics or a second
  kernel. *Shared structure.*

**Expression:** `wave_reduce(x, op=sum, axis=0)` in the default body is the
portable spelling; the primitive's body is provided per target (this is exactly
Axis H2 — the primitive-level override — at work for reductions). The author
*never writes `__shfl`*. For split-K / MoE reductions that span L0, the default
body uses a higher primitive (`reduce_across_ctas`) whose structure *is* shared.

### 2.5 Async copy + pipelining (L2 → a knob, not hand-rolled)

The multi-stage producer/consumer pipeline (compute tile `i` while loading tile
`i+1`) is where naïve kernels lose 2× to good ones. Today every kernel hand-rolls
the `cp.async` / TMA / global→LDS DMA calls, the barriers, and the stage count.
The model:

- `stage_async(load(...), into=scratch)` returns a **handle** (a future tile), not
  a synchronous value. The emitter lowers it to `cp.async.bulk` (TMA, sm_90),
  `cp.async` (sm_80), or `buffer_load … to LDS` (AMD), plus the right barrier.
- `num_stages` is a **declared knob** (`@targets(..., knobs={..., "num_stages":[3,4]})`),
  not a magic constant. The emitter rotates `num_stages` buffers and inserts the
  producer-consumer edges.
- The default body's `for k in range(...): stage_async(...); mma(...)` *is* the
  pipeline — the emitter infers the producer/consumer structure from the
  dataflow. **The author declares the depth; the emitter owns the plumbing.**

## 3. Worked example: the GEMM, annotated by level

Re-annotating `04` Ex.2's two bodies so the levels are visible:

```python
# ---- DEFAULT BODY: operates at L0/L2 only. No wave, no lane, no L5 shape. ----
@launch(grid=lambda a,c,*,BLOCK_M,BLOCK_N: (c.shape[0]//BLOCK_M, c.shape[1]//BLOCK_N))  # L0
def gemm(ctx, a, b, *, BLOCK_M, BLOCK_N, BLOCK_K):
    mi, ni = ctx.program_id(0), ctx.program_id(1)                      # L0
    acc = zeros((BLOCK_M, BLOCK_N), fp32)                              # L4 register; layout inferred
    for k in range(0, a.shape[1], BLOCK_K):                           # L2 streaming
        a_tile = stage_async(load(a[mi:mi+BLOCK_M, k:k+BLOCK_K]),      # L4 vectorize AUTO;
                             into=scratch)                            #   L2 scratch (smem/lds by target)
        b_tile = stage_async(load(b[k:k+BLOCK_K, ni:ni+BLOCK_N]), into=scratch)
        acc = mma(a_tile, b_tile, acc, accum=fp32)                    # L5 engine chosen by target
    store(acc.to(a.dtype), ctx.out.c[mi:mi+BLOCK_M, ni:ni+BLOCK_N])
# Nothing above names a wave (32/64), an instruction, or an L5 shape → portable.

# ---- sm_90 OVERRIDE: drops to L1/L2/L5 to reach the ceiling. ----
@gemm.target("cuda", arch="nvidia_sm90")
def gemm_sm90(ctx, a, b, *, BLOCK_M, BLOCK_N):
    mi, ni = ctx.program_id(0), ctx.program_id(1)
    a_desc = ctx.tma_descriptor(a, tile=(BLOCK_M, BLOCK_K))           # L5-grade bulk copy (descriptor space)
    acc = zeros((BLOCK_M, BLOCK_N), fp32)
    with ctx.cluster(2):                                              # L1 cluster (distributed smem)
        b_smem = stage_async(load(b[...]), into=scratch, policy="swizzled")  # L2 + L4 swizzle pinned
        for k in range(0, a.shape[1], BLOCK_K):
            acc = wgmma(a_desc[k], b_smem, acc,                        # L5 native warp-group MMA
                        mma_shape=(64, BLOCK_N//4, 16))                # L5 native shape (m=64, k=16)
    store(acc.to(a.dtype), ctx.out.c[...])
# This body names cluster (L1), swizzle (L4), and wgmma+shape (L5) — none of
# which exist on AMD. That's exactly why it's an override, not the shared body.
```

The contrast is the whole argument: **the default body is portable because it
operates above L3; the override reaches the ceiling because it operates at
L1/L4/L5.** Every portable line is one that names no vendor-specific level.

## 4. Who decides what (the auto-vs-explicit contract)

This is the table that makes the model *useful* — it tells the author when to
write something and when to trust the compiler:

| Decision | Default body | Override body |
|---|---|---|
| Output tile shape (`BLOCK_M/N`) | author (knob) | author (knob) |
| Streaming tile (`BLOCK_K`) | author (knob) | author (knob) |
| Pipeline depth (`num_stages`) | author (knob) | author (knob) |
| **Wave size (32/64)** | **compiler (never author)** | **bound by target, never literal** |
| **Vectorize width (L4)** | **compiler (auto)** | optional pin |
| **Scratch kind (smem/lds)** | **compiler (by target)** | bound by target |
| **MMA instruction (wmma/wgmma/mfma)** | **compiler (by target+dtype+shape)** | author names it |
| **L5 native shape** | compiler (must divide L2 tile; checked) | author pins it |
| Swizzle / bank-conflict layout | compiler | optional pin |
| TMA descriptor / cluster | *not expressible* (→ override) | author names it |

The bold rows are the portability surface: things the default body **never**
names, so it can't accidentally bake in 32 / smem / tensor-cores. That's the §10
anti-goals enforced *structurally* — there's literally no syntax in the default
body to hardcode them.

## 5. How this unifies with the H1/H2 question (`06` A1)

The override-granularity question was: can you reach a ceiling by swapping a
*primitive's body* (H2), or do you need a *whole new kernel body* (H1)? In
programming-model terms:

- **H2 = drop one level**: keep the L0/L2 structure, swap the L5/L3 primitive
  bodies (`mma` → wgmma, `wave_reduce` → shuffles). The default body *structure*
  survives; only the leaves change. Elegant; works when the ceiling is "use the
  matrix engine / use the right shuffle," which is most GEMM/norm cases.
- **H1 = restructure levels**: the ceiling needs L1 (clusters) or a different L2
  staging (TMA descriptors) that the default body can't express. New body, same
  contract. Needed for sm_90-top kernels, some attention.

**The programming model predicts the answer to H1/H2:** it's "which levels does
this target's ceiling live at?" If the delta is only at L3/L5 leaves → H2. If it
reaches up to L1 (clusters) or changes L2 staging → H1. The Phase-1 H1/H2
measurement (`06` A1) is therefore *not* ad-hoc — it's "count, per op, the
highest level the override had to drop to." That's a clean, compoundable metric.

**And this is where the IR ([`09`](09-agent-editable-ir.md)) makes it agent-driven:**
the H2 cases are exactly the ones reachable by *local schedule edits*
(`map_to`, `retile`, `add_stage`), so the agent pushes them autonomously; H1 is
the `promote_override` escape hatch where the agent freehand-edits a body and
`verify` gates. So H1/H2 is not just a measurement — it's the *division of labor
between autonomous tuning and human/strong-model authoring.*

## 6. Inspiration and non-rivalry

The model is a deliberate distillation, not an invention:
- **Triton** is the model for the default (block-level, compiler-maps), and the
  Triton lowering target (`03` Axis C).
- **CUTE / CUTLASS** is the model for the override (the hierarchy as named
  objects; layout algebra; copy atoms; TMA descriptors) — the DSL's `ctx.tma_descriptor`,
  `policy="swizzled"`, `wgmma(..., mma_shape=...)` borrow directly.
- **Composable Kernel** is the AMD-side analogue (LDS staging, MFMA shapes,
  `waves_per_eu`) for the HIP override.

The DSL is *not* trying to beat any of them at their own game. It's providing the
**spine** (the named hierarchy + the block-default/explicit-override rule) that
lets one source *drive* all three, with the contract (`02` Layer 1) guaranteeing
they compute the same thing. `02` §"non-rivalry"; this doc is the mechanism half
of that claim.

## 7. What's auto, what's checked, what's rejected

- **Auto** (compiler owns it in the default): wave-size, vectorize width,
  scratch kind, MMA-instruction selection, swizzle.
- **Checked** (compile-time error, never a `verify`-time surprise): an L2 tile
  not divisible by the target's L5 native shape; a reduction axis that crosses a
  wave boundary without a `wave_reduce`; a `reduce_dtype` mismatch with the
  accumulator (Axis F, `03`).
- **Rejected** (a literal that bakes in a vendor): `warp=32` in a default body;
  `smem`/`tensor_cores` in a multi-target body; an L1-`cluster` primitive outside
  an sm_90 override. These map 1:1 to the §10 anti-goals and the publish
  checklist in `06` §B.

This three-way split — *auto / checked / rejected* — is how a programming model
turns the §10 "don'ts" into compiler behavior instead of prose conventions. That
is the real reason to have a DSL rather than keep authoring in four paradigms:
**the constraints become enforceable, and the portability vocabulary becomes
syntax, not folklore.**

> **The agent-editable IR ([`09`](09-agent-editable-ir.md)) makes "checked" an
> *edit-time* gate, not just a compile-time one.** When the agent is pushing
> performance by editing the schedule IR, every edit (`retile`, `map_to`, …) is
> validated against these same rules *before* it lowers — so a tile that
> violates L5 divisibility is rejected at proposal time with a reason, not
> discovered at compile time. The auto/checked/rejected contract is therefore
> the same contract at three layers: authoring (`08`), editing (`09`), and
> publishing (`06`). One set of rules, enforced everywhere.
