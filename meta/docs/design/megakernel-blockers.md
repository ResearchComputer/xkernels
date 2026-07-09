# Megakernel for Apertus-8B — what's blocking it

**Status:** design note (open). Step 1 of the path is **done** (`fused_xielu_ffn@1.0.0`,
CPU-verified). Blocker (b)'s **substrate is now in place** — the `persistent`
schema sub-fields are pinned, the residual-add primitive exists, and the
`author-a-persistent-kernel` skill is authored (GPU-gated). What REMAINS of (b)
is the open `canonical_op` question + the first real persistent card; (c)
(hand-written-native attention) is unchanged. Steps below are recorded so a
future GPU-equipped task can pick them up without re-deriving the analysis.

**Target model:** `swiss-ai/Apertus-8B-Instruct-2509` — a Llama-3-style decoder,
bf16, 32 layers, hidden 4096, intermediate 21504, GQA (32 q-heads / 8 kv-heads),
`qk_norm: true`, separate `attention_layernorm` + `feedforward_layernorm`, RoPE
(llama3, θ=12M). Two non-Llama details drive everything below:

- **Non-gated FFN** — only `up_proj` + `down_proj`, **no `gate_proj`**. The block
  is `down_proj( xIELU( up_proj( rmsnorm(x) ) ) )`, *not* SwiGLU.
- **xIELU activation** (arXiv:2411.13010) — a *learnable* piecewise nonlinearity
  (`alpha_p`, `alpha_n` log-space + fixed `beta`, `eps`) that branches on `sign(x)`.

A **megakernel** here means one launch executing a whole transformer sub-block
(or layer) with intermediates kept on-chip (registers/LDS) instead of
round-tripping to DRAM. This is the most aggressive form of fusion, and it sits
in tension with `meta/docs/library.md` §10 ("One mega-kernel with a thousand
flags. Validity surface explodes; the agent can't reason about
applicability.") — so the path below is **narrow, specific fusions**, not a
universal layer JIT.

---

## What's already done (step 1)

`fused_xielu_ffn@1.0.0` — the non-gated FFN as a single verified op:

- **Op Spec** `registry/ops/fused_xielu_ffn.spec.json` (`canonical_op:
  activation`, `fusions: [xielu]`, `composes_with: [rmsnorm@1.0.0]`).
- **Reference** `src/xkernels/ops/ffn/reference.py:xielu_ffn_reference` —
  reuses the standalone `xielu@1.0.0` reference so the activation precision path
  is shared exactly (bit-identical activation; only the projections differ).
- **Triton card** `src/xkernels/ops/ffn/triton/xielu_ffn_kernel.py` — projections
  are `torch.matmul`, xIELU reuses the hand-written `xielu_triton` kernel.
- **Shape sweep** `registry/shape_sweeps/xielu_ffn.sweep.json` — 6 points
  covering near-init / moderate / the **Apertus-8B overflow checkpoint
  (alpha_p=166.0)** / the softplus boundary / real dims (M=2, K=4096, N=21504).
- **CPU gate passes**: `verify('fused_xielu_ffn.reference@1.0.0', arch='any')`
  → `compiled=True, correctness.passed=True, max_abs_err=0.0, n_points=6,
  determinism=True`. The triton card honestly reports `compiled=False` on a
  CPU-only box (the GPU gate fires later).

This collapses the FFN from **three launches** (up_proj GEMM → xielu → down_proj
GEMM) into one op, killing the `[M, N=21504]` intermediate's DRAM round-trip and
two launches. It is the immediate, tractable win and the precondition for
anything further.

---

## Blocker (b) — the persistent-grid substrate ~~is reserved but empty~~ (substrate landed 2026-07-08)

A true whole-layer megakernel needs to stage *multiple heterogeneous GEMMs plus
attention plus norms* on-chip across one persistent grid. The schema reserves a
place for this; the substrate to declare the contract honestly is now in place
(points 1, 3, 5 below landed; point 2 is the remaining open question).

1. **~~The `persistent` schema block is WIP / empty.~~ DONE — sub-fields pinned
   (2026-07-08).** `registry/schema/impl_card.schema.json` now pins six typed
   sub-fields under `persistent` (with `additionalProperties: true` kept open so
   new orchestration modes stream-add): `residency` (intermediates kept
   on-chip — the defining megakernel property), `warp_roles`, `register_budget`
   (regs/thread + spills), `lds_budget` (bytes + pipeline stages),
   `pipeline_stages` (vkl schedule IR `Stage(id, producer_ref, space, depth)`),
   `defused_edges` (edges that DO hit DRAM — the complement of residency).
   Vocabulary mirrors `vkl/schedule.py` + `vkl/cost.py`. Anti-regression test:
   `test_impl_card_schema_pins_persistent_block`. So a megakernel card can now
   declare its on-chip dataflow contract honestly and applicability can be
   reasoned about from metadata (§1.3.2).

2. **No Op Spec category for "transformer layer / persistent dataflow" — OPEN.**
   The `canonical_op` enum has no entry for a fused multi-op block; the closest
   existing pattern is `mhc_pre@1.0.0` + `hc_prenorm_gemm@1.0.0` (rmsnorm
   prologue + GEMM + sigmoid heads + sinkhorn + residual combine), whose
   `composes_with` links model *adjacent ops within one sub-block*. This is the
   one part of (b) deliberately LEFT OPEN: adding a `canonical_op: "persistent"`
   is a premature commitment to retrieval semantics before any real persistent
   card exists to define them (library.md §11 "Canonical op vocabulary: how
   granular?" is itself an open question). The interim, documented in the
   `author-a-persistent-kernel` skill, is to use the **dominant leaf
   canonical_op** of the block (attention sub-block → `attention`; FFN
   sub-block → `activation`/`gemm`) with rich `fusions` tags + `composes_with`.
   Commit a dedicated `canonical_op` when the first real persistent card lands
   and the retrieval semantics are clear.

3. **~~No megakernel authoring skill.~~ DONE — `author-a-persistent-kernel`
   skill authored (2026-07-08).** `.agents/skills/author-a-persistent-kernel/`
   is a GPU-gated peer of `author-an-op-spec` + `add-epilogue-fusion`. It is the
   LAST-RESORT skill: it fires ONLY after graph capture (`launch.graph`) is in
   place, narrow fusion (`add-epilogue-fusion` / `fuse-elementwise-chain`) is
   exhausted, AND a profile (rocprof/ncu) shows the on-chip-staged regime
   winning (decode-only). It covers the `persistent` block's on-chip contract,
   the register/LDS budget math (heterogeneous-GEMM trap, bound to ONE
   sub-block), the shape-regime gate (decode vs prefill), and the honest
   stance that a whole-layer megakernel is the §10 anti-goal in disguise.

4. **The register/LDS budget problem across heterogeneous GEMMs.** An Apertus
   layer has **5 GEMMs in 4 shapes** — q/k/v `[M,4096]×[4096,{4096,512,4096}]`,
   o_proj `[M,4096]×[4096,4096]`, up_proj `[M,4096]×[4096,21504]`, down_proj
   `[M,21504]×[21504,4096]` — plus attention, 3 norms, xielu, 2 residual adds.
   Keeping *both* the 4096-wide activations and the 21504-wide FFN hidden
   resident across the whole layer is a register/LDS budget problem no
   current single-op card addresses, and it is **shape-regime dependent**:
   - **Decode (M=1):** launch/latency-bound → a persistent megakernel pays off
     most (kills ~12 launches/layer).
   - **Prefill (M large):** the GEMMs are compute-bound and a monolithic
     megakernel usually *loses* to well-tiled separate GEMMs at the roofline.
   The `persistent` block has no way to express shape-gated applicability, so a
   megakernel card would need that added (or be decode-only by constraint).

5. **~~No residual-add primitive.~~ DONE — `residual_add@1.0.0`
   (2026-07-08).** The bare residual add now exists as a first-class contract:
   `registry/ops/residual_add.spec.json` (`canonical_op: elementwise`,
   `out = (x.float() + residual.float()).to(dtype)` — the
   `add_rmsnorm_ref` residual convention), reference in
   `src/xkernels/ops/comm/fused.py:residual_add_ref`, reference-only card
   (deliberately no separate Triton kernel — `torch.add` wins; the op's value is
   the contract a megakernel's `persistent.pipeline_stages` references).
   `composes_with: [rmsnorm@1.0.0, fused_xielu_ffn@1.0.0]`. CPU gate passes
   (`verify('residual_add.reference@1.0.0', arch='any')` → passed, max_abs=0.0).

### What (b) needs, concretely — status
- ✅ ~~Pin the `persistent` schema sub-fields~~ — DONE (six typed sub-fields
  pinned, `additionalProperties: true` kept open).
- ⏳ Add a `canonical_op` (or a dedicated spec shape) for a persistent multi-op
  block, with shape-gated applicability constraints — **OPEN** (deliberately
  deferred; interim = dominant leaf canonical_op + rich fusions tags, per the
  `author-a-persistent-kernel` skill; commit when the first real persistent
  card lands).
- ✅ ~~Author an `author-a-persistent-kernel` skill~~ — DONE (GPU-gated,
  last-resort).
- ✅ ~~Stand up a residual-add primitive~~ — DONE (`residual_add@1.0.0`,
  reference-only, CPU-verified).

So blocker (b)'s **substrate** — the pinned schema, the primitive, the skill —
is in place. What remains for a *card* is: the open `canonical_op` decision
(point 2) + the first real persistent card on a GPU (which also depends on
blocker (c), the hand-written-native attention path).

---

## Blocker (c) — attention forces the hand-written-native path

A whole-layer megakernel includes attention, and **attention cannot be produced
by the vkl DSL fast path** (`author-a-kernel-with-dsl`). The DSL's math IR
expresses a fixed DAG of pointwise / reduce / MMA — covering `gemm`, `norm`,
`reduce`, `activation` cleanly. It explicitly does **not** cover (see
`meta/docs/design/vkl.md` and the `author-a-kernel-with-dsl` SKILL.md routing
table):

- **online softmax** (the attention core — a running max + LSE, not a fixed
  reduction);
- **causal / data-dependent masking** (control flow the IR can't carry);
- **paged-KV gather** (`out[idx] = ...`, scatter/gather/indexing — the
  `paged_kv_gather@1.0.0` op exists precisely because the DSL couldn't emit it);
- **collectives** (all-reduce, all-to-all).

So a whole-layer megakernel **cannot be emitted from one `@kernel` source** the
way `fused_xielu_ffn` (an activation-category op) could lean on the existing
hand-written xielu kernel. It would have to be **hand-authored native CUDA/HIP**
— exactly the thing the library's whole design (`meta/docs/library.md` §1.2
"Compose over generate"; the vkl DSL; the agent-native retrieval/verify loop)
is built to make unnecessary.

This is the single biggest reason a full megakernel fights the grain of the
project: the leaf attention cards (`paged_attention@1.0.0`,
`paged_attention_prefill@1.0.0`, `sparse_mla_attention@1.0.0`) are already
hand-written Triton (the `triton` backend is the portable layer, not native
CUDA/HIP). Fusing them into a megakernel means either (i) a hand-written native
persistent kernel that subsumes the attention math — a large, non-portable,
hard-to-verify artifact; or (ii) a graph capture of the existing Triton cards
(see the pragmatic path below), which is lighter-weight but is *not* on-chip
fusion.

### What (c) means for the path
- A full megakernel is **not** reachable from the DSL; it is a native-authoring
  project gated on a GPU and on blocker (b)'s substrate.
- The honest interim is **CUDA/HIP graph capture** (the `launch.graph` schema
  block already supports it) to kill launch overhead without the
  validity-surface explosion — then pick off DRAM round-trips incrementally
  with `add-epilogue-fusion` / `fuse-elementwise-chain`, each gated by `verify`
  + `verify_parity`.
- Only if a profile (via `use-rocprof-compute` / `use-nsight-compute`) shows the
  on-chip-staged-whole-layer regime clearly winning on a target arch — almost
  certainly **decode-only** — does the native megakernel become worth the
  blocker-(b) substrate work.

---

## The contract-faithful path (summary)

1. ✅ **`fused_xielu_ffn@1.0.0`** — done (CPU-verified; GPU gate pending).
2. ⏳ **Fuse the attention block narrowly** (dual_rmsnorm → rope → paged_attention
   → o_proj) via `add-epilogue-fusion`, reusing the existing `composes_with`
   graph. CPU-doable contract; GPU-gated card.
3. ⏳ **Graph-capture** the per-op cards (`launch.graph`) to kill launch overhead
   without the mega-kernel validity-surface explosion.
4. ⏳ **Profile** (rocprof/ncu) to find the DRAM round-trips worth fusing; pick
   them off with `fuse-elementwise-chain` / `add-epilogue-fusion`, each
   `verify` + `verify_parity` gated.
5. ⏳ **Only then** consider the native persistent megakernel — and only if the
   profile shows the on-chip-staged regime winning (decode-only). This unblocks
   on (b) the persistent-grid substrate + authoring skill, and (c) the
   hand-written-native attention path.

The design contract's stance (§10) is deliberate: a monolithic kernel with a
thousand flags is an anti-goal. The path above respects it by fusing
**specific** chains and letting the agent reason about each card's
applicability — while leaving the door open to a *controlled* persistent-grid
megakernel through the reserved `persistent` schema block, once the substrate
to declare its contract honestly exists.
