# 01 — Why a DSL, and why now

## 1. What authoring a kernel actually costs today

Adding one op to `xkernels` (per
[`meta/docs/adding-a-kernel.md`](../../meta/docs/adding-a-kernel.md)) is a
multi-file, multi-paradigm undertaking. Take `dual_rmsnorm`, the simplest kernel
in the repo. To get it shipped today a human (or agent) writes/coordinates:

| Artifact | File | Paradigm | Live in repo? |
|---|---|---|---|
| Op Spec (contract) | `registry/ops/dual_rmsnorm.spec.json` | declarative JSON | ✅ |
| Backend-neutral reference | `src/xkernels/ops/norm/reference.py` | pure torch | ✅ |
| Shape sweep | `registry/shape_sweeps/dual_rmsnorm.sweep.json` | declarative JSON | ✅ |
| Seeded input generator | `src/xkernels/registry/input_gen.py` | Python, case-by-case | ✅ |
| Triton impl | `src/xkernels/ops/norm/triton/dual_rmsnorm_kernel.py` | `@triton.jit` + `tl.*` | ✅ |
| CUTE impl (NVIDIA) | `src/xkernels/ops/norm/cute/rmsnorm_kernel.py` | `cutlass.cute` DSL | ✅ |
| CUDA/HIP impl (NVIDIA/AMD) | `src/xkernels/ops/.../cuda/*.cu` | C++ templates + `<<<>>>` | ✅ |
| Impl Card per backend | `registry/impls/dual_rmsnorm.<backend>.card.json` | declarative JSON | ✅ |

That is **eight artifacts in four paradigms for one operation**, and the four
paradigms do not even agree on vocabulary:

- The **constraint** `"dtype(x1) == dtype(w1)"` is written as a string in the
  spec, re-checked by hand inside the kernel, and re-asserted nowhere else.
- The **wave size** (32 on NVIDIA, 64 on AMD) is a `tl.constexpr`-adjacent
  implicit in Triton, a literal `<<<blocks,threads>>>` in CUDA, and a property
  of a CUTE tiled layout — three different places, none of them the spec.
- The **numerics** (`reduce_dtype: fp32`, `cross_backend_rtol: 1.6e-2`) live in
  JSON; the actual `.to(tl.float32)` accumulation lives in the Triton source.
  Nothing checks that they agree.

This is not a complaint about the substrate — the substrate is correct to keep
the contract in machine-readable JSON. It is an observation that **the contract
and the source are authored by hand, separately, with no mechanical link.**
Every drift bug (a spec that says `fp32` accumulate while the kernel accumulates
in `bf16`; a card that claims `wave_size: 64` for a kernel tiled at 32) is a bug
this gap creates. The `verify` harness *catches* such bugs eventually — but
"eventually, at runtime, on a GPU" is exactly the expensive end of the feedback
path the design is trying to make cheap (§6.1).

### 1a. Two more pains the multi-target + graph stance specifically addresses

The hand-authoring cost above is the *correctness* pain. Two *performance* pains
follow from the same four-paradigm split:

- **Native ceilings are unreachable from one source.** To get an sm_90 GEMM that
  uses TMA + clusters + wgmma, or a CDNA3 GEMM that uses MFMA + global→LDS DMA,
  you hand-write a *different* kernel per target today (`tune-for-cdna`,
  `port-across-arch` skills). There is no source that lowers to *both* ceilings;
  the closest is Triton, which is portable-but-capped. So the library either
  ships a portable card that misses both ceilings, or N hand-written cards that
  share nothing. A DSL whose per-target override reaches each ceiling from one
  source (Layer 3, `02`) is the structural fix.

- **Launch overhead is paid per kernel, with no second lever.** The library has
  the *fusion* lever (`add-epilogue-fusion`, `fuse-elementwise-chain`) for chains
  that can collapse to one kernel. It has **no graph lever** for chains that
  *can't* fuse (different grids, a reduction boundary) but where each kernel is
  small enough that launch overhead (~5–10 µs/launch) dominates. CUDA/HIP graphs
  collapse N launches to one; the DSL's `@graph` (Layer 4, `07`) is how the
  library gets that lever without authors hand-writing `cudaGraphAddKernelNode`.

So the v0.2 stance (multi-target + graphs + perf, day 1) is not scope creep —
it's the design directly attacking the three pains (drift, unreachable ceilings,
launch overhead) that the four-paradigm split creates.

## 2. The portability vocabulary is re-encoded per kernel

`library.md` §4.1 names the five concepts that differ across vendors and "are
exactly the things that bite if left implicit":

| Concept | NVIDIA | AMD |
|---|---|---|
| Execution group | warp = **32** lanes | wavefront = **64** lanes |
| On-chip scratch | shared memory (smem) | LDS |
| Matrix engine | tensor cores (wmma/wgmma) | matrix cores (MFMA) |
| Bulk async copy | `cp.async` / TMA | global→LDS DMA |
| Tuning unit | `num_warps`, stages | `waves_per_eu`, stages |

Today, every kernel re-derives these. `ffn.cu` writes `<<<blocks, threads>>>`
with `threads=256` — a hidden `threads % 32 == 0` (NVIDIA) assumption that is
simply *wrong* on a 64-lane wavefront and has to be caught by a human. The
`dual_rmsnorm` card honestly admits `"wave_size: 0"` because the row-reduce is
wave-agnostic — but nothing in the source *says* that; a reader has to infer it.
The skill `tune-for-cdna` exists precisely because "hipified code left at
warp=32 tiling silently halves occupancy" is a recurring failure mode.

The contract already has the right vocabulary (`arch.wave_size`,
`arch.requires: [mfma]`, `arch.scratch.kind: lds`). The source does not. **A DSL
whose compute primitives take these as named parameters — not literals — closes
that gap by construction.**

## 3. The authoring surface is hostile to the agent loop

The library is explicitly *agent-native* (§1, §6): the consumer is an LLM agent.
But look at what an agent must do to author one new backend of one op today:

1. Pick a paradigm (Triton? CUDA? CUTE?) — and each has its own footguns the
   agent must not step on (Triton's `tl.constexpr` rules; CUDA's
   `AT_DISPATCH_FLOATING_TYPES_AND2`; CUTE's `from_dlpack` / dialect imports).
2. Hand-write the Op Spec JSON in a separate file, matching the kernel's actual
   dtypes/constraints by memory.
3. Hand-write a pure-torch reference that must agree with the device kernel to
   `rtol`/`atol` — and re-derive it if either changes.
4. Register the backend with `register(kernel, Backend.X)` and keep the dispatch
   key in sync with `op_spec.kernel`.

Every one of these is a place an agent can be subtly wrong in a way that only
`verify` on a GPU catches. The skills (`tile-a-gemm`, `port-cuda-to-hip`,
`tune-for-cdna`) paper over this by encoding the *procedure* — but they cannot
remove the degrees of freedom, because the freedom lives in the source, not the
skill. **A narrower authoring surface — fewer places to be wrong, an Op Spec that
is generated rather than typed — is the single highest-leverage way to make the
agent loop in §6 cheaper and more reliable.** That is the "why now."

## 4. This answers an open question, it doesn't reopen a closed one

`library.md` §11 lists, as open:

> *Portability production path: native per-backend cards only, or also a
> portable DSL (Triton, or CUTLASS + AMD Composable Kernel) as a fast way to
> seed both backends from one source? Current lean: allow DSL builds as one
> implementation among others, gated by the same verification, never as the
> sole backend.*

This brainstorm takes that "lean" and asks the natural next question: **what if
the DSL is not just one more backend, but the *authoring layer that produces
several*?** It does not contradict §10's "no single lowest-common-denominator
source for all backends," because (a) the DSL can emit *native per-backend*
cards, not a single shared source, and (b) hand-written native cards remain
first-class — the DSL is a producer, never a gatekeeper. Exactly the §8.4
"graceful degradation" stance: agents/humans who want to write raw Triton or C++
still can, and `verify` still governs everything.

## 5. Success criteria (what "the DSL worked" would mean)

1. **One source per op** produces a *valid, non-drifting* Op Spec + reference +
   sweep + ≥1 Impl Card, such that `verify` passes first try on a kernel the
   author wrote in the DSL (no hand-editing the JSON).
2. **Cross-backend parity is structurally encouraged, not heroic** — authoring
   one op for two backends from one DSL source is less work and *less
   bug-prone* than authoring two separate kernels.
3. **The portability vocabulary is explicit** — no kernel in the DSL hardcodes
   `32`, `smem`, or `tensor_cores`; these are named parameters or rejected.
4. **The agent loop gets cheaper** — measured (per §9's acceptance metrics) as
   fewer median iterations to a `verify`-passing, parity-passing card than the
   current hand-author path.
5. **It never becomes a gatekeeper** — a kernel written in raw Triton/CUDA/CUTE
   remains a first-class Impl Card, and the DSL never lowers the correctness bar.
6. **Performance portability is paid for, not assumed** — the v0.2 stance (perf
   day 1) means each target gets an explicit override body reaching its ceiling;
   a correct-but-slow DSL card is a failure, not a starting point. "Performance
   portability is not free" (§10) is honored by the override, not repealed.
7. **Graphs are an orchestration lever, not a kernel feature** — the DSL lowers
   *compositions* to CUDA/HIP graphs; it does not require every op to be captured
   (a single big kernel gains nothing from a graph). Graphs win on launch
   overhead for chained small kernels; the design says where, honestly (`07` §8).

## 6. Explicit non-goals (what this is NOT)

- **Not a replacement for the contract.** Op Specs / Cards / JSON Schema stay
  the source of truth. The DSL *produces* them; it does not *become* them.
- **Not "fast everywhere from one source" by magic.** §10 explicitly forbids
  the lowest-common-denominator trap. A DSL card that is slow on both backends
  is a failure even if it is correct on both.
- **Not a competitor to CUTE / CUTLASS / Composable Kernel.** Those are
  *lowering targets* the DSL can drive, not rivals (see `02`, `06`).
- **Not a bespoke *skill* DSL** (the §10 anti-goal is about skills, which stay
  SKILL.md). This is about *kernel source*, a different layer.
- **Not mandatory.** Per §8.4 the bottom tier ("just read the JSON") must keep
  working. The DSL is additive.
