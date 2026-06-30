---
name: fuse-elementwise-chain
description: >
  Collapse a CHAIN of independent pointwise elementwise ops (cast, scale, bias,
  activation, mul/add) running back-to-back between two heavier kernels into ONE
  kernel, killing the intermediate-DRAM round-trips and the per-op launch
  overhead. Pure pointwise fusion is numerically exact (tolerance unchanged); the
  only questions are (a) is the chain standalone (its own op) or embedded in a
  host kernel (a card variant / add-epilogue-fusion), and (b) is the bandwidth
  win worth a new kernel. Kernel-layer skill: the fused card must compile and
  pass verify on a GPU; the numerics are trivially CPU-checkable (pointwise ==
  exact). Use when a trace shows 2+ tiny pointwise kernels in a row, each reading
  the previous's full output from DRAM.
license: Apache-2.0
x-kernel-lib:
  id: fuse-elementwise-chain@1.0.0
  backend_scope: agnostic
  when_to_use:
    triggers:
      - "a trace shows >=2 pointwise (cast/scale/bias/activation/mul) kernels chained, each reading the prior's full tensor from DRAM"
      - "an op is literally 'silu(g)*u' / 'a*scale+bias' / 'cast then mul' as a standalone kernel"
      - "a benchmark calls for fusing a pointwise chain to cut launch + bandwidth overhead"
    preconditions:
      - "every op in the chain is POINTWISE (no reduction, no reshape that changes the element mapping, no gather/scatter) — a reduction in the chain routes to add-epilogue-fusion instead"
      - "the chain's inputs/outputs are known (so the fused op's contract can be written)"
  inputs_required:
    - "the ordered list of pointwise ops + their parameters"
    - "whether the chain is standalone (own op) or embedded in a host kernel"
    - "target arch + dtype"
  tools:
    - get_op_spec
    - verify
    - verify_parity
    - record_measurement
  validation:
    must_pass:
      - "fused card verifies against the pointwise-chain reference at the UNCHANGED tolerance (pointwise is exact: max_abs ~ 0, subject only to fp cast order)"
      - "if standalone: a NEW Op Spec exists (author-an-op-spec); if embedded: a card variant under the host Op Spec (add-epilogue-fusion case a)"
      - "verify_parity still agrees (a fp-cast-order difference across backends is the only possible numeric divergence — check it)"
      - "fused perf.ms < sum of chain perf.ms on the target arch (the win is launch + N-1 DRAM round-trips; if it doesn't win, the chain was already bandwidth-saturated)"
  references:
    - "src/xkernels/ops/ffn/triton/ffn_kernel.py (fused SwiGLU silu(g)*u — a 2-op pointwise chain as one kernel)"
    - "src/xkernels/ops/mhc/pre_post_reference.py (sigmoid pre/post heads — pointwise chains)"
    - "meta/docs/library.md §10 (mega-kernel anti-goal: fuse a SPECIFIC chain, don't build a universal pointwise JIT)"
  metrics:
    uses: 0
    success_rate: null
    median_iterations: null
    regression_count: 0
  provenance:
    authored_by: human
    created: "2026-06-25T00:00:00Z"
    supersedes: []
---

## Procedure

1. **Confirm every op is pointwise.** Walk the chain; each op must map each output
   element from the same-index input element(s) with no cross-element dependence.
   Cast, scale, bias, unary activation (silu/gelu/sigmoid/relu), binary mul/add:
   yes. Any reduction (sum/mean/max), layout-changing reshape, or gather/scatter:
   NO — that op routes the whole task to `add-epilogue-fusion` (a reduction
   epilogue) or `author-an-op-spec` (a genuinely different op), not here.

2. **Decide standalone vs embedded.** If the chain sits BETWEEN two heavy kernels
   as its own launch reading/writing full tensors, it is a standalone op → new Op
   Spec (author-an-op-spec). If the chain is the tail of a host kernel's output
   tile (e.g. SwiGLU right after the FFN gate/up projections), it is an epilogue
   → `add-epilogue-fusion` case (a). This choice is made up front; it sets whether
   you write a new spec or a card variant.

3. **Write the reference as the literal pointwise composition.** In the
   backend-neutral reference this is just the ops applied in order in pure torch
   (e.g. `out = (g * torch.sigmoid(g.float()).to(g.dtype)) * u` for SwiGLU). Keep
   fp32 for any transcendentals (sigmoid/exp) even if storage is bf16 — the cast
   to storage dtype happens last. This fp32-internal / storage-cast-last order IS
   the numerics contract; backends must match it.

4. **Author the fused kernel.** One program per output tile; load each chain
   input once, apply the ops in registers, store once. The win is eliminating
   `(N-1)` full-tensor DRAM writes + reads and `(N-1)` launches, so the chain
   must actually have `N >= 2` pointwise ops to be worth fusing (a single op is
   already one kernel).

5. **Author the card** (new spec for standalone; variant under host spec for
   embedded). `specialization_knobs` is usually empty (pointwise has no real
   tuning space beyond a tile size, which is internal and not declared). Set
   `perf.roofline: memory_bound` — a pointwise chain is pure element traffic.

6. **Verify + parity.** `verify` against the reference: expect `max_abs ~ 0`
   (pointwise is exact modulo the fp cast order). `verify_parity`: the only
   cross-backend divergence possible is cast order (e.g. one backend keeps fp32
   through the sigmoid, another casts early); confirm it's within the (looser)
   cross_backend_rtol. On GPU, confirm `fused ms < sum-of-chain ms` and
   `record_measurement`.

7. **Honest no-GPU branch.** Numerics (steps 1, 3, 6-reference) are fully
   CPU-doable and trivial for pointwise. The fused kernel compile + the
   bandwidth-win measurement are GPU-gated. Ship `compiled:false` honestly.

## Pitfalls

- **Fusing a chain that secretly contains a reduction/reshape.** silu(g)*u is
  pointwise; `sum(g)*u` or `reshape(g)*u` is not. A non-pointwise op in the chain
  makes the "pointwise == exact, tolerance unchanged" reasoning false. Re-route.
- **Different fp cast order across backends.** Pointwise is exact only if both
  backends keep transcendentals in fp32 and cast to storage last. If one casts to
  bf16 before the sigmoid, parity drifts — and because the divergence is small
  and the op is "just pointwise," it's easy to miss. verify_parity catches it.
- **Building a universal pointwise JIT instead of a specific fusion.** "A kernel
  that takes a list of ops and fuses them" is the §10 mega-kernel anti-goal
  (validity surface explodes, the agent can't reason about applicability). Fuse a
  named chain into a named card; emit one card per real chain.
- **Fusing a single op.** A one-op "chain" is just the existing kernel. There's
  no intermediate DRAM traffic to kill; fusion adds overhead. Require N >= 2.
- **Calling an embedded chain a standalone op (or vice versa).** An embedded
  chain (SwiGLU inside FFN) is a card variant of the host op, not its own op;
  calling it standalone fragments the op family and breaks retrieval. Step 2.
