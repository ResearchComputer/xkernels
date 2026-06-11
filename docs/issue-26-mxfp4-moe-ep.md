# Issue #26 — Expert parallelism (`ep_size > 1`) for the quantized fused-MoE

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch 2.11.0+rocm7.2,
`tokenspeed_triton`. **Validation:** `slurm/probe_moe_ep_beverin.sbatch`, job 381967.

## Context

DeepSeek-V4 (`deepseek_v4`) has 256–384 routed experts (top-6) and its mxfp4
weights do not fit one 128 GB MI300A replicated, so the experts must be sharded
across GPUs/nodes (**expert parallelism**). In tokenspeed the AMD quantized
fused-MoE (`mxfp4/{gluon,triton}_kernel.py`) `supports()` ends with
`return spec.ep_size <= 1 and ...`, so there is **no quantized MoE path under EP**
and V4 cannot be served on MI300A.

This repo's `xkernels` INT4-W4A16 grouped fused-MoE GEMM (issue #1) is the
portable mirror of that production kernel. The issue's root cause is host-side: the
grouped GEMM tiles output **by expert** and already filters per-block via an
`expert_ids` sentinel, so the kernel itself does not assume all experts are local.
What was missing is the **host-side wiring** to build the per-rank dispatch from a
local expert subset and to expose it on the public op.

## What ships

- **`moe_align_block_size_ep(topk_ids, block_size, num_experts, expert_map)`**
  (`ops/moe/w4a16.py`) — EP dispatch builder. The router still emits **global**
  `topk_ids`; `expert_map[g]` gives global expert `g`'s **local** weight-row in
  `[0, E_local)` (or `-1` if not on this rank). It remaps each routed slot to its
  local row, sends non-local slots to a sentinel id that the per-expert block
  builder never iterates (so they are dropped from this rank's compute), then
  reuses the existing, well-tested `moe_align_block_size_ref` over the `E_local`
  local experts. `expert_ids` then indexes the rank-local `[E_local, N, K//8]`
  weight tensor directly.

- **`fused_moe_int4_w4a16(..., expert_map=None)`** — new opt-in kwarg on the public
  op (and the reference + Triton backends). When given, `packed`/`scale` are the
  rank-local slice and the op returns this rank's **partial** `[M, N]` output
  containing only locally-held experts' contributions. Summing the partials across
  the EP group (the production all-reduce) reconstructs the full dense result.
  `expert_map=None` (default) is byte-for-byte the prior single-rank behavior.

No kernel-body change was required: the grouped GEMM already consumes an
`expert_ids` array and gathers tokens by `sorted_token_ids // top_k`; pre-zeroed
output buffers mean non-local token-slots (never written / never atomic-added)
stay zero, so the per-rank reduce is exactly the partial.

## Correctness (acceptance)

The invariant tested is the one the all-reduce relies on: for a contiguous-block EP
partition of the experts across `ep_size` ranks,

```
sum_over_ranks  fused_moe_int4_w4a16(A, local_packed[r], local_scale[r],
                                     topk_ids, topk_w, expert_map=emap[r])
  ==  full non-EP MoE output      (within bf16 atol/rtol = 2e-2)
```

| Check | Where | Result |
|-------|-------|--------|
| EP partition covers every routed slot exactly once, remapped to the right local row | CPU `TRITON_INTERPRET=1` + GPU | **PASS** |
| `sum(partials) == dense`, `mul_routed ∈ {False, True}`, ep ∈ {2,3,4} | CPU `TRITON_INTERPRET=1` (fp32, 3e-3) | **PASS** |
| `sum(partials) == dense`, decode (E=48,top_k=8,ep=4) + prefill shapes | **beverin MI300A, bf16, 2e-2** | **PASS** (job 381967) |
| `expert_map = identity` == non-EP path | CPU + **beverin** | **PASS** |
| Rank owns no experts → all-zero partial | CPU | **PASS** |
| INT4 MoE suite (issue #1/#20) regression | **beverin MI300A** | **PASS** (12/12) |

All EP shapes pass within tolerance on real gfx942 — acceptance met.

## Notes / scope

- This is the **single-GPU kernel + host dispatch** piece. The cross-rank
  all-reduce of the partials is a separate collective (see `xkernels.ops.comm`,
  issue #12) the serving layer already owns; lifting the tokenspeed
  `ep_size <= 1` gate and calling this path is the framework-side follow-up.
- The partition is assumed **contiguous** per rank (`expert_map`'s non-`-1` entries
  are `0..E_local-1`), matching the standard EP expert slice. A non-contiguous map
  works too as long as the local rows are dense `0..E_local-1`.
- No perf claim is made here: EP changes *which* experts a rank computes, not the
  per-expert GEMM, so per-rank throughput is the issue-#1 kernel on a smaller
  expert set. The win is **fitting** V4's experts across ranks, which the gate
  previously blocked.
