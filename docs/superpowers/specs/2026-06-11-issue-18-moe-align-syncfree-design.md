# Issue #18 — sync-free / fixed-output-shape `moe_align_block_size_triton`

**Date:** 2026-06-11
**Status:** Approved
**Issue:** ResearchComputer/kernels#18
**Target hardware:** AMD Instinct MI300A (gfx942), HIP-graph-captured decode.

## Purpose

`moe_align_block_size_triton` (#4/#15) is fast in a microbench but **cannot run
inside a HIP/CUDA-graph-captured decode region**, which is where it matters for
MoE serving (60 layers × every decode step, captured at concurrency 1→16). Two
trailing operations block capture:

1. `n = int(num_post.item())` — a device→host **sync** (forbidden during capture;
   also re-serializes the launch stream on the host-bound decode path).
2. `expert_ids[: n // block_size]` — a **data-dependent output shape** (a captured
   graph needs static shapes).

This change adds a **sync-free, fixed-output-shape mode** so the faster Triton
align can replace the torch reference on the captured decode hot path, not just
eager prefill.

## Design

Add `truncate: bool = True` to the op (dispatch + both backends):

- **`truncate=True`** (default, unchanged): `.item()` + `expert_ids[: n // block_size]`.
  Eager/prefill callers and all existing consumers are byte-for-byte unaffected.
- **`truncate=False`** (capturable): skip `.item()`; return `expert_ids` at its
  full allocated length `max_blocks = cdiv(max_pad, block_size)`. All shapes are
  static and no host sync occurs, so the five kernel launches plus the output
  allocations are graph-capturable and replayable.

### Unused-block sentinel

In `truncate=False`, blocks past `num_tokens_post_padded` are "unused". The
existing `_align_expert_ids` already computes, for block `b` at `off = b*block_size`,
`expert = #{e in [1,num_experts] : cumsum[e] <= off}`; for an unused block
`off >= cumsum[num_experts] = total_padded`, so `expert == num_experts` (one past
the valid 0-based range). Map that to sentinel **0** to match the tokenspeed
`moe_align_block_size_amd` contract the issue cites:

```python
expert = tl.where(expert >= num_experts, 0, expert)
```

This is safe in **both** modes: the fused-MoE GEMM consumer early-returns when
`pid_m * BLOCK_SIZE_M >= num_tokens_post_padded` and never reads the tail, and the
`truncate=True` path slices the tail off — so its output is unchanged. Sentinel 0
(not -1) matches the reference; -1 is reserved by the GEMM for EP-filtered blocks,
which this op does not produce.

### Why it's capturable after the change

`numel`, `max_pad`, `max_blocks`, `tokens_per_thread`, and every grid are derived
only from `topk_ids.shape`, `block_size`, `num_experts` — all static for a given
captured decode shape. The output buffers (`sorted_ids`, `expert_ids`, `num_post`,
`tokens_cnts`, `cumsum`) are fixed-size allocations (capturable via the graph's
private mempool). The removed `.item()` was the sole host sync.

## Components

1. **`src/xkernels/ops/moe/triton/align_kernel.py`** — `_align_expert_ids` gains
   the sentinel `tl.where`; `moe_align_block_size_triton` gains `truncate=True` and
   branches the return (trim vs full).
2. **`src/xkernels/ops/moe/w4a16.py`** — `moe_align_block_size_ref` gains
   `truncate=True`; when `False`, pad `expert_ids` to `max_blocks` with sentinel 0
   (so the two backends are parity-comparable in fixed-shape mode).
3. **`src/xkernels/ops/moe/align.py`** — `moe_align_block_size(..., truncate=True)`
   threads the kwarg through `dispatch`.
4. **`tests/test_moe_align_block_size.py`** — add `truncate=False` cases:
   fixed `max_blocks` length, `full[:used]` equals `truncate=True` and the
   reference, tail == 0; Triton and reference agree in both modes.
5. **`benchmarks/probe_align_capture.py`** + **`slurm/probe_align_capture_beverin.sbatch`**
   — GPU proof: warm up, then capture `truncate=False` in `torch.cuda.graph`, replay,
   and assert the replayed `(sorted_ids, expert_ids, num_post)` matches eager. Also
   confirms `truncate=True` fails/forbids capture (the sync), as the contrast.

## Data flow

`topk_ids → 5 static-shape kernel launches → (sorted_ids[max_pad],
expert_ids[max_blocks], num_post[1])`. `truncate=True` then host-syncs and trims;
`truncate=False` returns the fixed-shape triple directly. The fused-MoE GEMM
consumes `num_post` on-device and ignores `expert_ids` past `num_post`.

## Error handling / edge cases

- `truncate=False` on the **reference** is supported (full-length, sentinel 0) for
  parity testing only; the reference still uses `.item()`-equivalent python and is
  not itself capturable — that is fine, it is the eager oracle.
- All-tokens-to-one-expert / empty chunks: unchanged from the validated #4 kernel;
  the sentinel `where` only touches strictly-unused trailing blocks.
- Capture requires a warmup run before `torch.cuda.graph` (JIT compile is not
  capturable); the probe script does the warmup.

## Testing

- **CPU / `TRITON_INTERPRET=1`:** parity + fixed-shape + sentinel tests (above).
- **On device (beverin, gfx942):** the capture/replay probe demonstrates the
  `truncate=False` op is sync-free and graph-capturable and replays correctly.

## Out of scope

- Caller-preallocated output buffers (the graph's private mempool already handles
  the fixed-size allocations; no API change needed for capturability).
- Wiring the captured align into a specific serving loop (tokenspeed integration).
- Changing the default (`truncate=True` stays default to preserve every caller).

## Deliverable acceptance

- `truncate=False` returns fixed `max_blocks`-length `expert_ids` with no host sync;
  `full[:used]` matches `truncate=True` and the reference; tail == 0.
- Existing align tests and consumers unchanged (`truncate=True` default).
- On-device capture/replay probe passes on MI300A.
- Result reported on #18.
