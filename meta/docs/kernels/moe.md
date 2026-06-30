# Mixture-of-Experts — INT4 W4A16, MXFP4, expert parallelism, and the combine

The MoE family covers four kernels that together make DeepSeek-V4 / Kimi-K2.6's
routed-expert FFN runnable on AMD gfx942:

- **`fused_moe_int4_w4a16`** — the grouped INT4 W4A16 fused-MoE GEMM (Kimi-K2.6),
  launched from a **checked-in tuned config** (issues #1/#16).
- **`fused_moe_mxfp4`** — the V4-Flash MXFP4 grouped fused-MoE GEMM (issue #43).
- **`moe_align_block_size` + `moe_sum_reduce`** — the dispatch-builder and
  weighted top-k combine that bracket every grouped MoE GEMM (issues #4/#5).
- **Expert parallelism** (`expert_map`) and the **fused top-k combine** — two
  host/kernel extensions (issues #26 and #20).

> **One-line summary.** The grouped MoE GEMMs (INT4 23× / MXFP4 16× over the
> per-expert torch loop) are the headline wins. `moe_align_block_size` (34×) and
> `moe_sum_reduce` (8×) kill the host-side scheduling and reduction overhead that
> otherwise dominate decode. **Expert parallelism** is a host-wiring change (the
> grouped GEMM already gathers by expert id), and the **fused combine** is a
> correct optimization that *loses* on MI300A (kept opt-in/off) — atomics lose to
> the dedicated `moe_sum_reduce`.

---

## `fused_moe_int4_w4a16` — Kimi-K2.6 INT4 W4A16 grouped fused-MoE (issues #1/#16)

The portable mirror of tokenspeed's production AMD quantized fused-MoE. INT4
weights pack **8 nibbles per int32**, decoded `value = (nibble - 8) * scale`
(W4A16: bf16 activations). The op fuses the gate_up GEMM (clamped SwiGLU) and the
down GEMM as two grouped launches sharing one `@triton.jit` body.

**The win that made it fast:** runtime autotune was swapped for a **checked-in
tuned config** (issue #16). Winners were swept on MI300A for the Kimi-K2.6
per-rank shapes `w13` (N=4096, K=7168) and `w2` (N=7168, K=2048) across decode
buckets M∈{1,2,4,8,16,32,…} and prefill, stored in
`src/xkernels/ops/moe/triton/tuned_configs/` keyed by `(E,N,K,device,dtype)`; untuned
shapes fall back to the autotuner.

| M | gate_up (ms) | down (ms) |
|--:|---:|---:|
| 1  | 0.34 | 0.12 |
| 8  | 1.15 | — |
| 16 | 1.36 | 0.67 |

End-to-end (M=64, E=48, N=4096, K=7168, top_k=8): **1.36 ms, 23.2×** over the
per-expert dequant(int4→bf16)+matmul baseline (block-align excluded — that is the
`moe_align_block_size` row below, not timed here). Re-tune:
`sbatch scripts/archive/issues/tune_moe_int4_beverin.sbatch`.

---

## `fused_moe_mxfp4` — V4-Flash MXFP4 grouped fused-MoE (issue #43)

DeepSeek-V4-Flash routed experts are **MXFP4** (OCP MX: packed E2M1 4-bit values
+ `ue8m0` block-32 scales). On gfx942 the only working path was tokenspeed's
correctness-first `Mxfp4DequantBackend` — a Python per-expert loop that
dequantizes each active expert to bf16 and runs a plain `torch.matmul` grouped
GEMM. The OpenAI `triton_kernels` mxfp4 path is dead on this AMD Triton build
(`topk` emits a non-power-of-2 `tl.arange`). Nothing from the INT4 work was
reusable: V4's MXFP4 packs **2 E2M1 nibbles per uint8** with a **ue8m0**
power-of-two block scale and a **fused gate_up → clamped-SwiGLU → down** FFN.

### What ships

A new public op **`fused_moe_mxfp4`** + an autotuned Triton backend that consumes
the packed MXFP4 weights directly — **no full bf16 dequant** (a full dequant of
all 256 V4 experts is ~138 GB/rank and OOMs the APU). Only active experts are
touched.

The op fuses the whole routed-expert FFN as two grouped GEMMs sharing one
`@triton.jit` body (`_mxfp4_moe_gemm_kernel`, a `STAGE` constexpr picks the
epilogue):

1. **gate_up** (`STAGE=0`): `[M*top_k, 2*ispp] = A_gathered @ w13[e]^T`
   (contracted dim = `hidden`). Each program runs the **gate** and **up** halves
   of `w13` into the same N-tile (two fp32 accumulators) and fuses the V4 clamped
   SwiGLU `silu(clamp(gate, max=L)) * clamp(up, -L, L)` (`swiglu_limit=10.0`, no
   gpt-oss `+1`), with the optional per-expert `b13` added pre-activation, writing
   `act [M*top_k, ispp]`.
2. **down** (`STAGE=1`): `act @ w2[e]^T` (contracted dim = `ispp`), with the
   optional per-expert `b2` and the **routed-weighted top-k combine**
   (atomic-accumulate into the `[M, hidden]` fp32 output).

### Inline MXFP4 decode (in the K loop)

Matches `xkernels.ops.gather.mxfp4`: one coalesced `uint8` weight tile per 2
logical-K; the 2 nibbles unpacked with a broadcast shift `[0, 4]` (low nibble =
even / lower-K element); an E2M1 magnitude LUT `{0,.5,1,1.5,2,3,4,6}` (bit `0x8` =
sign) via a branchless `tl.where` ladder; the `ue8m0` block scale `2**(byte-127)`
fetched once per 32-element group and broadcast (`0xFF` NaN code → 0).

Tiling and the CDNA3 lowering knobs (`matrix_instr_nonkdim=16`, `waves_per_eu`,
`kpack`) mirror the INT4 W4A16 kernel; the gate_up stage carries two
accumulators, so the config space (`mxfp4_configs.py`) leans toward moderate
`BLOCK_SIZE_N` to avoid VGPR spills.

### API

```python
out = fused_moe_mxfp4(
    A,                       # [M, hidden] bf16
    w13, w13_scale,          # [E, 2*ispp, hidden//2] uint8, [E, 2*ispp, hidden//32] uint8
    w2,  w2_scale,           # [E, hidden, ispp//2]   uint8, [E, hidden, ispp//32]   uint8
    topk_ids, topk_w,        # [M, top_k] int32, [M, top_k] fp32
    b13=None, b2=None,       # optional [E, 2*ispp] / [E, hidden] bf16 biases
    swiglu_limit=10.0,       # V4 clamp; None disables
    group_size=32,
    mul_routed_weight=True,
    expert_map=None,         # optional EP global->local row map (see below)
    backend="auto",
)                            # -> [M, hidden] in A.dtype
```

### Performance

End-to-end (E=256, hidden=4096, ispp=512, top_k=6, M=48): **9.41 ms, 15.9×** over
the per-expert torch loop. Speedup grows with batch (≈1× at M=1, ≈16–19× at
M≥256) because the kernel amortizes expert scheduling and weight unpack across
tokens.

---

## Expert parallelism — `expert_map` (issue #26)

DeepSeek-V4 has 256–384 routed experts (top-6) and its mxfp4 weights do not fit
one 128 GB MI300A replicated, so the experts must be sharded across GPUs/nodes
(**expert parallelism**). In tokenspeed the AMD quantized fused-MoE `supports()`
ended with `return spec.ep_size <= 1 and ...`, so there was **no quantized MoE
path under EP** and V4 could not be served on MI300A.

**The root cause was host-side, not kernel-side.** The grouped GEMM tiles output
**by expert** and already filters per-block via an `expert_ids` sentinel, so the
kernel itself does not assume all experts are local. What was missing was the
**host-side wiring** to build the per-rank dispatch from a local expert subset.

### What ships

- **`moe_align_block_size_ep(topk_ids, block_size, num_experts, expert_map)`**
  (`ops/moe/w4a16.py`) — EP dispatch builder. The router still emits **global**
  `topk_ids`; `expert_map[g]` gives global expert `g`'s **local** weight-row in
  `[0, E_local)` (or `-1` if not on this rank). It remaps each routed slot to its
  local row, sends non-local slots to a sentinel id that the per-expert block
  builder never iterates (so they are dropped from this rank's compute), then
  reuses the well-tested `moe_align_block_size_ref` over the `E_local` local
  experts.
- **`fused_moe_int4_w4a16(..., expert_map=None)`** and
  **`fused_moe_mxfp4(..., expert_map=None)`** — new opt-in kwarg. When given,
  `packed`/`scale` are the rank-local slice and the op returns this rank's
  **partial** output. Summing the partials across the EP group (the production
  all-reduce) reconstructs the full dense result. `expert_map=None` (default) is
  byte-for-byte the prior single-rank behavior.

**No kernel-body change was required.** The grouped GEMM already consumes an
`expert_ids` array and gathers tokens by `sorted_token_ids // top_k`; pre-zeroed
output buffers mean non-local token-slots (never written / never atomic-added)
stay zero, so the per-rank reduce is exactly the partial. The partition is
assumed **contiguous** per rank (matching the standard EP expert slice); a
non-contiguous map works too as long as the local rows are dense `0..E_local-1`.

**The invariant the all-reduce relies on:**
```
sum_over_ranks  fused_moe(A, local_packed[r], local_scale[r],
                         topk_ids, topk_w, expert_map=emap[r])
  ==  full non-EP MoE output      (within bf16 atol/rtol = 2e-2)
```

Validated CPU (`TRITON_INTERPRET=1`) and on-device (beverin, bf16) across
ep∈{2,3,4}, `mul_routed ∈ {F,T}`, decode + prefill shapes — all PASS. No perf
claim: EP changes *which* experts a rank computes, not the per-expert GEMM; the
win is **fitting** V4's experts across ranks. The cross-rank all-reduce of the
partials is a separate collective (see `kernels/comm.md`) the serving layer owns.

---

## Fused top-k combine — kept opt-in / off by default (issue #20)

The fused weighted top-k combine (atomic-accumulate each expert's down-proj
result directly into `[M, hidden]`) is **implemented and numerically correct**, but
on MI300A at decode shapes it is **~25–35% slower** than the unfused
`GEMM + moe_sum_reduce` path. It ships **opt-in and off by default**
(`fused_combine=False`); the separate `moe_sum_reduce` kernel stays recommended.

| M | GEMM + moe_sum_reduce (ms) | fused combine (ms) | speedup | max\|err\| |
|--:|---:|---:|---:|---:|
| 1  | 0.1236 | 0.1603 | 0.77× | 0.0065 |
| 8  | 0.5581 | 0.7541 | 0.74× | 0.0107 |
| 16 | 0.6769 | 0.9010 | 0.75× | 0.0099 |

**Why fusion loses here:** the grouped GEMM tiles output **by expert**; a token
routes to `top_k=8` distinct experts, so each token's `[1, hidden]` row is
produced by 8 different program instances. Fusing means those 8 instances must
`atomic_add` into the same addresses — (1) atomic contention per N-tile, (2) fp32
write traffic (CDNA3 has native fp32 global `atomic_add`, so the combine buffer
is fp32 = 2× the bf16 bytes), (3) a pre-zero memset. The dedicated
`moe_sum_reduce` (already 8× vs torch) is fast enough that eliminating it does
not pay for the atomic + fp32 costs. The issue's traffic argument (skip the
`[M*top_k, hidden]` round-trip) is real but outweighed here. *(Same shape as the
`hierarchical_all_reduce` finding: a correct optimization the hardware does not
reward.)*

---

## `moe_align_block_size` + `moe_sum_reduce` — the dispatch & combine kernels

These bracket every grouped MoE GEMM and are themselves big wins over torch.

### `moe_align_block_size` (issue #4) — **33.8× / 24.5×** (MI300A / MI250X)

Shape M=16384, top_k=8, E=48, block=16. The Triton perf backend
(vLLM/SGLang-style 4-stage histogram + padded prefix-sum + scatter) is validated
bit-for-bit against the reference. The win is large because the reference pays a
full `argsort` plus a 48-iteration per-expert Python padding loop with per-step
host syncs; the kernel collapses that into 5 launches. The speedup holds across
token counts (≈14× at M=16, rising to ≈33× at M=16384 on MI300A). Profile verdict:
**dispatch/index-bound** (AI ≈ 0, ncu DRAM 3% / Compute 4%, occupancy 6%) — the
win is launch + python-loop overhead elimination, not bandwidth or compute.

### `moe_sum_reduce` (issue #5) — **8.0× / 7.8×** (MI300A / MI250X)

Shape M=8192, top_k=8, H=7168. Weighted top-k reduction; `top_k=8` is tiny so a
per-thread serial Kahan sum over k (no block-wide reduce), one CTA per token row.
Profile verdict: **strongly memory-bound** (MI300A AI 0.47 F/B; A100 85% DRAM,
19% compute, 95% occupancy). A100 achieved ~1.3–1.6 TB/s of the ~1.94 TB/s peak;
little left to tune on the bandwidth axis.

---

## Reproduce

```bash
# the consolidated benchmark table (INT4 + align + sum_reduce + pre/post)
scripts/cluster.sh submit --host beverin                              # bench_all_beverin.sbatch
scripts/cluster.sh run --host bristen python3 -u meta/benchmarks/bench_all.py
# MoE-specific
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/bench_moe_combine.py     # #20 fused combine
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/bench_moe_int4_w4a16.py
scripts/cluster.sh run --host beverin python3 -u meta/benchmarks/bench_moe_mxfp4.py
# correctness
scripts/cluster.sh run --host beverin python3 -u tests/test_moe_align_block_size.py
scripts/cluster.sh run --host beverin python3 -u tests/test_moe_sum_reduce.py
```
