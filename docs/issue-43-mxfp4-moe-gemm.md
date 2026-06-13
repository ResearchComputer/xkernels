# Issue #43 — Fast MXFP4 grouped fused-MoE GEMM for DeepSeek-V4 (gfx942)

**Hardware:** AMD Instinct MI300A (gfx942, CDNA3). **Stack:** torch + `tokenspeed_triton`.
**Validation:** `slurm/test_moe_mxfp4_beverin.sbatch`.

## Context

DeepSeek-V4-Flash routed experts are **MXFP4** (OCP MX: packed E2M1 4-bit values +
`ue8m0` block-32 scales). On gfx942 the only working path was tokenspeed's
correctness-first `Mxfp4DequantBackend` — a Python per-expert loop that dequantizes
each active expert to bf16 and runs a plain `torch.matmul` grouped GEMM. The OpenAI
`triton_kernels` mxfp4 path is dead on this AMD Triton build (`topk` emits a
non-power-of-2 `tl.arange`), so neither `triton_kernel` nor `gluon_kernel` runs.
The per-expert dequant+matmul loop is the dominant V4-Flash decode cost on MI300A.

The existing xkernels MoE GEMM work (#1/#16/#26/#30) targets Kimi's **INT4 W4A16**
experts (8 nibbles per int32, `value = (nibble - 8) * scale`), so nothing was
reusable as-is: V4's MXFP4 packs **2 E2M1 nibbles per uint8** with a **ue8m0**
power-of-two block scale and a **fused gate_up → clamped-SwiGLU → down** FFN.

## What ships

A new public op **`fused_moe_mxfp4`** (`xkernels.ops.moe`) and an autotuned Triton
backend that consumes the packed MXFP4 weights directly — **no full bf16 dequant**
(a full dequant of all 256 V4 experts is ~138 GB/rank and OOMs the APU). Only
active experts are touched.

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

Inline MXFP4 decode in the K loop (matches `xkernels.ops.gather.mxfp4`): one
coalesced `uint8` weight tile per 2 logical-K, the 2 nibbles unpacked with a
broadcast shift `[0, 4]` (low nibble = even / lower-K element), an E2M1 magnitude
LUT `{0,.5,1,1.5,2,3,4,6}` (bit `0x8` = sign) via a branchless `tl.where` ladder,
and the `ue8m0` block scale `2**(byte-127)` fetched once per 32-element group and
broadcast across the group (`0xFF` NaN code → 0).

Tiling and the CDNA3 lowering knobs (`matrix_instr_nonkdim=16`, `waves_per_eu`,
`kpack`) mirror the INT4 W4A16 kernel; the gate_up stage carries two accumulators,
so the config space (`mxfp4_configs.py`) leans toward moderate `BLOCK_SIZE_N` to
avoid VGPR spills.

### Expert parallelism

The same `expert_map` plumbing as #26 is wired through: pass the rank-local weight
slice + a global→local `expert_map` (`-1` = not on this rank) and the op returns
this rank's **partial** output; non-local slots are dropped from compute and the
pre-zeroed combine buffer keeps them zero, so summing the partials reconstructs the
dense result.

## API

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
    expert_map=None,         # optional EP global->local row map
    backend="auto",
)                            # -> [M, hidden] in A.dtype
```

## Correctness (acceptance)

Acceptance: match the per-expert dequant-then-matmul stack within
`atol/rtol ~ 2e-2` (bf16).

| Check | Where | Result |
|-------|-------|--------|
| Triton == per-expert torch oracle, decode + prefill shapes, `mul_routed ∈ {F,T}`, `bias ∈ {F,T}` | CPU `TRITON_INTERPRET=1` (fp32) | **PASS** |
| reference backend == independent dequant-then-matmul oracle | CPU | **PASS** |
| `swiglu_limit=None` (unclamped) parity | CPU | **PASS** |
| EP: `sum(partials) == dense` (ep=2) | CPU | **PASS** |
| MXFP4 packed-weight dequant roundtrip | CPU | **PASS** |
| Triton == oracle, **V4 decode (M=48, E=256, hidden=4096, ispp=512, top_k=6)** | **beverin MI300A, bf16, 2e-2** | _see PR_ |

## Notes / scope

- **W4A16** (bf16 activations) — the portable path the tokenspeed dequant backend
  also serves. A W4A8 (fp8-activation) variant is a follow-up.
- Numerics match the tokenspeed reference exactly: same E2M1/ue8m0 decode and the
  same clamped SwiGLU. The bias convention is the same as tokenspeed — `b13`
  (column-parallel) added unconditionally; `b2` (row-parallel, replicated) is the
  caller's responsibility to add on one TP rank only.
- The two GEMMs are launched separately (gate_up writes an `act` scratch consumed
  by down) rather than as a single mega-kernel; this keeps the SwiGLU epilogue and
  the per-stage autotune tractable, matching the INT4 kernel's per-GEMM structure.
