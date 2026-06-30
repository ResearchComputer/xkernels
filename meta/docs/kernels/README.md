# Per-kernel optimizations & performance

The per-kernel-family deep dives: what each op computes, the kernel strategy, the
**measured performance**, and the key lessons (positive *and* negative results).
These consolidate the older per-issue write-ups into one coherent doc per family.

The cross-cutting **benchmark & profile campaign** that produced most of these
numbers lives in `meta/wiki/` (both arches, roofline, regime, gotchas); this
directory is the per-op narrative on top of it.

## Families

| Doc | Ops covered | Headline |
|---|---|---|
| [`gemm.md`](gemm.md) | bf16 dense GEMM, `mm_fp8_blockscale` (portable + native fp8 MFMA) | fp8 MFMA **359 TFLOP/s**, 3.4–9.1× over torch (needs `fnuz` operands); bf16 is an env knob, not a kernel |
| [`moe.md`](moe.md) | `fused_moe_int4_w4a16`, `fused_moe_mxfp4`, EP (`expert_map`), fused combine, `moe_align_block_size`, `moe_sum_reduce` | INT4 **23×** / MXFP4 **16×**; align **34×**; sum_reduce **8×**; fused combine & EP analysis |
| [`attention.md`](attention.md) | `sparse_mla_attention`, `dsa_indexer_logits`, `mxfp4_paged_gather`, `mha_merge_state` | V4's full attention path on gfx942; one compute kernel serves CSA/HCA/SWA |
| [`mhc.md`](mhc.md) | `hc_prenorm_gemm`, `mhc_pre`, `mhc_post`, V4 perf pass | the last NVIDIA-only-gated V4 layer, now portable; pre/post **35.5×** |
| [`comm.md`](comm.md) | `hierarchical_all_reduce`, fused residual+RMSNorm | correct + capture-cracked, but the perf premise does not hold (RCCL is already topology-aware) |
| [`norm.md`](norm.md) | `dual_rmsnorm`, `fused_ffn` | both memory/launch-bound, at the roofline — no compute headroom |

## How to read these

Each doc follows the same shape so they compose into a coherent source of truth:

- **One-line summary** — the headline, including honest negative results.
- **The math / contract** — what the op computes (matches the Op Spec).
- **Implementation strategy** — tiling, the matrix-engine path, the host wiring.
- **Performance results** — measured tables, with the arch + shape + baseline.
- **Key lessons / gotchas** — cross-linked into `meta/wiki/` where they're general.

Every number cites an on-device job id or a reproducible `meta/benchmarks/` /
`tests/` path. Where a kernel *lost* (portable fp8 GEMM, fused combine,
hierarchical all-reduce), the doc says so and explains why — these negative
results are recorded so the next task skips them.
