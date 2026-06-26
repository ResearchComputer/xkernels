"""One-shot write-back of the 2026-06-26 benchmark/profile campaign into the
cards' ``perf.measured`` (docs/library.md §6.2 — the compounding loop).

Numbers are transcribed from wiki/02-benchmarks.md (ms, tflops) and
wiki/03-profiling.md (achieved_bw_pct = ncu DRAM Throughput % on A100). Every
entry cites a reproducible SLURM job ``source`` and an ``arch`` (§2.4 invariant;
un-sourced/arch-less entries are dropped by the loader).

Uses registry.writeback.record_measurement (validated, dedup-by-point path) so
the written cards are guaranteed to pass the loader.
"""
from xkernels.registry.writeback import record_measurement

BEV_BENCH = "bench_all.py on beverin MI300A gfx942 (slurm #530058)"
BEV_FP8 = "bench_fp8_blockscale_gemm.py on beverin MI300A gfx942 (slurm #530059)"
BRI = "bench_one.py on bristen A100 sm_80 (slurm #71892); ncu roof profile (slurm #71893-71899)"


def moe_int4_tflops(ms: float) -> float:
    """2 * M_eff * N * K, M_eff = M*top_k = 64*8, N=4096, K=7168."""
    return 2 * 64 * 8 * 4096 * 7168 / (ms * 1e-3) / 1e12


# (card_id, arch, shape, dtype, knobs, source, ms, tflops, achieved_bw_pct)
ROWS = [
    # ---- beverin (amd_cdna3) : bench_all #530058 ----
    ("dual_rmsnorm.triton@1.0.0", "amd_cdna3",
     {"T": 8192, "D1": 1536, "D2": 512}, "bf16", {}, BEV_BENCH, 0.054, None, None),
    ("moe_sum_reduce.triton@1.0.0", "amd_cdna3",
     {"M": 8192, "top_k": 8, "H": 7168}, "bf16", {}, BEV_BENCH, 0.373, None, None),
    ("mha_merge_state.triton@1.0.0", "amd_cdna3",
     {"T": 8192, "H": 128, "D": 128}, "bf16", {}, BEV_BENCH, 0.784, None, None),
    ("sparse_mla_attention.triton@1.0.0", "amd_cdna3",
     {"T": 8, "H": 128, "D": 512, "topk": 512, "Kv": 8192},
     "bf16", {}, BEV_BENCH, 0.111, None, None),
    ("mhc_pre.triton@1.0.0", "amd_cdna3",
     {"T": 8, "hc_mult": 4, "hidden": 4096}, "bf16", {}, BEV_BENCH, 0.080, None, None),
    # hc_prenorm_gemm at T=8 is launch-overhead-dominated (0.013 ms); recorded as-is.
    ("hc_prenorm_gemm.triton@1.0.0", "amd_cdna3",
     {"T": 8, "K": 16384, "N": 24, "n_splits": 16}, "bf16", {}, BEV_BENCH, 0.013, None, None),
    ("moe_align_block_size.triton@1.0.0", "amd_cdna3",
     {"M": 16384, "top_k": 8, "E": 48, "block": 16}, "int32", {}, BEV_BENCH, 1.644, None, None),
    ("fused_ffn.triton@1.0.0", "amd_cdna3",
     {"M": 4096, "d_model": 4096, "d_ff": 11008}, "fp16", {}, BEV_BENCH, 5.285, None, None),
    ("moe_int4_w4a16.triton@1.0.0", "amd_cdna3",
     {"M": 64, "E": 48, "N": 4096, "K": 7168, "top_k": 8},
     "bf16", {}, BEV_BENCH, 1.364, None, None),

    # ---- beverin (amd_cdna3) : mm_fp8_blockscale, native fp8 MFMA, #530059 ----
    # tflops are the bench script's own computed values (authoritative).
    ("mm_fp8_blockscale.triton@1.0.0", "amd_cdna3",
     {"M": 2048, "N": 512, "K": 7168, "block": 128}, "fp8",
     {"path": "mfma"}, BEV_FP8, 0.061, 246.1, None),
    ("mm_fp8_blockscale.triton@1.0.0", "amd_cdna3",
     {"M": 4096, "N": 7168, "K": 2048, "block": 128}, "fp8",
     {"path": "mfma"}, BEV_FP8, 0.331, 363.6, None),

    # ---- bristen (nvidia_sm80) : bench_one #71892 + ncu roof (achieved_bw_pct) ----
    ("dual_rmsnorm.triton@1.0.0", "nvidia_sm80",
     {"T": 8192, "D1": 1536, "D2": 512}, "bf16", {}, BRI, 0.053, None, 68.0),
    ("moe_sum_reduce.triton@1.0.0", "nvidia_sm80",
     {"M": 8192, "top_k": 8, "H": 7168}, "bf16", {}, BRI, 0.651, None, 85.0),
    ("mha_merge_state.triton@1.0.0", "nvidia_sm80",
     {"T": 8192, "H": 128, "D": 128}, "bf16", {}, BRI, 1.046, None, 41.0),
    ("moe_align_block_size.triton@1.0.0", "nvidia_sm80",
     {"M": 16384, "top_k": 8, "E": 48, "block": 16}, "int32", {}, BRI, 0.883, None, 3.0),
    ("fused_ffn.triton@1.0.0", "nvidia_sm80",
     {"M": 4096, "d_model": 4096, "d_ff": 11008}, "fp16", {}, BRI, 4.288, None, 87.0),
    ("moe_int4_w4a16.triton@1.0.0", "nvidia_sm80",
     {"M": 64, "E": 48, "N": 4096, "K": 7168, "top_k": 8}, "bf16", {}, BRI, 2.225, None, 18.0),
]


def main() -> None:
    # moe_int4 tflops computed from shape (not printed by bench).
    for i, (cid, arch, shape, dtype, knobs, src, ms, tf, bw) in enumerate(ROWS):
        if tf is None and cid.startswith("moe_int4"):
            tf = round(moe_int4_tflops(ms), 1)
        res = record_measurement(
            cid, arch=arch, shape=shape, dtype=dtype, source=src,
            knobs=knobs or None, tflops=tf, achieved_bw_pct=bw, ms=ms,
        )
        print(f"[{i+1:02d}] {cid:34s} {arch:13s} ms={ms} "
              f"tflops={tf} bw%={bw} -> total={res['total_measurements']}")


if __name__ == "__main__":
    main()
