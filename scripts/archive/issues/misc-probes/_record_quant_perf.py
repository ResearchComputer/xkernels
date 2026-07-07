# Record baseline perf for the two fp8 quant triton cards on GB10 (§6.2 loop).
from xkernels import verify
from xkernels.registry.writeback import record_measurement

ARCH = "nvidia_sm121"
# per_token_group: a V4 activation group view. M=4096,K=7168,block=128 -> G=229376,B=128
ptg = {"dtype": "bf16", "G": 229376, "B": 128}
r = verify("per_token_group_quant_fp8.triton@1.0.0", arch=ARCH, measure_perf=True, shapes=[ptg])
ms = r["perf"]["ms"]; rid = r["artifacts"]["run_id"]
print("ptg:", r["correctness"]["passed"], "ms=%.6g" % ms, rid)
record_measurement("per_token_group_quant_fp8.triton@1.0.0", ARCH, ptg, "bf16",
                   source=rid, ms=ms, roofline_note=None) if False else None
record_measurement("per_token_group_quant_fp8.triton@1.0.0", ARCH, ptg, "bf16", source=rid, ms=ms)

# per_block: a V4 weight block view. N=K=7168,block=128 -> G=3136,B=16384
pb = {"dtype": "bf16", "G": 3136, "B": 16384}
r = verify("per_block_quant_fp8.triton@1.0.0", arch=ARCH, measure_perf=True, shapes=[pb])
ms = r["perf"]["ms"]; rid = r["artifacts"]["run_id"]
print("pb:", r["correctness"]["passed"], "ms=%.6g" % ms, rid)
record_measurement("per_block_quant_fp8.triton@1.0.0", ARCH, pb, "bf16", source=rid, ms=ms)
print("recorded.")
