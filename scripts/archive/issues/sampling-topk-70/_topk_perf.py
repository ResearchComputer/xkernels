import sys
sys.path.insert(0, "src")
from xkernels import verify
from xkernels.registry.writeback import record_measurement
# DeepSeek-V3 MoE gate: M=4096 tokens, E=256 experts, topk=8, bf16, renorm.
shape = {"dtype": "bf16", "M": 4096, "E": 256, "topk": 8, "renormalize": True}
r = verify("topk_softmax.triton@1.0.0", arch="nvidia_sm121", measure_perf=True, shapes=[shape])
ms = r["perf"]["ms"]; rid = r["artifacts"]["run_id"]
print("V4-shape ms=%.6g run=%s passed=%s" % (ms, rid, r["correctness"]["passed"]))
record_measurement("topk_softmax.triton@1.0.0", "nvidia_sm121", shape, "bf16", source=rid, ms=ms)
print("recorded.")
