import sys
sys.path.insert(0, "src")
from xkernels import verify
from xkernels.registry.writeback import record_measurement

for cid, shape in [
    ("sampling_from_probs.triton@1.0.0", {"dtype": "fp32", "B": 256, "V": 4096}),
    ("top_k_sampling_from_probs.triton@1.0.0",
     {"dtype": "fp32", "B": 256, "V": 4096, "top_k": 40}),
]:
    r = verify(cid, arch="nvidia_sm121", measure_perf=True, shapes=[shape])
    ms = r["perf"]["ms"]
    rid = r["artifacts"]["run_id"]
    print(cid, "ms=%.6g" % ms, "passed=%s" % r["correctness"]["passed"])
    record_measurement(cid, "nvidia_sm121", shape, "fp32", source=rid, ms=ms)
print("recorded.")
