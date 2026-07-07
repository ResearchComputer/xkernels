import sys
sys.path.insert(0, "src")
from xkernels import verify
from xkernels.vkl.surface import spec_of
from xkernels.vkl.examples.rope import apply_rope
from xkernels.vkl.lower.triton import register_dsl
from xkernels.registry.writeback import record_measurement
register_dsl(spec_of(apply_rope), "triton")
shape = {"dtype": "bf16", "T": 64, "H": 8, "D": 128, "P": 512}
r = verify("apply_rope.triton@1.0.0", arch="nvidia_sm121", measure_perf=True, shapes=[shape])
ms = r["perf"]["ms"]; rid = r["artifacts"]["run_id"]
print("ms=%.6g run=%s passed=%s" % (ms, rid, r["correctness"]["passed"]))
record_measurement("apply_rope.triton@1.0.0", "nvidia_sm121", shape, "bf16", source=rid, ms=ms)
print("recorded.")
