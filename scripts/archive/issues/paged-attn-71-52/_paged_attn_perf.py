import sys
sys.path.insert(0, "src")
from xkernels import verify
from xkernels.registry.writeback import record_measurement

# Decode shape: B=64 active requests, Qwen3-4B GQA (Hq=32, Hkv=8, D=128), seq_len~512.
cid = "paged_attention.triton@1.0.0"
shape = {"dtype": "bf16", "B": 64, "H_q": 32, "H_kv": 8, "D": 128,
         "block_size": 1, "max_seq_len": 512}
r = verify(cid, arch="nvidia_sm121", measure_perf=True, shapes=[shape])
ms = r["perf"]["ms"]
rid = r["artifacts"]["run_id"]
print("ms=%.6g" % ms, "passed=%s" % r["correctness"]["passed"], "run=%s" % rid)
record_measurement(cid, "nvidia_sm121", shape, "bf16", source=rid, ms=ms)
print("recorded.")
