import sys
sys.path.insert(0, "src")
from xkernels import verify
from xkernels.registry.writeback import record_measurement

# Prefill shape: 3 seqs, Qwen3-4B GQA (Hq=32/Hkv=8/D=128), each ~512 tokens
# (a realistic prompt batch). page_size=1 (mini-sglang's RadixCache).
cid = "paged_attention_prefill.triton@1.0.0"
shape = {"dtype": "bf16", "num_seqs": 3, "max_seq_len_q": 512, "max_seq_len_k": 512,
         "H_q": 32, "H_kv": 8, "D": 128, "block_size": 1, "prefix_frac": 0.0}
r = verify(cid, arch="nvidia_sm121", measure_perf=True, shapes=[shape])
ms = r["perf"]["ms"]
rid = r["artifacts"]["run_id"]
print("ms=%.6g" % ms, "passed=%s" % r["correctness"]["passed"], "run=%s" % rid)
record_measurement(cid, "nvidia_sm121", shape, "bf16", source=rid, ms=ms)
print("recorded.")
