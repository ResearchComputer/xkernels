# Canonical gate: register_dsl + verify + verify_parity on the real cards.
import sys, json, traceback
import torch
sys.path.insert(0, "src")
from xkernels.vkl.surface import spec_of
from xkernels.vkl.examples.rope import apply_rope
from xkernels.vkl.lower.triton import register_dsl
from xkernels import verify, verify_parity

register_dsl(spec_of(apply_rope), "triton")
for cid in ("apply_rope.triton@1.0.0", "apply_rope.reference@1.0.0"):
    r = verify(cid, arch="nvidia_sm121")
    c = r["correctness"]
    print(f"verify({cid}): compiled={r['compiled']} passed={c['passed']}",
          f"max_abs={c.get('max_abs_err')} max_rel={c.get('max_rel_err')}",
          f"n_points={c.get('n_points')} err={r['artifacts'].get('error')}")
r = verify_parity("apply_rope@1.0.0")
print("verify_parity:", "agree=", r.get("agree"), "max_rel=", r.get("max_rel"))

# Also measure perf for the compounding loop.
r = verify("apply_rope.triton@1.0.0", arch="nvidia_sm121", measure_perf=True,
           shapes=[{"dtype": "bf16", "T": 64, "H": 8, "D": 128, "P": 512}])
print("perf V4-shape: ms=", r["perf"]["ms"], "run=", r["artifacts"]["run_id"])
