import torch
import xkernels
from xkernels import verify, verify_parity

for cid in ("topk_softmax.reference@1.0.0", "topk_softmax.triton@1.0.0"):
    v = verify(cid, arch="nvidia_sm121")
    c = v["correctness"]
    print(cid, "compiled=", v["compiled"], "passed=", c["passed"],
          "max_abs=", c["max_abs_err"], "max_rel=", c["max_rel_err"],
          "n=", c["n_points"], "failing=", len(c["failing_shapes"]))
    if c["failing_shapes"]:
        print("   first fail:", c["failing_shapes"][0])

p = verify_parity("topk_softmax@1.0.0", archs=["nvidia_sm121"])
print("parity: agree=", p.get("agree"), "max_rel=", p.get("max_pairwise_rel_err"))

# direct id-exactness on the previously-failing bf16 E=256 shape
g = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
wt, it = xkernels.topk_softmax(g, 8, backend="triton")
wr, ir = xkernels.topk_softmax(g, 8, backend="reference")
print("E=256 bf16: ids_exact=", bool((it == ir).all()),
      "weights_maxabs=", (wt.float() - wr.float()).abs().max().item())
