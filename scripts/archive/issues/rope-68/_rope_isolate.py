# Isolate point 1 only, fresh process. Compare direct mathbody.launch vs the
# lower_to_triton launcher, on the EXACT make_inputs-generated tensors.
import sys, traceback
import torch
sys.path.insert(0, "src")
from xkernels.vkl.surface import spec_of
from xkernels.vkl.examples.rope import apply_rope
from xkernels.vkl.lower.triton import lower_to_triton
from xkernels.vkl.reference import make_inputs
from xkernels.vkl.lower import mathbody

spec = spec_of(apply_rope)
ir = mathbody.trace_ir(spec) if hasattr(mathbody, "trace_ir") else None
from xkernels.vkl.reference import trace_ir
ir = trace_ir(spec)
launcher = lower_to_triton(spec)
p = {"dtype": "bf16", "T": 8, "H": 8, "D": 64, "P": 64}

dev_in = {k: v.to("cuda") for k, v in make_inputs(spec, p, seed=0, device="cpu").items()}
print("positions max=", dev_in["positions"].max().item(), "P=64")

MODE = sys.argv[1] if len(sys.argv) > 1 else "direct"
if MODE == "dump":
    gen = mathbody._TritonGenMultiDim(ir, "bf16", mathbody._symbol_values(ir, dev_in))
    print(gen.kernel_source())
    sys.exit(0)
try:
    if MODE == "direct":
        out = mathbody.launch(ir, dev_in, "bf16", pattern="elementwise")
        print("DIRECT mathbody.launch: OK finite=", torch.isfinite(out["query_out"]).all().item())
    elif MODE == "launcher":
        out = launcher(**dev_in)
        print("LAUNCHER: OK finite=", torch.isfinite(out[0]).all().item())
except Exception:
    print(MODE, "CRASH")
    print("@@TB@"); print(traceback.format_exc()[-1500:]); print("@@TB@")
