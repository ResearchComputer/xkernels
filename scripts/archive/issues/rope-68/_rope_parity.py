import sys, json
sys.path.insert(0, "src")
from xkernels.vkl.surface import spec_of
from xkernels.vkl.examples.rope import apply_rope
from xkernels.vkl.lower.triton import register_dsl
from xkernels import verify_parity
register_dsl(spec_of(apply_rope), "triton")
r = verify_parity("apply_rope@1.0.0", archs=["nvidia_sm121"])
print("agree=", r.get("agree"), "max_pairwise_rel_err=", r.get("max_pairwise_rel_err"),
      "per_backend_runnable=", r.get("per_backend_runnable"))
