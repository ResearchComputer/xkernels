from xkernels.vkl import spec_of, register_dsl
from xkernels.vkl.examples import rope
from xkernels.vkl.lower import triton as low
spec = spec_of(rope.apply_rope)
# Try to get the generated source via the lowering
import inspect
try:
    src = low.generate_kernel_source(spec, "triton")  # guess API
    print(src[:4000])
except Exception as e:
    print("generate_kernel_source not the API:", type(e).__name__, e)
    print("lower.triton exports:", [n for n in dir(low) if not n.startswith('_')])
