import pytest
import torch

from xkernels import fused_ffn
from xkernels._backends import Backend
from xkernels._dispatch import registered_backends
from xkernels.ops.ffn.reference import ffn_reference
from xkernels.utils.testing import assert_close


def _inputs(M=16, d_model=32, d_ff=64, dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(M, d_model, dtype=dtype, generator=g)
    w_gate = torch.randn(d_model, d_ff, dtype=dtype, generator=g)
    w_up = torch.randn(d_model, d_ff, dtype=dtype, generator=g)
    w_down = torch.randn(d_ff, d_model, dtype=dtype, generator=g)
    return x, w_gate, w_up, w_down


def test_reference_matches_manual_swiglu():
    x, wg, wu, wd = _inputs()
    g = x @ wg
    # The reference accumulates the SwiGLU activation silu(g)*u = g*sigmoid(g)*u
    # in fp32 (Op Spec reduce_dtype), using the manual g*sigmoid(g) product (NOT
    # F.silu, whose fused fp32 impl differs at ULP) so the reference shares ONE
    # precision path with the Triton kernel -> bit-identical activation (issue #82).
    expected = (g.float() * torch.sigmoid(g.float()) * (x @ wu).float()).to(g.dtype) @ wd
    assert_close(ffn_reference(x, wg, wu, wd), expected)


def test_public_op_reference_backend_on_cpu():
    x, wg, wu, wd = _inputs()
    out = fused_ffn(x, wg, wu, wd, backend=Backend.REFERENCE)
    assert out.shape == (x.shape[0], wd.shape[1])
    assert_close(out, ffn_reference(x, wg, wu, wd))


def test_public_op_preserves_leading_dims():
    x = torch.randn(3, 5, 32)
    wg = torch.randn(32, 64)
    wu = torch.randn(32, 64)
    wd = torch.randn(64, 32)
    out = fused_ffn(x, wg, wu, wd, backend=Backend.REFERENCE)
    assert out.shape == (3, 5, 32)


_GPU_BACKENDS = [
    b for b in registered_backends("ffn") if b not in (Backend.REFERENCE,)
]


@pytest.mark.parametrize("backend", _GPU_BACKENDS, ids=lambda b: b.name)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_gpu_backend_matches_reference(backend, dtype):
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")
    x, wg, wu, wd = _inputs(dtype=dtype)
    x, wg, wu, wd = (t.cuda() for t in (x, wg, wu, wd))
    out = fused_ffn(x, wg, wu, wd, backend=backend)
    assert_close(out, ffn_reference(x, wg, wu, wd))
