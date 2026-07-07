"""Validate the fused_ffn fix: fp32 activation in BOTH reference and kernel -> bit-match?"""
import torch
import torch.nn.functional as F
import triton, triton.language as tl
from xkernels.registry.input_gen import generate_inputs


@triton.jit
def _swiglu_fp32(g_ptr, u_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    out = (g * tl.sigmoid(g)) * u        # fully fp32
    tl.store(out_ptr + offs, out, mask=mask)  # auto-convert to ptr dtype


def swiglu_fp32(g, u, BLOCK=1024):
    g = g.contiguous(); u = u.contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    _swiglu_fp32[(triton.cdiv(n, BLOCK),)](g, u, out, n, BLOCK=BLOCK)
    return out


def ffn_ref_fp32act(x, wg, wu, wd):
    g = x @ wg; u = x @ wu
    h = (F.silu(g.float()) * u.float()).to(g.dtype)   # activation in fp32 per spec
    return h @ wd


print("=== fused_ffn, all sweep points: OLD vs FIX ===")
from xkernels.registry import load_shape_sweep
for p in load_shape_sweep("ffn"):
    inp = generate_inputs("fused_ffn@1.0.0", p, 1729, "cuda")
    x, wg, wu, wd = inp["x"], inp["w_gate"], inp["w_up"], inp["w_down"]
    g = x @ wg; u = x @ wu
    # old reference (activation in input dtype) vs old kernel
    h_old_ref = F.silu(g) * u
    from xkernels.ops.ffn.triton.ffn_kernel import _swiglu_triton
    h_old_tri = _swiglu_triton(g, u)
    # fixed: both fp32 activation
    h_fix_ref = (F.silu(g.float()) * u.float()).to(g.dtype)
    h_fix_tri = swiglu_fp32(g, u)
    out_fix_ref = ffn_ref_fp32act(x, wg, wu, wd)
    out_fix_tri_g = g; out_fix_tri_u = u
    # full ffn with fixed kernel
    from xkernels.ops.ffn._activation import SwigluAct
    h_full = SwigluAct.apply(g, u, swiglu_fp32)
    out_fix_tri = h_full @ wd
    def rel(a, b):
        return ((a.float()-b.float()).abs() / (b.float().abs()+1e-8)).max().item()
    print(f"{p}")
    print(f"   ACT old |ref-tri| rel: {rel(h_old_ref, h_old_tri):.4e}   FIX |ref-tri| rel: {rel(h_fix_ref, h_fix_tri):.4e}")
    print(f"   OUT FIX |ref-tri| rel: {rel(out_fix_tri, out_fix_ref):.4e}  abs: {(out_fix_tri.float()-out_fix_ref.float()).abs().max().item():.4e}")
