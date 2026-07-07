"""Focused root-cause diagnostic for issue #82 — element-level + fp32 ground truth."""
import torch
import torch.nn.functional as F

from xkernels.registry.input_gen import generate_inputs
from xkernels.ops.ffn.triton.ffn_kernel import _swiglu_triton


def fp32_truth(x, wg, wu, wd):
    x, wg, wu, wd = (t.float() for t in (x, wg, wu, wd))
    return (F.silu(x @ wg) * (x @ wu)) @ wd


print("=== fused_ffn pt5 (bf16, M=128,K=64,N=128) ===")
p = {"dtype": "bf16", "M": 128, "K": 64, "N": 128}
inp = generate_inputs("fused_ffn@1.0.0", p, 1729, "cuda")
x, wg, wu, wd = inp["x"], inp["w_gate"], inp["w_up"], inp["w_down"]
g = x @ wg
u = x @ wu
# three activation variants
h_ref = F.silu(g) * u                     # reference path (bf16 throughout)
h_tri = _swiglu_triton(g, u)              # triton kernel
h_fp32act = (F.silu(g.float()) * u.float()).to(g.dtype)  # silu in fp32, then round
# full FFN outputs
out_ref = h_ref @ wd
out_tri = h_tri @ wd
out_fp32act = h_fp32act @ wd
truth = fp32_truth(x, wg, wu, wd)
print(f"activation diff |ref-tri| max: {(h_ref.float()-h_tri.float()).abs().max().item():.4e}")
print(f"activation diff |ref-fp32act| max: {(h_ref.float()-h_fp32act.float()).abs().max().item():.4e}")
print(f"activation |tri-fp32truth| max: {(h_tri.float()-F.silu(g.float()).float()*u.float()).abs().max().item():.4e}")
print(f"OUTPUT |out_ref - truth|: abs={(out_ref.float()-truth).abs().max().item():.4e} rel={((out_ref.float()-truth).abs()/(truth.abs()+1e-8)).max().item():.4e}")
print(f"OUTPUT |out_tri - truth|: abs={(out_tri.float()-truth).abs().max().item():.4e} rel={((out_tri.float()-truth).abs()/(truth.abs()+1e-8)).max().item():.4e}")
print(f"OUTPUT |out_ref - out_tri|: abs={(out_ref.float()-out_tri.float()).abs().max().item():.4e} rel={((out_ref.float()-out_tri.float()).abs()/(out_ref.float().abs()+1e-8)).max().item():.4e}")
print(f"truth absmax: {truth.abs().max().item():.3f}; out_ref absmax {out_ref.abs().max().item():.3f}; out_tri absmax {out_tri.abs().max().item():.3f}")

print()
print("=== moe_sum_reduce pt4 (fp32, M=8192, top_k=8, H=7168) ===")
from xkernels.registry import reference_callable, backend_callable
p = {"dtype": "fp32", "M": 8192, "top_k": 8, "H": 7168}
inp = generate_inputs("moe_sum_reduce@1.0.0", p, 1729, "cuda")
ref = reference_callable("moe_sum_reduce@1.0.0")(**inp)
tri = backend_callable("moe_sum_reduce@1.0.0", "TRITON")(**inp)
print("input y shape", inp["y"].shape, "w shape", inp["w"].shape, "w sample", inp["w"].flatten()[:4])
print(f"|ref-tri| abs max: {(ref.float()-tri.float()).abs().max().item():.4e}  rel max: {((ref.float()-tri.float()).abs()/(ref.float().abs()+1e-8)).max().item():.4e}")
print(f"ref absmax {ref.abs().max().item():.4e}, tri absmax {tri.abs().max().item():.4e}")
# where is the worst element?
diff = (ref.float()-tri.float()).abs()
idx = diff.argmax()
print(f"worst element flat idx {idx.item()}: ref={ref.flatten()[idx].item():.6f} tri={tri.flatten()[idx].item():.6f}")
# is it a reduction-order issue? recompute in fp64
y, w = inp["y"].double(), inp["w"].double()
manual = (y * w.unsqueeze(-1)).sum(dim=1)
print(f"fp64 manual vs ref rel: {((manual.float()-ref.float()).abs()/(ref.float().abs()+1e-8)).max().item():.4e}")
print(f"fp64 manual vs tri rel: {((manual.float()-tri.float()).abs()/(ref.float().abs()+1e-8)).max().item():.4e}")
