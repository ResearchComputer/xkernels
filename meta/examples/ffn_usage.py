"""Minimal xkernels usage: a differentiable SwiGLU FFN forward+backward."""
import torch

from xkernels import fused_ffn

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(4, 8, 512, device=device, requires_grad=True)
w_gate = torch.randn(512, 1024, device=device, requires_grad=True)
w_up = torch.randn(512, 1024, device=device, requires_grad=True)
w_down = torch.randn(1024, 512, device=device, requires_grad=True)

y = fused_ffn(x, w_gate, w_up, w_down)  # backend="auto"
y.sum().backward()

print("output:", tuple(y.shape), "| x.grad:", tuple(x.grad.shape))
