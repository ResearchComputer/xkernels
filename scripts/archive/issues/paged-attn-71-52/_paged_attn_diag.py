import torch
import xkernels

def run(B, Hq, Hkv, D, bs, msl, dt):
    maxb = (msl + bs - 1) // bs
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, Hq, D, generator=g, device="cuda", dtype=dt)
    kc = torch.randn(B*maxb, bs, Hkv, D, generator=g, device="cuda", dtype=dt)
    vc = torch.randn(B*maxb, bs, Hkv, D, generator=g, device="cuda", dtype=dt)
    bt = torch.arange(B*maxb, device="cuda", dtype=torch.int32).reshape(B, maxb)
    sl = torch.randint(1, msl+1, (B,), generator=g, device="cuda", dtype=torch.int32)
    kt = xkernels.paged_attention(q, kc, vc, bt, sl, scale=D**-0.5, backend="triton")
    ref = xkernels.paged_attention(q, kc, vc, bt, sl, scale=D**-0.5, backend="reference")
    diff = (kt.float()-ref.float()).abs()
    rel = (diff/ref.float().abs().clamp_min(1e-6)).max().item()
    print(f"dt={str(dt):14s} B={B} Hq={Hq} Hkv={Hkv} D={D} sl~{msl}: max_abs={diff.max().item():.3e} max_rel={rel:.3e}")

print("=== bf16 vs fp32 at the SAME shape (isolate dtype vs algorithm) ===")
for dt in [torch.float32, torch.bfloat16, torch.float16]:
    run(8, 32, 8, 128, 1, 256, dt)
print("=== seq_len scaling (does the flash gap grow with context?) ===")
for msl in [16, 64, 256, 1024]:
    run(8, 32, 8, 128, 1, msl, torch.bfloat16)
print("=== D=64 (Llama-70B) ===")
run(8, 64, 8, 64, 1, 256, torch.bfloat16)
