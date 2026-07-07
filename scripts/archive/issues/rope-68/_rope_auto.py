import torch, xkernels
T,H,D,P = 8,8,128,64
q = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
k = torch.randn(T,H,D, dtype=torch.bfloat16, device="cuda")
pos = torch.randint(0, P, (T,), dtype=torch.int32, device="cuda")
csc = torch.randn(P, D, dtype=torch.float32, device="cuda")
try:
    oq, ok = xkernels.apply_rope(q, k, pos, csc)   # backend='auto' -> tries triton (crash) -> fallback?
    torch.cuda.synchronize()
    print("auto RESULT: shapes", tuple(oq.shape), tuple(ok.shape), "dtypes", oq.dtype, ok.dtype, "finite=", torch.isfinite(oq).all().item())
except Exception as e:
    print("auto RAISED:", type(e).__name__, str(e)[:120])
# now try explicit reference to see if context survived
try:
    oq2, ok2 = xkernels.apply_rope(q, k, pos, csc, backend="reference")
    print("reference after auto: OK finite=", torch.isfinite(oq2).all().item())
except Exception as e:
    print("reference RAISED:", type(e).__name__, str(e)[:120])
