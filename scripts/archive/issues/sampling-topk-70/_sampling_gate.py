import torch
import xkernels
from xkernels import verify, verify_parity

print("=== reference + triton + parity on the sweep ===")
for cid in ("sampling_from_probs.reference@1.0.0",
            "sampling_from_probs.triton@1.0.0",
            "top_k_sampling_from_probs.reference@1.0.0",
            "top_k_sampling_from_probs.triton@1.0.0"):
    v = verify(cid, arch="nvidia_sm121")
    c = v["correctness"]
    print(cid.split("@")[0], "compiled=", v["compiled"], "passed=", c["passed"],
          "max_abs=", c["max_abs_err"], "n=", c["n_points"], "det=", v["determinism_check"])
    if c["failing_shapes"]:
        print("   FAIL:", c["failing_shapes"][0])
for op in ("sampling_from_probs@1.0.0", "top_k_sampling_from_probs@1.0.0"):
    p = verify_parity(op, archs=["nvidia_sm121"])
    print("parity", op.split("@")[0], "agree=", p.get("agree"), "max_rel=", p.get("max_pairwise_rel_err"))

print("=== STRESS: the exact case that failed (B=256,V=4096) + larger ===")
for B, V, tk in [(256, 4096, 40), (512, 8192, 50), (1024, 4096, 40), (64, 16384, 1)]:
    for dt in (torch.float32, torch.bfloat16):
        logits = torch.randn(B, V, device="cuda", dtype=dt) * 3.0
        probs = torch.softmax(logits.float(), dim=1).to(dt)
        u = torch.rand(B, device="cuda") * 0.999
        kr = xkernels.sampling_from_probs(probs, u, backend="triton")
        ref = xkernels.sampling_from_probs(probs, u, backend="reference")
        s_ok = bool((kr == ref).all())
        kt = xkernels.top_k_sampling_from_probs(probs, u, tk, backend="triton")
        reft = xkernels.top_k_sampling_from_probs(probs, u, tk, backend="reference")
        t_ok = bool((kt == reft).all())
        print(f"B={B:4d} V={V:5d} top_k={tk:2d} {str(dt):14s}: sampling={s_ok} top_k={t_ok}")
        if not (s_ok and t_ok):
            # locate the mismatch
            bad = (kr != ref).nonzero()[:3]
            print("   sampling mismatches at rows:", bad.flatten().tolist())
