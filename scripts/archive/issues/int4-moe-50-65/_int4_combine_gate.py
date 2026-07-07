"""Gate for issue #72: fused_moe_int4_w4a16(fused_combine=True) soundness fix.

Before the fix, fused_combine=True was silently wrong under @triton.autotune
(result ~N_configs x too big). The fix resolves a default config so the combine
path always takes the single-run direct launch (no autotune atomic-accumulate).

Verifies: combine==reference, combine==scratch, and the workspace combine path.
"""
import sys
sys.path.insert(0, "src")
import torch
from xkernels.ops.moe import (
    fused_moe_int4_w4a16, make_w4a16_weights, MoeInt4Workspace,
)
from xkernels.ops.moe.reference import moe_w4a16_ref
DEV = "cuda"; dtype = torch.bfloat16
torch.manual_seed(0)
ok = True
def check(name, cond, extra=""):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{extra}")

def inputs(M, E, N, K, top_k, dev, seed=1, gs=32):
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=seed)
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    tids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, packed, scale, tids, tw

# Reproduce the original bug shape + a sweep of decode/prefill buckets.
cases = [
    ("decode M=8",   8,  8, 256, 256, 4),
    ("decode M=4",   4,  8, 96,  64, 2),
    ("decode M=16", 16,  8, 256, 256, 4),
    ("decode M=32", 32,  8, 256, 256, 4),
    ("prefill M=128", 128, 8, 256, 256, 4),
]
# The bug's tolerance: combine was ~480x too big; correct is ~bf16 GEMM noise.
# int4 bf16 GEMM vs fp32 reference: the existing test uses atol=2e-2; combine
# does fp32 accumulate so it's at least as accurate as the bf16 scratch path.

print("=== §1 fused_combine=True now correct (was silently wrong under autotune) ===")
for tag, M, E, N, K, top_k in cases:
    A, packed, scale, tids, tw = inputs(M, E, N, K, top_k, DEV, seed=2)
    ref = moe_w4a16_ref(A, packed, scale, tids, tw, mul_routed_weight=True, fused_combine=False)
    o_scratch = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=False, backend="triton")
    o_combine = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
    ref_abs = ref.float().abs().max().item()
    cs_vs_ref = (o_combine.float() - ref.float()).abs().max().item()
    cs_vs_sc = (o_combine.float() - o_scratch.float()).abs().max().item()
    sc_vs_ref = (o_scratch.float() - ref.float()).abs().max().item()
    # combine within bf16-atol of the reference (2e-2, the existing int4 tolerance)
    check(f"{tag}: combine vs ref",
          cs_vs_ref < 2e-2,
          f"  combine-ref={cs_vs_ref:.6f} scratch-ref={sc_vs_ref:.6f} (|ref|max={ref_abs:.3f})")
    # combine agrees with scratch (both correct)
    check(f"{tag}: combine vs scratch",
          cs_vs_sc < 2e-2,
          f"  diff={cs_vs_sc:.6f}")

print()
print("=== §2 workspace combine path now active + correct (was guarded out) ===")
for tag, M, E, N, K, top_k in cases[:3]:
    A, packed, scale, tids, tw = inputs(M, E, N, K, top_k, DEV, seed=2)
    N_ = packed.shape[1]
    ws = MoeInt4Workspace.allocate(M, top_k, N_, dtype=dtype, device=DEV)
    o_combine = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton")
    o_ws = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton", workspace=ws)
    d = (o_combine.float() - o_ws.float()).abs().max().item()
    check(f"{tag}: workspace combine == alloc combine", d < 1e-3, f"  diff={d:.6f}")
    # re-zero on reuse (no 2x accumulate)
    o_ws2 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=True, backend="triton", workspace=ws)
    d2 = (o_ws.float() - o_ws2.float()).abs().max().item()
    check(f"{tag}: workspace re-zero on reuse", d2 < 1e-3, f"  diff={d2:.6f}")

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
