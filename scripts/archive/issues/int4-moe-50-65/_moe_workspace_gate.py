"""MoE workspace gate (issue #52): workspace==alloc, reuse, graph capture.

Covers moe_align_block_size (5 scratch buffers), fused_moe_int4_w4a16
(combine + scratch), and fused_moe_mxfp4 (act + out). Verifies the
atomic-add outputs are correctly RE-ZEROED on reuse (no stale-accumulate
corruption) and that whole-call CUDA graph capture works.
"""
import sys
sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.moe import (
    MoeAlignWorkspace, MoeInt4Workspace, MoeMxfp4Workspace,
    dequant_w4a16, fused_moe_int4_w4a16, make_w4a16_weights,
    fused_moe_mxfp4, make_mxfp4_moe_weights,
)
DEV = "cuda"
dtype = torch.bfloat16
torch.manual_seed(0)
ok = True
def check(name, cond):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

def int4_inputs(M, E, N, K, top_k, dev, seed=1, gs=32):
    packed, scale, _ = make_w4a16_weights(E, N, K, gs, device=dev, seed=seed)
    A = (torch.randn(M, K, device=dev) * 0.1).to(dtype)
    tids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, packed, scale, tids, tw

def mxfp4_inputs(M, E, hidden, ispp, top_k, dev, seed=1, gs=32):
    d = make_mxfp4_moe_weights(E, hidden, ispp, group_size=gs, device=dev, seed=seed)
    A = (torch.randn(M, hidden, device=dev) * 0.1).to(dtype)
    tids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
    tw = torch.rand(M, top_k, device=dev, dtype=torch.float32)
    return A, d["w13"], d["w13_scale"], d["w2"], d["w2_scale"], tids, tw

# ─── §1  moe_align_block_size ──────────────────────────────────────────────
print("=== §1 moe_align_block_size workspace ===")
E, BS = 8, 64
tids = torch.randint(0, E, (8, 4), device=DEV, dtype=torch.int32)  # M=8, top_k=4
s_a, e_a, n_a = xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False)
ws = MoeAlignWorkspace.allocate(8, 4, E, BS, device=DEV)
s_w, e_w, n_w = xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
check("sorted_ids workspace==alloc", torch.equal(s_w, s_a))
check("expert_ids workspace==alloc", torch.equal(e_w, e_a))
check("num_post workspace==alloc", torch.equal(n_w, n_a))
check("sorted_ids wrote into ws.sorted_ids", torch.equal(s_w, ws.sorted_ids))
check("expert_ids wrote into ws.expert_ids", torch.equal(e_w, ws.expert_ids))
# address stability
s_w2, _, _ = xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
check("buffer address stable", s_w2.data_ptr() == s_w.data_ptr() == ws.sorted_ids.data_ptr())
# re-init correctness: counters must reset between calls -> two calls agree
s_w3, e_w3, n_w3 = xkernels.moe_align_block_size(tids, BS, E, backend="triton", truncate=False, workspace=ws)
check("counters re-init (3rd call == 1st)", torch.equal(s_w3, s_w) and torch.equal(n_w3, n_w))
# smaller-M reuse
tids4 = torch.randint(0, E, (4, 4), device=DEV, dtype=torch.int32)
ws_big = MoeAlignWorkspace.allocate(16, 4, E, BS, device=DEV)
s4_a, e4_a, n4_a = xkernels.moe_align_block_size(tids4, BS, E, backend="triton", truncate=False)
s4_w, e4_w, n4_w = xkernels.moe_align_block_size(tids4, BS, E, backend="triton", truncate=False, workspace=ws_big)
check("smaller-M reuse matches alloc", torch.equal(s4_w, s4_a) and torch.equal(n4_w, n4_a))
# stale data: the pad region must be pad_id, not stale
max_pad4 = 4*4 + (E+1)*(BS-1)
check("sorted_ids[:max_pad] valid (pad fill re-applied)",
      ws_big.sorted_ids[:max_pad4].tolist() == s4_a.tolist())

# ─── §2  fused_moe_int4_w4a16 ──────────────────────────────────────────────
# NOTE: only fused_combine=False is tested here. fused_combine=True relies on
# the RESOLVED-config launch path for soundness (see the SOUNDNESS GUARD in
# _moe_int4_w4a16_triton): with config=None (no tuned config for M=8 on GB10)
# @triton.autotune accumulates atomic-adds across candidate configs, breaking
# the ALLOC path equally (~480x too big). A serving stack reuses a cached config
# where the workspace IS active+correct; verifying that needs a populated config
# cache (follow-up, not GB10-testable).
print("=== §2 fused_moe_int4_w4a16 workspace (fused_combine=False, the sound path) ===")
for combine_mode in [False]:
    tag = f"fused_combine={combine_mode}"
    A, packed, scale, tids, tw = int4_inputs(8, E, 256, 256, 4, DEV, seed=2)
    N = packed.shape[1]
    o_alloc = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=combine_mode, backend="triton")
    ws4 = MoeInt4Workspace.allocate(8, 4, N, dtype=dtype, device=DEV)
    o_ws = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=combine_mode, backend="triton", workspace=ws4)
    check(f"{tag}: workspace==alloc", torch.allclose(o_ws.float(), o_alloc.float(), atol=1e-3, rtol=1e-3))
    # run twice -> atomic-add combine must re-zero (no 2x accumulate)
    o_ws2 = fused_moe_int4_w4a16(A, packed, scale, tids, tw, fused_combine=combine_mode, backend="triton", workspace=ws4)
    check(f"{tag}: re-zero on reuse (2nd call == 1st)", torch.allclose(o_ws.float(), o_ws2.float(), atol=1e-3, rtol=1e-3))
    # smaller-M reuse
    A2, packed2, scale2, tids2, tw2 = int4_inputs(4, E, 256, 256, 4, DEV, seed=3)
    o_small_alloc = fused_moe_int4_w4a16(A2, packed2, scale2, tids2, tw2, fused_combine=combine_mode, backend="triton")
    ws_big4 = MoeInt4Workspace.allocate(16, 4, N, dtype=dtype, device=DEV)
    o_small_ws = fused_moe_int4_w4a16(A2, packed2, scale2, tids2, tw2, fused_combine=combine_mode, backend="triton", workspace=ws_big4)
    check(f"{tag}: smaller-M reuse matches alloc", torch.allclose(o_small_ws.float(), o_small_alloc.float(), atol=1e-3, rtol=1e-3))

# ─── §3  fused_moe_mxfp4 ───────────────────────────────────────────────────
print("=== §3 fused_moe_mxfp4 workspace ===")
A, w13, w13s, w2, w2s, tids, tw = mxfp4_inputs(8, E, 256, 128, 4, DEV, seed=2)
hidden, ispp = 256, 128
o_alloc = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton")
# need block_m for the workspace -- resolve from the same config the kernel uses
from xkernels.ops.moe.triton.moe_mxfp4_kernel import get_default_config
block_m = get_default_config(8)["BLOCK_SIZE_M"]
wsm = MoeMxfp4Workspace.allocate(8, 4, E, block_m, ispp, hidden, dtype=dtype, device=DEV)
o_ws = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton", workspace=wsm)
check("workspace==alloc", torch.allclose(o_ws.float(), o_alloc.float(), atol=1e-2, rtol=1e-2))
# re-zero: down-stage combine is atomic-add -> must re-zero
o_ws2 = fused_moe_mxfp4(A, w13, w13s, w2, w2s, tids, tw, backend="triton", workspace=wsm)
check("re-zero on reuse (2nd == 1st)", torch.allclose(o_ws.float(), o_ws2.float(), atol=1e-2, rtol=1e-2))

# ─── §4  CUDA graph capture (align — the cleanest) ─────────────────────────
print("=== §4 CUDA graph capture (moe_align) ===")
tids_g = torch.randint(0, E, (4, 4), device=DEV, dtype=torch.int32)
ws_g = MoeAlignWorkspace.allocate(4, 4, E, BS, device=DEV)
for _ in range(3):
    xkernels.moe_align_block_size(tids_g, BS, E, backend="triton", truncate=False, workspace=ws_g)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    s_g, e_g, n_g = xkernels.moe_align_block_size(tids_g, BS, E, backend="triton", truncate=False, workspace=ws_g)
tids_g.add_(1)  # mutate inputs
tids_g.clamp_(max=E-1)
g.replay()
s_ref, e_ref, n_ref = xkernels.moe_align_block_size(tids_g, BS, E, backend="triton", truncate=False)
check("graph replay matches fresh-alloc on new inputs",
      torch.equal(s_g, s_ref) and torch.equal(e_g, e_ref))
check("graph output in workspace buffer", s_g.data_ptr() == ws_g.sorted_ids.data_ptr())

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
