import sys
sys.path.insert(0, "src")
import torch
import xkernels
from xkernels.ops.attention import (
    PagedAttentionWorkspace,
    PagedAttentionPrefillWorkspace,
    SparseMlaAttentionWorkspace,
)
from xkernels.registry.input_gen import generate_inputs

DEV = "cuda"
torch.manual_seed(0)
ok = True
def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# ─── §1  paged_attention (decode) workspace ────────────────────────────────
print("=== §1 paged_attention decode workspace ===")
shape = {"dtype": "bf16", "B": 8, "H_q": 32, "H_kv": 8, "D": 128,
         "block_size": 1, "max_seq_len": 64}
inp = generate_inputs("paged_attention@1.0.0", shape, seed=0, device=DEV)
o_alloc = xkernels.paged_attention(backend="triton", **inp)
ws = PagedAttentionWorkspace.allocate(8, 32, 128, device=DEV, dtype=torch.bfloat16)
o_ws = xkernels.paged_attention(backend="triton", workspace=ws, **inp)
check("workspace==alloc output", torch.equal(o_ws, o_alloc))
check("workspace wrote into ws.out[:B]", torch.equal(o_ws, ws.out[:8]))
# address stability (graph-capture enabler): same buffer across calls
o_ws2 = xkernels.paged_attention(backend="triton", workspace=ws, **inp)
check("buffer address stable across calls", o_ws2.data_ptr() == o_ws.data_ptr())
# smaller-M reuse: allocate B_max=16, run B=8
ws_big = PagedAttentionWorkspace.allocate(16, 32, 128, device=DEV, dtype=torch.bfloat16)
o_small = xkernels.paged_attention(backend="triton", workspace=ws_big, **inp)
check("smaller-M reuse (B_max=16, run B=8) matches alloc",
      torch.equal(o_small, o_alloc))
check("smaller-M returns [:8] slice", o_small.shape[0] == 8)
# stale data: B=16 then B=8 -- the [8:] region is stale but [:8] is correct
shape2 = {**shape, "B": 16}
inp2 = generate_inputs("paged_attention@1.0.0", shape2, seed=1, device=DEV)
xkernels.paged_attention(backend="triton", workspace=ws_big, **inp2)  # fill 16
o_after = xkernels.paged_attention(backend="triton", workspace=ws_big, **inp)  # B=8
check("no stale-data leak [:8] correct after B=16 fill",
      torch.equal(o_after, o_alloc))
# shape mismatch rejected
try:
    ws_bad = PagedAttentionWorkspace.allocate(4, 32, 128, device=DEV, dtype=torch.bfloat16)
    xkernels.paged_attention(backend="triton", workspace=ws_bad, **inp)  # B=8 > 4
    check("too-small buffer rejected", False)
except ValueError:
    check("too-small buffer rejected", True)

# ─── §2  paged_attention_prefill workspace ─────────────────────────────────
print("=== §2 paged_attention_prefill workspace ===")
pshape = {"dtype": "bf16", "num_seqs": 4, "max_seq_len_q": 64, "max_seq_len_k": 64,
          "H_q": 32, "H_kv": 8, "D": 128, "block_size": 1, "prefix_frac": 0.0}
pinp = generate_inputs("paged_attention_prefill@1.0.0", pshape, seed=0, device=DEV)
nt = pinp["q"].shape[0]
po_alloc = xkernels.paged_attention_prefill(backend="triton", **pinp)
pws = PagedAttentionPrefillWorkspace.allocate(nt, 32, 128, device=DEV, dtype=torch.bfloat16)
po_ws = xkernels.paged_attention_prefill(backend="triton", workspace=pws, **pinp)
check("workspace==alloc output", torch.equal(po_ws, po_alloc))
check("workspace wrote into pws.out[:nt]", torch.equal(po_ws, pws.out[:nt]))
# address stability
po_ws2 = xkernels.paged_attention_prefill(backend="triton", workspace=pws, **pinp)
check("buffer address stable", po_ws2.data_ptr() == po_ws.data_ptr())
# larger workspace, smaller run
pws_big = PagedAttentionPrefillWorkspace.allocate(nt * 4, 32, 128, device=DEV, dtype=torch.bfloat16)
po_small = xkernels.paged_attention_prefill(backend="triton", workspace=pws_big, **pinp)
check("larger-ws reuse matches alloc", torch.equal(po_small, po_alloc))

# ─── §3  sparse_mla_attention workspace (dataclass logic only) ─────────────
# NOTE: the sparse_mla Triton kernel has a PRE-EXISTING bug on GB10 (it passes
# the AMD-only `waves_per_eu` knob to a kernel that rejects it on NVIDIA), so
# the write-into-buffer path can't be exercised on ds5. The dataclass logic
# (allocate/matches/validation) is tested here; the write path is structurally
# identical to the verified paged_attention path above.
print("=== §3 sparse_mla_attention workspace (dataclass logic) ===")
mws = SparseMlaAttentionWorkspace.allocate(8, 4, 64, device=DEV, dtype=torch.bfloat16)
check("allocate gives correct shapes",
      mws.out.shape == (8,4,64) and mws.lse.shape == (8,4) and mws.maxl.shape == (8,4))
check("allocate gives correct dtypes",
      mws.out.dtype == torch.bfloat16 and mws.lse.dtype == torch.float32)
check("matches exact shape", mws.matches(8, 4, 64, device=torch.device(DEV), dtype=torch.bfloat16))
check("matches smaller T (reuse)", mws.matches(4, 4, 64, device=torch.device(DEV), dtype=torch.bfloat16))
check("rejects larger T", not mws.matches(16, 4, 64, device=torch.device(DEV), dtype=torch.bfloat16))
check("rejects wrong H", not mws.matches(8, 8, 64, device=torch.device(DEV), dtype=torch.bfloat16))

# ─── §4  graph capture end-to-end (decode) ─────────────────────────────────
print("=== §4 CUDA graph capture with workspace (decode) ===")
gshape = {"dtype": "bf16", "B": 4, "H_q": 32, "H_kv": 8, "D": 128,
          "block_size": 1, "max_seq_len": 128}
ginp = generate_inputs("paged_attention@1.0.0", gshape, seed=5, device=DEV)
gws = PagedAttentionWorkspace.allocate(4, 32, 128, device=DEV, dtype=torch.bfloat16)
# warmup
for _ in range(3):
    xkernels.paged_attention(backend="triton", workspace=gws, **ginp)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
# capture -- workspace keeps addresses stable
o_capture = xkernels.paged_attention(backend="triton", workspace=gws, **ginp)
with torch.cuda.graph(g):
    o_graph = xkernels.paged_attention(backend="triton", workspace=gws, **ginp)
# replay with NEW inputs (mutate q in place) -- graph reuses the captured addresses
ginp["q"].add_(0.05)
ginp["k_cache"].mul_(0.99)
g.replay()
o_replay_ref = xkernels.paged_attention(backend="triton", **ginp)  # fresh alloc, new inputs
check("graph replay matches fresh-alloc on new inputs",
      torch.allclose(o_graph.float(), o_replay_ref.float(), atol=0.1, rtol=0.01))
check("graph output in captured workspace buffer",
      o_graph.data_ptr() == gws.out.data_ptr())

print()
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
