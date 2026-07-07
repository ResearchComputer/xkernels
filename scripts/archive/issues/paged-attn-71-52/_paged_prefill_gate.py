import sys
sys.path.insert(0, "src")
import torch
import torch.nn.functional as F
import xkernels
from xkernels import verify, verify_parity

# ─── §A: verify + parity on the mandatory sweep ────────────────────────────
print("=== reference card (CPU-style oracle) ===")
r = verify("paged_attention_prefill.reference@1.0.0", arch="cpu")
print("reference compiled=", r["compiled"], "passed=", r["correctness"]["passed"],
      "max_abs=", r["correctness"].get("max_abs"), "n=", r["correctness"].get("n"))

print("=== triton card (GB10/sm_121) ===")
t = verify("paged_attention_prefill.triton@1.0.0", arch="nvidia_sm121")
print("triton compiled=", t["compiled"], "passed=", t["correctness"]["passed"],
      "max_abs=", t["correctness"].get("max_abs"), "max_rel=",
      t["correctness"].get("max_rel"), "n=", t["correctness"].get("n"))

print("=== parity (reference vs triton, GPU) ===")
p = verify_parity("paged_attention_prefill@1.0.0", archs=["nvidia_sm121"])
print("agree=", p["agree"], "max_rel=", p.get("max_pairwise_rel_err"),
      "runnable=", p.get("per_backend_runnable"))

# direct ABS parity gap at a real shape (the rel-only metric is ill-conditioned
# on near-zero attention outputs; confirm the abs gap is bf16-quantization noise)
print("=== direct abs parity gap (is it near-zero noise?) ===")
from xkernels.registry.input_gen import generate_inputs as build_inputs
for dt_name in ["fp32", "bf16"]:
    pt = {"dtype": dt_name, "num_seqs": 4, "max_seq_len_q": 128, "max_seq_len_k": 128,
          "H_q": 32, "H_kv": 8, "D": 128, "block_size": 1, "prefix_frac": 0.0}
    inp = build_inputs("paged_attention_prefill@1.0.0", pt, seed=0, device="cuda")
    o = xkernels.paged_attention_prefill(backend="triton", **inp)
    r2 = xkernels.paged_attention_prefill(backend="reference", **inp)
    d = (o.float() - r2.float()).abs()
    print(f"  {dt_name}: max_abs={d.max().item():.3e}  "
          f"(|out|max={o.float().abs().max().item():.3e})")

# ─── §B: causal-math gold check -- pure prefill vs SDPA with is_causal ──────
print("=== §B causal gold check: paged prefill vs F.scaled_dot_product_attention(is_causal) ===")
def gold_check(B_seqs, L, Hq, Hkv, D, dt):
    """One sequence of length L, pure prefill (nk==nq==L). Compare the paged
    prefill op to torch SDPA with the causal flag -- the ground-truth causal
    attention. GQA via repeat_interleave."""
    g = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(1, L, Hq, D, generator=g, device="cuda", dtype=dt)  # [1,L,Hq,D]
    k = torch.randn(1, L, Hkv, D, generator=g, device="cuda", dtype=dt)
    v = torch.randn(1, L, Hkv, D, generator=g, device="cuda", dtype=dt)
    group = Hq // Hkv
    ke = k.repeat_interleave(group, dim=2)  # [1,L,Hq,D]
    ve = v.repeat_interleave(group, dim=2)
    # SDPA expects [B, H, L, D]
    gold = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), ke.permute(0, 2, 1, 3), ve.permute(0, 2, 1, 3),
        is_causal=True, scale=D ** -0.5,
    ).permute(0, 2, 1, 3)  # [1, L, Hq, D]
    # stage into paged cache (page_size=1), one seq
    q2 = q.squeeze(0)  # [L, Hq, D]
    kc = k.squeeze(0)[:, None]  # [L, 1, Hkv, D]  (page_size=1)
    vc = v.squeeze(0)[:, None]
    bt = torch.arange(L, device="cuda", dtype=torch.int32).reshape(1, L)
    cu_q = torch.tensor([0, L], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, L], dtype=torch.int32, device="cuda")
    out = xkernels.paged_attention_prefill(
        q2, kc, vc, bt, cu_q, cu_k, scale=D ** -0.5, backend="triton")
    ref = xkernels.paged_attention_prefill(
        q2, kc, vc, bt, cu_q, cu_k, scale=D ** -0.5, backend="reference")
    for name, o in [("triton", out), ("reference", ref)]:
        d = (o.float() - gold.float()).abs()
        rel = (d / gold.float().abs().clamp_min(1e-6)).max().item()
        print(f"  L={L} Hq={Hq} Hkv={Hkv} D={D} {str(dt):14s} {name:9s}: "
              f"max_abs={d.max().item():.3e} max_rel={rel:.3e}")

gold_check(1, 16, 32, 8, 128, torch.float32)
gold_check(1, 64, 32, 8, 128, torch.bfloat16)
gold_check(1, 64, 64, 8, 64, torch.bfloat16)   # Llama-70B shape
gold_check(1, 32, 8, 1, 128, torch.bfloat16)    # MQA

# ─── §C: extend (nk > nq) gold check ───────────────────────────────────────
print("=== §C extend gold check: prefix KV already cached, new tokens are a suffix ===")
def extend_check(prefix, nq, Hq, Hkv, D, dt):
    """nq new tokens appended after `prefix` already-cached tokens. The new
    tokens attend causally to [0, prefix+p+1). Build full seq of len prefix+nq,
    run SDPA(is_causal) on the FULL seq, then compare ONLY the last nq rows
    (the new tokens) to the paged-extend op."""
    nk = prefix + nq
    g = torch.Generator(device="cuda").manual_seed(7)
    qfull = torch.randn(1, nk, Hq, D, generator=g, device="cuda", dtype=dt)
    k = torch.randn(1, nk, Hkv, D, generator=g, device="cuda", dtype=dt)
    v = torch.randn(1, nk, Hkv, D, generator=g, device="cuda", dtype=dt)
    group = Hq // Hkv
    gold = F.scaled_dot_product_attention(
        qfull.permute(0,2,1,3), k.repeat_interleave(group,dim=2).permute(0,2,1,3),
        v.repeat_interleave(group,dim=2).permute(0,2,1,3), is_causal=True, scale=D**-0.5,
    ).permute(0,2,1,3)  # [1,nk,Hq,D]
    gold_new = gold[0, prefix:]  # [nq, Hq, D] -- only the new tokens
    # paged-extend: q is only the NEW nq tokens; kv cache holds all nk
    q2 = qfull[0, prefix:]  # [nq, Hq, D]
    kc = k[0][:, None]      # [nk,1,Hkv,D]
    vc = v[0][:, None]
    bt = torch.arange(nk, device="cuda", dtype=torch.int32).reshape(1, nk)
    cu_q = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, nk], dtype=torch.int32, device="cuda")
    out = xkernels.paged_attention_prefill(q2, kc, vc, bt, cu_q, cu_k,
                                           scale=D**-0.5, backend="triton")
    ref = xkernels.paged_attention_prefill(q2, kc, vc, bt, cu_q, cu_k,
                                           scale=D**-0.5, backend="reference")
    for name, o in [("triton", out), ("reference", ref)]:
        d = (o.float() - gold_new.float()).abs()
        rel = (d / gold_new.float().abs().clamp_min(1e-6)).max().item()
        print(f"  prefix={prefix} nq={nq} nk={nk} {str(dt):14s} {name:9s}: "
              f"max_abs={d.max().item():.3e} max_rel={rel:.3e}")

extend_check(48, 16, 32, 8, 128, torch.bfloat16)
extend_check(128, 32, 32, 8, 128, torch.float32)

# ─── §D: multi-seq packed (cu_seqlens partitions correctly) ────────────────
print("=== §D multi-seq packed batch ===")
def packed_check():
    g = torch.Generator(device="cuda").manual_seed(99)
    Ls = [37, 16, 64]   # 3 ragged seqs
    Hq, Hkv, D, dt = 32, 8, 128, torch.bfloat16
    num_tokens = sum(Ls)
    max_blocks = max(Ls)
    bt = torch.arange(3*max_blocks, device="cuda", dtype=torch.int32).reshape(3, max_blocks)
    cu_q = torch.tensor([0, Ls[0], Ls[0]+Ls[1], num_tokens], dtype=torch.int32, device="cuda")
    cu_k = cu_q.clone()
    q = torch.randn(num_tokens, Hq, D, generator=g, device="cuda", dtype=dt)
    kc = torch.randn(3*max_blocks, 1, Hkv, D, generator=g, device="cuda", dtype=dt)
    vc = torch.randn(3*max_blocks, 1, Hkv, D, generator=g, device="cuda", dtype=dt)
    out = xkernels.paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=D**-0.5, backend="triton")
    ref = xkernels.paged_attention_prefill(q, kc, vc, bt, cu_q, cu_k, scale=D**-0.5, backend="reference")
    d = (out.float()-ref.float()).abs()
    # per-seq gold via SDPA
    off = 0
    for si, L in enumerate(Ls):
        # gold must gather via the SAME paged layout the op uses
        pages = bt[si, :L].long()
        kcs = kc[pages].squeeze(1)  # [L, Hkv, D]
        vcs = vc[pages].squeeze(1)
        qs = q[off:off+L]
        gold = F.scaled_dot_product_attention(
            qs[None].permute(0,2,1,3), kcs.repeat_interleave(4,1)[None].permute(0,2,1,3),
            vcs.repeat_interleave(4,1)[None].permute(0,2,1,3), is_causal=True, scale=D**-0.5,
        ).permute(0,2,1,3)[0]
        dd = (out[off:off+L].float()-gold.float()).abs()
        print(f"  seq{si} L={L}: triton-vs-sdpa max_abs={dd.max().item():.3e}")
        off += L
    print(f"  triton-vs-reference overall max_abs={d.max().item():.3e}")
packed_check()
