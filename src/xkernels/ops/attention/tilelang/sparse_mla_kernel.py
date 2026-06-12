# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""TileLang sparse-MLA attention backend for AMD MI300A (gfx942), issue #32.

A split-KV flash-MLA kernel (adapted from TileLang's AMD-tuned
``examples/deepseek_mla/amd`` reference) that parallelizes the top-k reduction
across the GPU — the lever the one-program-per-``(token,head)`` Triton kernel
lacks (measured 1.8-6.2x faster at top-k 512-2048 on MI300A). Extends the
reference with the attention **sink** (folded into the combine) and an **lse**
output. Phase 1 covers the unmasked full-top-k case; the per-token length mask
for padded/variable top-k is Phase 2 (it needs TileLang's varlen layout
handling), so this backend is opt-in (not in the "auto" order) for now.

TileLang has no ROCm wheel — this backend self-registers only where the
from-source ROCm build is importable (a gfx942 serving image); elsewhere the
import fails quietly and ``"auto"`` falls through to the Triton/reference path.
The compute operates on pre-gathered latent KV (the gather is the same torch op
as the Triton decode path); ``q`` is split into nope (value-bearing, ``d_v``) and
rope (score-only) along the last axis.
"""

import functools
import importlib.util

import torch

from ...._backends import Backend
from ...._dispatch import register

__all__ = ["sparse_mla_attention_tilelang"]

_LOG2E = 1.44269504


@functools.lru_cache(maxsize=64)
def _build(Tn, H, topk, Kv, dim, d_v, pe_dim, block_N, block_H, num_split, threads, sm_scale):
    # ``dim`` = padded value/working dim (e.g. 512); ``d_v`` = actual nope width
    # gathered from the pool (e.g. 448); ``pe_dim`` = rope (64); pool row is
    # [nope(d_v) | rope(pe_dim)] of width ``pool_d``. Lazy import keeps tilelang
    # off the ``import xkernels`` critical path (conflicts with tokenspeed_triton).
    import tilelang
    import tilelang.language as T

    pool_d = d_v + pe_dim
    scale = float(sm_scale * _LOG2E)  # exp2 domain
    dtype = T.bfloat16
    acc = T.float32
    VBH = min(block_H, H)
    split_len = topk // num_split

    @T.prim_func
    def kernel(
        Q: T.Tensor([Tn, H, dim], dtype),
        Q_pe: T.Tensor([Tn, H, pe_dim], dtype),
        KVp: T.Tensor([Kv, pool_d], dtype),
        Idx: T.Tensor([Tn, topk], T.int32),
        glse: T.Tensor([Tn, H, num_split], dtype),
        Op: T.Tensor([Tn, H, num_split, dim], dtype),
        Output: T.Tensor([Tn, H, dim], dtype),
    ):
        # ---- split: per (token, head-tile, split) flash partial ----
        with T.Kernel(Tn, H // VBH, num_split, threads=threads) as (bx, by, bz):
            Q_l = T.alloc_fragment([block_H, dim], dtype)
            Qpe_l = T.alloc_fragment([block_H, pe_dim], dtype)
            KV_s = T.alloc_shared([block_N, dim], dtype)
            Kpe_s = T.alloc_shared([block_N, pe_dim], dtype)
            idx_s = T.alloc_shared([block_N], T.int32)
            acc_s = T.alloc_fragment([block_H, block_N], acc)
            acc_s_c = T.alloc_fragment([block_H, block_N], dtype)
            acc_o = T.alloc_fragment([block_H, dim], acc)
            m = T.alloc_fragment([block_H], acc)
            m_prev = T.alloc_fragment([block_H], acc)
            sscale = T.alloc_fragment([block_H], acc)
            ssum = T.alloc_fragment([block_H], acc)
            lsum = T.alloc_fragment([block_H], acc)

            T.use_swizzle(10)
            T.copy(Q[bx, by * VBH:(by + 1) * VBH, :], Q_l)
            T.copy(Q_pe[bx, by * VBH:(by + 1) * VBH, :], Qpe_l)
            T.fill(acc_o, 0)
            T.fill(lsum, 0)
            T.fill(m, -T.infinity(acc))

            for k in T.Pipelined(T.ceildiv(split_len, block_N), num_stages=0):
                kv0 = split_len * bz + k * block_N
                # In-kernel gather: pull each selected pool row by its top-k index
                # (zero-padding nope d_v->dim), avoiding a torch gather + pad cat.
                for j in T.Parallel(block_N):
                    idx_s[j] = Idx[bx, kv0 + j]
                T.clear(KV_s)
                for j, c in T.Parallel(block_N, d_v):
                    KV_s[j, c] = KVp[idx_s[j], c]
                for j, c in T.Parallel(block_N, pe_dim):
                    Kpe_s[j, c] = KVp[idx_s[j], d_v + c]
                T.clear(acc_s)
                T.gemm(Q_l, KV_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Qpe_l, Kpe_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(m, m_prev)
                T.fill(m, -T.infinity(acc))
                T.reduce_max(acc_s, m, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    m[i] = T.max(m[i], m_prev[i])
                for i in T.Parallel(block_H):
                    sscale[i] = T.exp2(m_prev[i] * scale - m[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - m[i] * scale)
                T.reduce_sum(acc_s, ssum, dim=1)
                T.copy(acc_s, acc_s_c)
                for i in T.Parallel(block_H):
                    lsum[i] = lsum[i] * sscale[i] + ssum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= sscale[i]
                T.gemm(acc_s_c, KV_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] = acc_o[i, j] / lsum[i]
            for i in T.Parallel(block_H):
                # log2-domain lse of this split (m is raw-score max; *scale folds sm_scale+log2e)
                lsum[i] = T.log2(lsum[i]) + m[i] * scale
            T.copy(lsum, glse[bx, by * VBH:(by + 1) * VBH, bz])
            T.copy(acc_o, Op[bx, by * VBH:(by + 1) * VBH, bz, :])

        # ---- combine: reduce the per-split partials (sink-LESS), write Output ----
        # Byte-for-byte the proven AMD reference combine. The attention **sink**
        # is applied entirely wrapper-side as an exact per-(token,head) rescale
        # (out *= sigmoid(lnZ_real - sink)); keeping the kernel unmodified avoids
        # the TileLang LayoutInference failure the in-kernel sink fold triggered.
        with T.Kernel(H, Tn, threads=128) as (by, bz):
            po = T.alloc_fragment([dim], dtype)
            oacc = T.alloc_fragment([dim], acc)
            lse_split = T.alloc_var(acc)
            llog = T.alloc_var(acc)
            lmax = T.alloc_var(acc)
            sc = T.alloc_var(acc)

            T.clear(llog)
            T.clear(oacc)
            lmax = -T.infinity(acc)
            for k in T.serial(num_split):
                lmax = T.max(lmax, glse[bz, by, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_split = glse[bz, by, k]
                llog += T.exp2(lse_split - lmax)
            llog = T.log2(llog) + lmax
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po[i] = Op[bz, by, k, i]
                lse_split = glse[bz, by, k]
                sc = T.exp2(lse_split - llog)
                for i in T.Parallel(dim):
                    oacc[i] += po[i] * sc
            for i in T.Parallel(dim):
                Output[bz, by, i] = oacc[i]

    # Output(6) is the only returned tensor; glse(4)/Op(5) are caller-allocated
    # scratch the kernel writes in place (glse read back for lse/sink rescale).
    return tilelang.compile(
        kernel, out_idx=[6], target="hip",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True},
    )


_BLOCK_N, _BLOCK_H, _THREADS = 32, 64, 128


def sparse_mla_attention_tilelang(
    q, kv, indices, *, sm_scale, topk_length=None, attn_sink=None, d_v=None,
):
    """TileLang sparse-MLA backend (Phase 1: unmasked full-top-k + sink + lse).

    Covers the case where every selected slot is valid: ``topk_length is None``,
    no ``-1`` sentinels, and ``topk`` divisible by ``block_N * num_split``. Other
    cases raise ``NotImplementedError`` (the caller picks another backend) until
    the per-token length mask lands; this backend is therefore opt-in (not in the
    "auto" order). ``d_v`` splits the latent into nope (value) + rope (score-only).
    """
    dev = q.device
    Tn, H, D = q.shape
    topk = indices.shape[1]
    d_v = D if d_v is None else d_v
    pe_dim = D - d_v
    if pe_dim <= 0:
        raise NotImplementedError("tilelang sparse-MLA needs a rope split (d_v < D)")
    if topk_length is not None:
        raise NotImplementedError("tilelang backend does not yet support topk_length")
    if bool((indices < 0).any()):
        raise NotImplementedError("tilelang backend does not yet support -1 padding")

    num_split = max(1, min(16, topk // 256))
    if topk % (_BLOCK_N * num_split) != 0:
        raise NotImplementedError(
            f"tilelang backend needs topk % {_BLOCK_N * num_split} == 0 (got {topk})"
        )

    # Pad the value/nope dim up to a multiple of 128 (e.g. V4's 448 -> 512): the
    # FullRow gemm fragment layout TileLang infers is valid at 512 but not 448.
    # The KV gather + zero-pad happens IN-KERNEL from the raw pool (no torch
    # gather / pad cat); only q's small nope is padded here.
    dim_p = ((d_v + 127) // 128) * 128
    kv_pool = kv.contiguous()                            # [Kv, D] = [nope(d_v) | rope]
    Kv = kv_pool.shape[0]
    idx = indices.to(torch.int32).contiguous()
    q_rope = q[:, :, d_v:].contiguous()
    q_nope = q[:, :, :d_v]
    if dim_p != d_v:
        q_nope = torch.cat([q_nope, q_nope.new_zeros(Tn, H, dim_p - d_v)], dim=2)
    q_nope = q_nope.contiguous()

    kern = _build(Tn, H, topk, Kv, dim_p, d_v, pe_dim, _BLOCK_N, _BLOCK_H, num_split,
                  _THREADS, float(sm_scale))
    glse = torch.empty(Tn, H, num_split, device=dev, dtype=torch.bfloat16)
    op = torch.empty(Tn, H, num_split, dim_p, device=dev, dtype=torch.bfloat16)
    out = kern(q_nope, q_rope, kv_pool, idx, glse, op)[:, :, :d_v].float()  # slice padding

    # glse[t,h,k] = log2(Z_k), Z_k = sum_j exp(sm_scale * q.k) over split k. The
    # kernel output divides by Z_real = sum_k Z_k (no sink). Apply the attention
    # sink exactly as a per-(token,head) rescale: out *= Z_real/(Z_real+exp(sink))
    # = sigmoid(lnZ_real - sink). lse/max_logits derived here too.
    g = glse.float()
    gmax = g.amax(dim=2)
    ln_z = (gmax + (g - gmax.unsqueeze(2)).exp2().sum(dim=2).clamp_min(1e-30).log2()) / _LOG2E
    if attn_sink is not None:
        sink = attn_sink.float().reshape(-1)[:H].unsqueeze(0)        # [1, H]
        out = out * torch.sigmoid(ln_z - sink).unsqueeze(-1)
        lse = ln_z + torch.nn.functional.softplus(sink - ln_z)       # ln(Z_real + e^sink)
        maxl = torch.maximum(ln_z, sink)
    else:
        lse = ln_z
        maxl = ln_z
    return out.to(q.dtype), lse, maxl


# Register only where TileLang is installed (the from-source ROCm build, i.e. a
# gfx942 serving image). ``find_spec`` checks availability without importing
# tilelang — so this stays off the ``import xkernels`` critical path.
if importlib.util.find_spec("tilelang") is not None:
    register("sparse_mla_attention", Backend.TILELANG)(sparse_mla_attention_tilelang)
