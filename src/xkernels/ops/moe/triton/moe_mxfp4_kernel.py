# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Fast MXFP4 W4A16 grouped fused-MoE GEMM for DeepSeek-V4 on AMD MI300A
(gfx942, CDNA3).

DeepSeek-V4-Flash routed experts are MXFP4 (packed E2M1 nibbles + ``ue8m0``
block-32 scales). The only working AMD path today is tokenspeed's
correctness-first ``Mxfp4DequantBackend`` — a Python per-expert loop that
dequantizes each active expert to bf16 and runs ``torch.matmul``. This kernel is
the performant, packed-weight replacement: an MFMA grouped GEMM that unpacks the
E2M1 nibbles and applies the ``ue8m0`` scale *inline* in the K loop (no full bf16
dequant — a full dequant of all 256 V4 experts is ~138 GB/rank and OOMs the APU).

Two grouped GEMMs are fused into the op:

* **gate_up**: ``[M*top_k, 2*ispp] = A_gathered @ w13[e]^T`` (contracted dim =
  ``hidden``), with the V4 clamped-SwiGLU epilogue
  ``silu(clamp(gate, max=L)) * clamp(up, -L, L)`` (optional per-expert bias
  ``b13`` added pre-activation) producing ``act [M*top_k, ispp]``.
* **down**: ``out = act @ w2[e]^T`` (contracted dim = ``ispp``), with the optional
  per-expert bias ``b2`` and the routed-weighted top-k combine
  (atomic-accumulate into ``[M, hidden]``).

Both share one ``@triton.jit`` body (:func:`_mxfp4_moe_gemm_kernel`) parameterized
by a ``STAGE`` constexpr that selects the epilogue.

MXFP4 layout / decode (matches ``xkernels.ops.gather.mxfp4``)
------------------------------------------------------------
* Packed ``B``: ``[E, N, K // 2]`` uint8, two E2M1 nibbles per byte; low nibble =
  even (lower-K) element, high nibble = odd.
* E2M1 magnitude LUT ``{0, .5, 1, 1.5, 2, 3, 4, 6}``; bit 3 (``0x8``) is sign.
* Scale ``S``: ``[E, N, K // group]`` uint8 ue8m0; multiplier ``2**(byte - 127)``
  shared across ``group`` (32) consecutive K. ``0xFF`` is the NaN code -> 0.

Why Triton (portable), tiling, and CDNA3 knobs mirror the INT4 W4A16 kernel
(``moe_int4_kernel.py``); see that file for the rationale. The unpack here is 2
nibbles/byte through a LUT + exponent scale, vs 8 nibbles/int32 + offset there.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...._backends import Backend
from ...._dispatch import register
from ..w4a16 import moe_align_block_size_ep, moe_align_block_size_ref
from .mxfp4_configs import get_autotune_configs, get_default_config, prune_configs

__all__ = ["mxfp4_moe_gemm", "fused_mxfp4_moe_kernel"]

# E2M1 magnitudes for the 8 unsigned codes (index = nibble & 0x7); bit 3 = sign.
# Declared as a length-8 constexpr tuple consumed by ``tl.where`` chains so the
# decode is branchless and lives entirely in registers.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@triton.jit
def _e2m1_decode(nib):
    """Decode a tile of 4-bit E2M1 nibbles (0..15) to fp32 signed values.

    ``nib`` is an int tile; returns the same-shape fp32 tile. The magnitude is a
    branchless ``tl.where`` ladder over the low 3 bits; bit 3 is the sign.
    """
    code = nib & 0x7
    mag = tl.where(
        code == 0, 0.0,
        tl.where(
            code == 1, 0.5,
            tl.where(
                code == 2, 1.0,
                tl.where(
                    code == 3, 1.5,
                    tl.where(
                        code == 4, 2.0,
                        tl.where(code == 5, 3.0, tl.where(code == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((nib & 0x8) != 0, -mag, mag)


@triton.jit
def _mxfp4_moe_gemm_kernel(
    a_ptr,  # [M_a, K] activations (gate_up: A [M, hidden]; down: act [M*top_k, ispp])
    b_ptr,  # [E, N, K // 2] uint8 packed E2M1 weights
    c_ptr,  # output: gate_up -> [M*top_k, ispp]; down -> [M, hidden] (combine)
    bscale_ptr,  # [E, N, K // group] uint8 ue8m0 scales
    bias_ptr,  # [E, N] bias (gate_up: b13 [E, 2*ispp]; down: b2 [E, hidden]) or null
    topk_weights_ptr,  # [num_valid_tokens] fp32 routing weights (sorted order)
    sorted_token_ids_ptr,  # [EM] int32 token-slot ids grouped by expert
    expert_ids_ptr,  # [num_m_blocks] int32 expert per M-block (-1 = filtered)
    num_tokens_post_padded_ptr,  # [1] int32
    N,  # B-tensor row count (gate_up: 2*ispp; down: hidden) — bias / gate-up split
    N_OUT,  # tiled output width (gate_up: ispp; down: hidden) — grid + masking
    K,  # contracted dim (gate_up: hidden; down: ispp)
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,  # stride over the packed-K (uint8) dim of B
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,  # stride over the group dim of the scale tensor
    group_k: tl.constexpr,
    top_k: tl.constexpr,
    swiglu_limit,  # fp32 SwiGLU clamp limit (gate_up stage only)
    STAGE: tl.constexpr,  # 0 = gate_up + SwiGLU; 1 = down + bias + combine
    HAS_BIAS: tl.constexpr,
    HAS_LIMIT: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    compute_type: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # for STAGE 0 this tiles ISPP (the act width)
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    FILTER_EXPERT: tl.constexpr,
    # AMD/CDNA3 lowering knobs (see moe_int4_kernel); no effect on the math.
    waves_per_eu: tl.constexpr = 0,
    matrix_instr_nonkdim: tl.constexpr = 16,
    kpack: tl.constexpr = 2,
):
    """Grouped per-expert MXFP4 GEMM with inline E2M1+ue8m0 dequant.

    STAGE 0 (gate_up): one program computes a ``[BLOCK_M, BLOCK_N]`` tile of the
    SwiGLU activation ``act[:, n] = silu(clamp(gate, L)) * clamp(up, L)`` where
    ``gate = A @ w13[:, n]^T`` and ``up = A @ w13[:, n + ispp]^T`` — i.e. it runs
    two GEMMs (the gate half and the up half of ``w13``) into the same N-tile and
    fuses the activation. Output width is ``ISPP = N // 2``.

    STAGE 1 (down): standard grouped GEMM ``act @ w2[e]^T`` with optional bias and
    the atomic routed-weighted top-k combine into the ``[M, hidden]`` output.
    """
    PACK: tl.constexpr = 2  # nibbles per uint8
    BLOCK_K_PACK: tl.constexpr = BLOCK_SIZE_K // PACK

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_OUT, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if FILTER_EXPERT and off_experts == -1:
        # Filtered (EP) block: in STAGE 0 zero the act scratch row; in STAGE 1 the
        # [M, hidden] combine buffer is pre-zeroed, so just exit.
        if STAGE == 0:
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N_OUT)
            tl.store(
                c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask
            )
        return

    # ---- A pointers ----
    # gate_up: gather token rows from A [M, hidden] via offs_token // top_k.
    # down:    act scratch is already token-indexed [M*top_k, ispp]; row = offs_token.
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if STAGE == 0:
        a_row = offs_token // top_k
    else:
        a_row = offs_token
    a_ptrs = a_ptr + (a_row[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_kp = tl.arange(0, BLOCK_K_PACK)
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // group_k
    offs_g_base = tl.arange(0, GROUPS_PER_BLOCK)
    # Nibble shifts for the 2 packed values per byte: [0, 4]. Applied to the
    # broadcast byte to extract the low (even-K) and high (odd-K) nibble.
    shifts = (tl.arange(0, PACK) * 4).to(tl.int32)  # [PACK]

    if STAGE == 0:
        # gate_up: tile N over ISPP. The gate half is rows [pid_n*BN, ...]; the up
        # half is the same column offset shifted by ISPP. Run both GEMMs in this
        # loop and fuse SwiGLU at the end.
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        ispp_dyn = N // 2
        offs_ng = offs_n % ispp_dyn  # gate rows
        offs_nu = offs_ng + ispp_dyn  # up rows
        bg_ptrs = (
            b_ptr + off_experts * stride_be
            + offs_kp[:, None] * stride_bk + offs_ng[None, :] * stride_bn
        )
        bu_ptrs = (
            b_ptr + off_experts * stride_be
            + offs_kp[:, None] * stride_bk + offs_nu[None, :] * stride_bn
        )
        sg_base = bscale_ptr + off_experts * stride_bse + offs_ng[None, :] * stride_bsn
        su_base = bscale_ptr + off_experts * stride_bse + offs_nu[None, :] * stride_bsn
        acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    else:
        offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
        b_ptrs = (
            b_ptr + off_experts * stride_be
            + offs_kp[:, None] * stride_bk + offs_n[None, :] * stride_bn
        )
        s_base = bscale_ptr + off_experts * stride_bse + offs_n[None, :] * stride_bsn
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    for kb in range(0, num_k_blocks):
        k0 = kb * BLOCK_SIZE_K
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k0), other=0.0
            )
        kp0 = k0 // PACK
        g0 = k0 // group_k
        offs_g = g0 + offs_g_base

        if STAGE == 0:
            # ---- gate half ----
            if EVEN_K:
                bg = tl.load(bg_ptrs)
                bu = tl.load(bu_ptrs)
            else:
                kpm = (kp0 + offs_kp) < (K // PACK)
                bg = tl.load(bg_ptrs, mask=kpm[:, None], other=0)
                bu = tl.load(bu_ptrs, mask=kpm[:, None], other=0)
            bg = bg.to(tl.int32)
            bu = bu.to(tl.int32)
            # Unpack 2 nibbles per byte. Broadcast the byte over the PACK axis with
            # shifts [0, 4] so the PACK dim sits between KP and N: [KP, 2, N] ->
            # reshape [K, N] gives logical k = kp*2 + nibble (low nibble = even K),
            # matching the MXFP4 packing (mirrors the INT4 kernel's unpack order).
            nibg = (bg[:, None, :] >> shifts[None, :, None]) & 0xF  # [KP, PACK, N]
            nibu = (bu[:, None, :] >> shifts[None, :, None]) & 0xF
            bvg = tl.reshape(_e2m1_decode(nibg), (BLOCK_SIZE_K, BLOCK_SIZE_N))
            bvu = tl.reshape(_e2m1_decode(nibu), (BLOCK_SIZE_K, BLOCK_SIZE_N))
            # ue8m0 scales, one fetch per group, broadcast across group_k.
            sg = tl.load(
                sg_base + offs_g[:, None] * stride_bsk,
                mask=(offs_g[:, None] < tl.cdiv(K, group_k)), other=0,
            ).to(tl.int32)
            su = tl.load(
                su_base + offs_g[:, None] * stride_bsk,
                mask=(offs_g[:, None] < tl.cdiv(K, group_k)), other=0,
            ).to(tl.int32)
            mg = tl.where(sg == 0xFF, 0.0, tl.exp2((sg - 127).to(tl.float32)))
            mu = tl.where(su == 0xFF, 0.0, tl.exp2((su - 127).to(tl.float32)))
            mg = tl.reshape(
                tl.broadcast_to(mg[:, None, :], (GROUPS_PER_BLOCK, group_k, BLOCK_SIZE_N)),
                (BLOCK_SIZE_K, BLOCK_SIZE_N),
            )
            mu = tl.reshape(
                tl.broadcast_to(mu[:, None, :], (GROUPS_PER_BLOCK, group_k, BLOCK_SIZE_N)),
                (BLOCK_SIZE_K, BLOCK_SIZE_N),
            )
            acc_g = tl.dot(a, (bvg * mg).to(a.dtype), acc=acc_g)
            acc_u = tl.dot(a, (bvu * mu).to(a.dtype), acc=acc_u)
            bg_ptrs += BLOCK_K_PACK * stride_bk
            bu_ptrs += BLOCK_K_PACK * stride_bk
        else:
            if EVEN_K:
                bpk = tl.load(b_ptrs)
            else:
                kpm = (kp0 + offs_kp) < (K // PACK)
                bpk = tl.load(b_ptrs, mask=kpm[:, None], other=0)
            bpk = bpk.to(tl.int32)
            nib = (bpk[:, None, :] >> shifts[None, :, None]) & 0xF  # [KP, PACK, N]
            bv = tl.reshape(_e2m1_decode(nib), (BLOCK_SIZE_K, BLOCK_SIZE_N))
            s = tl.load(
                s_base + offs_g[:, None] * stride_bsk,
                mask=(offs_g[:, None] < tl.cdiv(K, group_k)), other=0,
            ).to(tl.int32)
            m = tl.where(s == 0xFF, 0.0, tl.exp2((s - 127).to(tl.float32)))
            m = tl.reshape(
                tl.broadcast_to(m[:, None, :], (GROUPS_PER_BLOCK, group_k, BLOCK_SIZE_N)),
                (BLOCK_SIZE_K, BLOCK_SIZE_N),
            )
            accumulator = tl.dot(a, (bv * m).to(a.dtype), acc=accumulator)
            b_ptrs += BLOCK_K_PACK * stride_bk

        a_ptrs += BLOCK_SIZE_K * stride_ak

    if STAGE == 0:
        # ---- clamped-SwiGLU epilogue ----
        gate = acc_g
        up = acc_u
        if HAS_BIAS:
            offs_ng2 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            # b13 is [E, 2*ispp] = [E, N]; gate slice [0:ispp], up slice [ispp:2*ispp].
            bg_bias = tl.load(
                bias_ptr + off_experts * N + (offs_ng2 % N_OUT),
                mask=(offs_ng2 < N_OUT), other=0.0,
            ).to(tl.float32)
            bu_bias = tl.load(
                bias_ptr + off_experts * N + (offs_ng2 % N_OUT) + N_OUT,
                mask=(offs_ng2 < N_OUT), other=0.0,
            ).to(tl.float32)
            gate = gate + bg_bias[None, :]
            up = up + bu_bias[None, :]
        if HAS_LIMIT:
            gate = tl.minimum(gate, swiglu_limit)
            up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
        act = (gate * tl.sigmoid(gate)) * up  # silu(gate) * up
        act = act.to(compute_type)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N_OUT)
        tl.store(c_ptrs, act, mask=c_mask)
    else:
        # ---- down epilogue: bias + routed combine ----
        # Kept in the STAGE-1 ``else`` (not after a STAGE-0 ``return``) so the
        # compiler sees ``accumulator`` in scope on every path it type-checks
        # (the constexpr STAGE dead-codes the other branch).
        if HAS_BIAS:
            offs_n2 = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            b2_bias = tl.load(
                bias_ptr + off_experts * N + offs_n2, mask=(offs_n2 < N), other=0.0
            ).to(tl.float32)
            accumulator = accumulator + b2_bias[None, :]
        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
            accumulator = accumulator * moe_weight[:, None]
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        out_rows = offs_token // top_k
        c_ptrs = c_ptr + stride_cm * out_rows[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


fused_mxfp4_moe_kernel = triton.autotune(
    configs=get_autotune_configs(),
    key=["N", "K", "EM", "num_valid_tokens", "STAGE"],
    prune_configs_by={"early_config_prune": prune_configs},
)(
    triton.heuristics({"EVEN_K": lambda a: a["K"] % a["BLOCK_SIZE_K"] == 0})(
        _mxfp4_moe_gemm_kernel
    )
)


def _launch(
    a, b, c, bscale, bias, topk_weights, sorted_ids, expert_ids, num_post,
    *, N, N_out, K, top_k, group_size, stage, has_bias, swiglu_limit, mul_routed_weight,
    compute_type, filter_expert, config,
):
    num_valid_tokens = topk_weights.numel()
    has_limit = stage == 0 and swiglu_limit is not None and swiglu_limit > 0
    limit_val = float(swiglu_limit) if has_limit else 0.0
    bias_ptr = bias if bias is not None else b  # dummy when HAS_BIAS=False
    common = (
        a, b, c, bscale, bias_ptr, topk_weights, sorted_ids, expert_ids, num_post,
        N, N_out, K, sorted_ids.shape[0], num_valid_tokens,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        bscale.stride(0), bscale.stride(1), bscale.stride(2),
    )
    common_kw = dict(
        group_k=group_size, top_k=top_k, swiglu_limit=limit_val,
        STAGE=stage, HAS_BIAS=has_bias, HAS_LIMIT=has_limit,
        MUL_ROUTED_WEIGHT=mul_routed_weight, compute_type=compute_type,
        FILTER_EXPERT=filter_expert,
    )
    if config is not None:
        bm = config["BLOCK_SIZE_M"]
        bn = config["BLOCK_SIZE_N"]
        grid = (triton.cdiv(sorted_ids.shape[0], bm) * triton.cdiv(N_out, bn),)
        _mxfp4_moe_gemm_kernel[grid](
            *common, **common_kw,
            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            EVEN_K=(K % config["BLOCK_SIZE_K"] == 0),
            waves_per_eu=config.get("waves_per_eu", 0),
            matrix_instr_nonkdim=config.get("matrix_instr_nonkdim", 16),
            kpack=config.get("kpack", 2),
            num_warps=config["num_warps"], num_stages=config["num_stages"],
        )
        return

    def grid(meta):
        return (
            triton.cdiv(sorted_ids.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(N_out, meta["BLOCK_SIZE_N"]),
        )

    fused_mxfp4_moe_kernel[grid](*common, **common_kw)


def mxfp4_moe_gemm(
    a: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    M: int,
    top_k: int,
    ispp: int,
    hidden: int,
    group_size: int,
    b13: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    swiglu_limit: float | None = 10.0,
    mul_routed_weight: bool = True,
    compute_type: tl.dtype = tl.bfloat16,
    filter_expert: bool = True,
    config: dict | None = None,
) -> torch.Tensor:
    """Launch the two fused MXFP4 grouped GEMMs (gate_up+SwiGLU, then down+combine).

    Returns the ``[M, hidden]`` fp32 combined output (caller casts).
    """
    assert w13.dtype == torch.uint8 and w2.dtype == torch.uint8
    assert sorted_token_ids.stride(0) == 1
    EM = sorted_token_ids.shape[0]
    act_dtype = tl.float32 if a.dtype == torch.float32 else tl.bfloat16
    act = torch.empty(EM, ispp, dtype=a.dtype, device=a.device)

    # STAGE 0: gate_up (K = hidden, N = 2*ispp) -> SwiGLU -> act [EM, ispp].
    _launch(
        a, w13, act, w13_scale, b13, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        N=2 * ispp, N_out=ispp, K=hidden, top_k=top_k, group_size=group_size, stage=0,
        has_bias=b13 is not None, swiglu_limit=swiglu_limit,
        mul_routed_weight=False, compute_type=act_dtype, filter_expert=filter_expert,
        config=config,
    )

    # STAGE 1: down (K = ispp, N = hidden), act is token-indexed [EM, ispp].
    out = torch.zeros((M, hidden), dtype=torch.float32, device=a.device)
    _launch(
        act, w2, out, w2_scale, b2, topk_weights, sorted_token_ids, expert_ids,
        num_tokens_post_padded,
        N=hidden, N_out=hidden, K=ispp, top_k=top_k, group_size=group_size, stage=1,
        has_bias=b2 is not None, swiglu_limit=None,
        mul_routed_weight=mul_routed_weight, compute_type=tl.float32,
        filter_expert=filter_expert, config=config,
    )
    return out


# --- xkernels backend registration -----------------------------------------


def _moe_mxfp4_triton(
    A: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    b13: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    swiglu_limit: float | None = 10.0,
    group_size: int = 32,
    mul_routed_weight: bool = True,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    M, top_k = topk_ids.shape
    E, two_ispp, _ = w13.shape
    ispp = two_ispp // 2
    hidden = A.shape[1]
    # Resolve a fixed launch config and take the DIRECT (non-autotune) path. The
    # down stage atomic-accumulates into the output, so running @triton.autotune
    # against the real buffer would add every benchmarked config's result. The
    # sort/pad block MUST equal the kernel BLOCK_SIZE_M (see align_block_m), so
    # derive it from the config.
    config = get_default_config(M)
    block_m = config["BLOCK_SIZE_M"]
    if expert_map is not None:
        sorted_ids, expert_ids, num_post = moe_align_block_size_ep(
            topk_ids, block_m, expert_map.numel(), expert_map
        )
        filter_expert = True
    else:
        sorted_ids, expert_ids, num_post = moe_align_block_size_ref(topk_ids, block_m, E)
        filter_expert = False
    out = mxfp4_moe_gemm(
        A, w13, w13_scale, w2, w2_scale,
        topk_w.reshape(-1).float(), sorted_ids, expert_ids, num_post,
        M=M, top_k=top_k, ispp=ispp, hidden=hidden, group_size=group_size,
        b13=b13, b2=b2, swiglu_limit=swiglu_limit,
        mul_routed_weight=mul_routed_weight, filter_expert=filter_expert,
        config=config,
    )
    return out.to(A.dtype)


register("moe_mxfp4", Backend.TRITON)(_moe_mxfp4_triton)
