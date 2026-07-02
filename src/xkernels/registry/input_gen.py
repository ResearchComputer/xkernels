"""Seeded input generators for the verification harness.

One generator per Op Spec id. Each takes a shape-sweep ``point`` (symbolic dims
+ dtype) and a ``seed`` and returns a kwargs dict splat-able into both the
backend-neutral reference and the backend callable (they share the signature
modulo ``backend=``). Generators are pinned and seeded — determinism rule §5.4.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .dtypes import to_torch_dtype


def _gen(device: str, dtype: torch.dtype, *shape: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(*shape, generator=g, device=device, dtype=dtype)


def _ffn(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    s = seed
    x = _gen(device, dt, point["M"], point["K"], seed=s)
    w_gate = _gen(device, dt, point["K"], point["N"], seed=s + 1)
    w_up = _gen(device, dt, point["K"], point["N"], seed=s + 2)
    w_down = _gen(device, dt, point["N"], point["K"], seed=s + 3)
    return {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}


def _dual_rmsnorm(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    x1 = _gen(device, dt, point["T"], point["d1"], seed=seed)
    w1 = _gen(device, dt, point["d1"], seed=seed + 1)
    x2 = _gen(device, dt, point["T"], point["d2"], seed=seed + 2)
    w2 = _gen(device, dt, point["d2"], seed=seed + 3)
    return {"x1": x1, "w1": w1, "x2": x2, "w2": w2, "eps": 1e-6}


def _moe_sum_reduce(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    y = _gen(device, dt, point["M"], point["top_k"], point["H"], seed=seed)
    w = _gen(device, torch.float32, point["M"], point["top_k"], seed=seed + 1)
    return {"y": y, "w": w, "routed_scaling_factor": 1.0}


def _mha_merge_state(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    out_a = _gen(device, dt, point["T"], point["H"], point["D"], seed=seed)
    out_b = _gen(device, dt, point["T"], point["H"], point["D"], seed=seed + 1)
    lse_a = _gen(device, torch.float32, point["T"], point["H"], seed=seed + 2).abs()
    lse_b = _gen(device, torch.float32, point["T"], point["H"], seed=seed + 3).abs()
    return {"out_a": out_a, "lse_a": lse_a, "out_b": out_b, "lse_b": lse_b}


def _moe_align_block_size(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    # Integer dispatch builder: routed expert ids are the only tensor input.
    # block_size / num_experts / truncate are semantic scalar params of the op
    # (not perf knobs), so they flow to BOTH reference and card via **inputs.
    g = torch.Generator(device=device).manual_seed(int(seed))
    num_experts = int(point["num_experts"])
    topk_ids = torch.randint(
        0,
        num_experts,
        (int(point["M"]), int(point["top_k"])),
        generator=g,
        device=device,
        dtype=torch.int32,
    )
    return {
        "topk_ids": topk_ids,
        "block_size": int(point["block_size"]),
        "num_experts": num_experts,
        "truncate": bool(point.get("truncate", True)),
    }


def _mm_fp8_blockscale(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    # fp8 block-scale operands are produced by the exact-dequant quant helpers,
    # so reference and every backend consume byte-identical fp8 inputs.
    from ..ops.gemm.reference import per_block_quant_fp8, per_token_group_quant_fp8

    block = int(point.get("block", 128))
    M, K, N = int(point["M"]), int(point["K"]), int(point["N"])
    out_dtype = to_torch_dtype(point["dtype"])
    a = _gen(device, torch.float32, M, K, seed=seed)
    b = _gen(device, torch.float32, N, K, seed=seed + 1)
    a_fp8, a_scales = per_token_group_quant_fp8(a, block=block)
    b_fp8, b_scales = per_block_quant_fp8(b, block=block)
    return {
        "a_fp8": a_fp8,
        "a_scales": a_scales,
        "b_fp8": b_fp8,
        "b_scales": b_scales,
        "block": block,
        "out_dtype": out_dtype,
    }


def _hc_prenorm_gemm(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    # n_splits=1: the per-split tensor equals the sum and is element-wise
    # comparable (the split-K>1 sum-invariant is validated in tests/).
    dt = to_torch_dtype(point["dtype"])
    T, K, N = int(point["T"]), int(point["K"]), int(point["N"])
    a = _gen(device, dt, T, K, seed=seed)
    fn = _gen(device, torch.float32, N, K, seed=seed + 1)
    return {"a": a, "fn": fn, "n_splits": 1}


def _moe_int4_w4a16(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    from ..ops.moe.w4a16 import make_w4a16_weights

    dt = to_torch_dtype(point["dtype"])
    M, K, N = int(point["M"]), int(point["K"]), int(point["N"])
    E, top_k = int(point["E"]), int(point["top_k"])
    group_size = int(point.get("group_size", 32))
    A = _gen(device, dt, M, K, seed=seed)
    # exact-inverse packed/scale generator -> both backends dequant identically
    packed, scale, _w = make_w4a16_weights(E, N, K, group_size, device=device, seed=seed + 1)
    g = torch.Generator(device=device).manual_seed(seed + 2)
    topk_ids = torch.randint(0, E, (M, top_k), generator=g, device=device, dtype=torch.int32)
    topk_w = torch.rand(M, top_k, generator=g, device=device, dtype=torch.float32)
    return {
        "A": A,
        "packed": packed,
        "scale": scale,
        "topk_ids": topk_ids,
        "topk_w": topk_w,
        "group_size": group_size,
        "mul_routed_weight": True,
    }


def _sparse_mla_attention(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    T, H, D = int(point["T"]), int(point["H"]), int(point["D"])
    Kv, topk = int(point["Kv"]), int(point["topk"])
    q = _gen(device, dt, T, H, D, seed=seed)
    kv = _gen(device, dt, Kv, D, seed=seed + 1)
    g = torch.Generator(device=device).manual_seed(seed + 2)
    # all-valid columns (no -1 padding / sink in the mandatory sweep)
    indices = torch.randint(0, Kv, (T, topk), generator=g, device=device, dtype=torch.int32)
    sm_scale = float(1.0 / (D**0.5))
    return {"q": q, "kv": kv, "indices": indices, "sm_scale": sm_scale}


def _mhc_pre(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    T, hc_mult, hidden = int(point["T"]), int(point["hc_mult"]), int(point["hidden"])
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    residual = _gen(device, dt, T, hc_mult, hidden, seed=seed)
    fn = _gen(device, torch.float32, hc_mult3, hc_mult * hidden, seed=seed + 1)
    hc_scale = _gen(device, torch.float32, 3, seed=seed + 2)
    hc_base = _gen(device, torch.float32, hc_mult3, seed=seed + 3)
    return {
        "residual": residual,
        "fn": fn,
        "hc_scale": hc_scale,
        "hc_base": hc_base,
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "sinkhorn_iters": int(point.get("sinkhorn_iters", 2)),
    }


def _temperature_softmax(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    dt = to_torch_dtype(point["dtype"])
    B, V = int(point["B"]), int(point["V"])
    logits = _gen(device, dt, B, V, seed=seed)
    g = torch.Generator(device=device).manual_seed(seed + 1)
    temperatures = torch.rand(B, generator=g, device=device, dtype=torch.float32) + 0.25
    return {"logits": logits, "temperatures": temperatures}


def _topk_softmax(point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    # Seeded gating logits over [M, E] experts (the MoE router input). Distinct
    # fp32 values -> distinct softmax probabilities -> unambiguous top-k selection
    # (the op's top-k is integer-exact; near-ties are measure-zero for real
    # logits, see registry/ops/topk_softmax.spec.json numerics.notes).
    dt = to_torch_dtype(point["dtype"])
    M, E = int(point["M"]), int(point["E"])
    gating = _gen(device, dt, M, E, seed=seed)
    return {
        "gating_output": gating,
        "topk": int(point["topk"]),
        "renormalize": bool(point["renormalize"]),
    }


_GENERATORS: dict[str, Callable[[dict, int, str], dict[str, Any]]] = {
    "fused_ffn@1.0.0": _ffn,
    "dual_rmsnorm@1.0.0": _dual_rmsnorm,
    "moe_sum_reduce@1.0.0": _moe_sum_reduce,
    "mha_merge_state@1.0.0": _mha_merge_state,
    "moe_align_block_size@1.0.0": _moe_align_block_size,
    "mm_fp8_blockscale@1.0.0": _mm_fp8_blockscale,
    "hc_prenorm_gemm@1.0.0": _hc_prenorm_gemm,
    "moe_int4_w4a16@1.0.0": _moe_int4_w4a16,
    "sparse_mla_attention@1.0.0": _sparse_mla_attention,
    "mhc_pre@1.0.0": _mhc_pre,
    "temperature_softmax@1.0.0": _temperature_softmax,
    "topk_softmax@1.0.0": _topk_softmax,
}


def has_generator(op_id: str) -> bool:
    return op_id in _GENERATORS


def register_input_gen(op_id: str, fn) -> None:
    """Register (or supersede) a seeded input generator for ``op_id``.

    Additive entry point so a generator can be wired WITHOUT editing the
    ``_GENERATORS`` dict literal above — e.g. the vkl DSL's ``register_dsl``
    delegates to the spec's ``shape_symbols`` generator (``vkl.reference.make_inputs``)
    so an emitted op is ``verify``-able with no hand-editing of this file.
    Supersedes any prior registration (same op re-emitted takes over).
    """
    _GENERATORS[op_id] = fn


def generate_inputs(op_id: str, point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    if op_id not in _GENERATORS:
        try:
            from ..vkl import examples as _vkl_examples  # noqa: F401  lazy DSL generator wiring
        except Exception:
            pass
    if op_id not in _GENERATORS:
        raise KeyError(
            f"no input generator registered for {op_id!r}; "
            f"add one in xkernels.registry.input_gen. Have {sorted(_GENERATORS)}"
        )
    return _GENERATORS[op_id](point, seed, device)


def supported_op_ids() -> list[str]:
    """Op ids for which a seeded generator exists (i.e. verify is wired)."""
    return sorted(_GENERATORS)
