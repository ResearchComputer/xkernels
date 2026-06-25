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
        0, num_experts, (int(point["M"]), int(point["top_k"])),
        generator=g, device=device, dtype=torch.int32,
    )
    return {
        "topk_ids": topk_ids,
        "block_size": int(point["block_size"]),
        "num_experts": num_experts,
        "truncate": bool(point.get("truncate", True)),
    }


_GENERATORS: dict[str, Callable[[dict, int, str], dict[str, Any]]] = {
    "fused_ffn@1.0.0": _ffn,
    "dual_rmsnorm@1.0.0": _dual_rmsnorm,
    "moe_sum_reduce@1.0.0": _moe_sum_reduce,
    "mha_merge_state@1.0.0": _mha_merge_state,
    "moe_align_block_size@1.0.0": _moe_align_block_size,
}


def has_generator(op_id: str) -> bool:
    return op_id in _GENERATORS


def generate_inputs(op_id: str, point: dict[str, Any], seed: int, device: str) -> dict[str, Any]:
    if op_id not in _GENERATORS:
        raise KeyError(
            f"no input generator registered for {op_id!r}; "
            f"add one in xkernels.registry.input_gen. Have {sorted(_GENERATORS)}"
        )
    return _GENERATORS[op_id](point, seed, device)


def supported_op_ids() -> list[str]:
    """Op ids for which a seeded generator exists (i.e. verify is wired)."""
    return sorted(_GENERATORS)
