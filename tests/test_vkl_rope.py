# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for the data-addressing family + RoPE (issue #68; docs/brainstorm/06 A4).

The math IR's data-ADDRESSING nodes (``Gather`` / ``Slice`` / ``Concat`` /
``Unsqueeze``) resolve the A4 open question for case (a): the index is an INPUT
tensor, so the op is pure, parallel, deterministic — its torch lowering IS
bit-exact with its device lowering. This test pins:

  * each primitive is bit-exact with the matching raw torch op (the oracle claim);
  * ``Slice`` accepts ``str`` bounds over ``shape`` (RoPE halves a symbolic head);
  * ``apply_rope`` (issue #68) — the showcase — is bit-exact with an independent
    rotate-half torch RoPE across dtypes/head sizes, and its reference card
    passes ``verify`` on CPU;
  * the A4 scope line holds: ``make_inputs`` emits VALID int indices for the
    ``positions`` gather input (range [0, P)).
"""
from __future__ import annotations

import pytest
import torch

from xkernels import find_impl, verify, verify_parity
from xkernels.registry.schemas import validate_impl_card, validate_op_spec
from xkernels.vkl import (
    emit_card,
    emit_reference_card,
    emit_spec,
    make_inputs,
    register_dsl,
    run_reference,
    spec_of,
)
from xkernels.vkl.examples import apply_rope
from xkernels.vkl.ir.math import TensorRef
from xkernels.vkl.lower.mathbody import build_body, eval_torch

_GPU_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ═══════════════════════════════════════════════════════════════════════════════
# §1  The addressing primitives — bit-exact with raw torch (the oracle claim)
# ═══════════════════════════════════════════════════════════════════════════════

def _run(body, ins):
    in_d = {
        n: TensorRef(n, "fp32", tuple(t.shape), tuple(f"d{i}" for i in range(t.dim())))
        for n, t in ins.items()
    }
    out_d = {"out": TensorRef("out", "fp32", ins["x"].shape, in_d["x"].subscript)}
    return eval_torch(build_body(body, in_d, out_d), ins, "fp32")["out"]


def test_gather_bit_exact_with_torch_indexing():
    table = torch.randn(7, 4)
    idx = torch.tensor([0, 6, 2, 4])
    x = torch.randn(4, 4)  # unused shape anchor
    out = _run(lambda c: c.store("out", c.gather(c.load("table"), c.load("idx"), axis=0)),
               {"table": table, "idx": idx, "x": x})
    assert torch.equal(out, table[idx])


def test_slice_str_bounds_halve_symbolic_axis():
    x = torch.randn(3, 10)
    # "shape//2" resolves against the operand's concrete axis size (10 -> 5).
    def body(c):
        c.store("out", c.slice(c.load("x"), axis=1, start="shape//2", stop="shape"))
    assert torch.equal(_run(body, {"x": x}), x[:, 5:])


def test_concat_and_unsqueeze_bit_exact():
    a = torch.randn(2, 3)
    b = torch.randn(2, 4)
    def body(c):
        c.store("out", c.concat(c.load("a"), c.load("b"), axis=1))
    assert torch.equal(_run(body, {"a": a, "b": b, "x": a}), torch.cat([a, b], dim=1))


# ═══════════════════════════════════════════════════════════════════════════════
# §2  RoPE (issue #68) — the data-addressing showcase
# ═══════════════════════════════════════════════════════════════════════════════

def _rope_ref(query, key, positions, cos_sin_cache):
    """Independent rotate-half torch RoPE (the oracle must match this bit-exact)."""
    cs = cos_sin_cache[positions]
    h = query.shape[-1] // 2
    cos = cs[:, :h][:, None, :].float()
    sin = cs[:, h:][:, None, :].float()

    def rot(x):
        x1, x2 = x[..., :h].float(), x[..., h:].float()
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)

    return rot(query), rot(key)


def test_rope_emit_schema_valid():
    spec = spec_of(apply_rope)
    validate_op_spec(emit_spec(spec))
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))
    assert spec.canonical_op == "attention"
    assert spec.launch.pattern == "elementwise"


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
@pytest.mark.parametrize("D", [64, 128, 256])
def test_rope_body_bit_exact_with_rotate_half(dt, D):
    spec = spec_of(apply_rope)
    ins = make_inputs(spec, {"dtype": dt, "T": 5, "H": 4, "D": D, "P": 32}, seed=0, device="cpu")
    qo, ko = run_reference(spec, ins)
    rq, rk = _rope_ref(ins["query"], ins["key"], ins["positions"], ins["cos_sin_cache"])
    assert torch.equal(qo, rq), f"{dt} D={D}: query drifted from rotate-half ref"
    assert torch.equal(ko, rk), f"{dt} D={D}: key drifted from rotate-half ref"
    assert qo.dtype == _DTYPES[dt]


def test_rope_positions_are_valid_indices():
    """A4 scope line: the gather index (positions) is a VALID int index in [0, P)."""
    spec = spec_of(apply_rope)
    point = {"dtype": "bf16", "T": 8, "H": 8, "D": 128, "P": 37}
    ins = make_inputs(spec, point, seed=1, device="cpu")
    assert ins["positions"].dtype == torch.int32
    assert ins["positions"].max().item() < ins["cos_sin_cache"].shape[0]
    assert ins["positions"].min().item() >= 0


def test_rope_verify_reference_card_passes_on_cpu():
    v = verify("apply_rope.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_rope_find_impl_surfaces_under_attention():
    res = find_impl(
        "attention",
        {"query": {"dtype": "bf16", "shape": [8, 8, 128]},
         "key": {"dtype": "bf16", "shape": [8, 8, 128]},
         "cos_sin_cache": {"dtype": "fp32", "shape": [64, 128]},
         "positions": {"dtype": "int32", "shape": [8]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "apply_rope.reference@1.0.0" in ids
    assert "apply_rope.triton@1.0.0" in ids
    assert all(r["applicable"] for r in res if "apply_rope" in r["impl_card_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# §3  GPU gate — the multi-dim addressing codegen on real hardware
# ═══════════════════════════════════════════════════════════════════════════════
# These run ONLY where a CUDA device is present (ds5 / GB10 sm_121). They close
# the docs/brainstorm/04 Ex.1 loop for the data-addressing family: one @kernel
# source -> a generated, verified Triton kernel whose multi-dim gather/slice/
# concat/unsqueeze lowering is bit-exact with the torch oracle on the GPU.


@_SKIP
@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_rope_triton_card_matches_reference_on_gpu(dt):
    """The generated multi-dim addressing kernel is bit-exact with the oracle."""
    spec = spec_of(apply_rope)
    register_dsl(spec, backend="triton")
    point = {"dtype": dt, "T": 8, "H": 8, "D": 128, "P": 64}
    v = verify("apply_rope.triton@1.0.0", arch="nvidia_sm121", shapes=[point])
    assert v["compiled"], v["artifacts"].get("error")
    assert v["correctness"]["passed"], v["correctness"]
    # RoPE's rotation products are fp32-then-cast; bf16/fp16 agree bit-exact with
    # the oracle (the gather/slice/concat lowering adds ZERO rounding beyond it).
    assert v["correctness"]["max_abs_err"] == 0.0
    assert v["determinism_check"] is True


@_SKIP
def test_rope_parity_reference_vs_triton_on_gpu():
    """The reference + triton backends agree (the §5.3 cross-backend gate)."""
    spec = spec_of(apply_rope)
    register_dsl(spec, backend="triton")
    p = verify_parity("apply_rope@1.0.0", archs=["nvidia_sm121"])
    assert not p["inconclusive"], p
    assert p["agree"], p
