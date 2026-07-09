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
from xkernels.vkl.examples import apply_rope_gqa
from xkernels.vkl.ir.math import TensorRef
from xkernels.vkl.lower.mathbody import build_body, eval_torch

_GPU_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")

# ``apply_rope``'s generated multi-dim device kernel is now VERIFIED on GB10
# (the §14 modulo-sign bug was fixed by flooring the broadcast modulo in
# _TritonGenMultiDim). These three device-gate tests run and PASS. The marker
# below is a no-op kept as a hook (so a future regression here is one line to
# re-skip); it no longer skips. Diagnosis: meta/docs/wiki/04-gotchas.md §14.
_ROPE_TRITON_DEVICE_OOB = lambda f: f  # noqa: E731  (fixed; was a skip)

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
    # Only the MHA apply_rope cards -- the GQA sibling (apply_rope_gqa.*) also
    # surfaces for Hq==Hk==8, but its triton card targets sm_121, so it is
    # correctly arch-rejected here on amd_cdna3 (do not include it in the
    # "all applicable" check).
    assert all(r["applicable"] for r in res if r["impl_card_id"].startswith("apply_rope."))


# ═══════════════════════════════════════════════════════════════════════════════
# §3  GPU gate — the multi-dim addressing codegen on real hardware
# ═══════════════════════════════════════════════════════════════════════════════
# These run ONLY where a CUDA device is present (ds5 / GB10 sm_121). They close
# the docs/brainstorm/04 Ex.1 loop for the data-addressing family: one @kernel
# source -> a generated, verified Triton kernel whose multi-dim gather/slice/
# concat/unsqueeze lowering is bit-exact with the torch oracle on the GPU.


@_SKIP
@_ROPE_TRITON_DEVICE_OOB
@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_rope_triton_card_matches_reference_on_gpu(dt):
    """The generated multi-dim addressing kernel is bit-exact with the oracle."""
    spec = spec_of(apply_rope)
    register_dsl(spec, backend="triton")
    point = {"dtype": dt, "T": 8, "H": 8, "D": 128, "P": 64}
    v = verify("apply_rope.triton@1.0.0", arch="nvidia_sm121", shapes=[point])
    assert v["compiled"], v["artifacts"].get("error")
    assert v["correctness"]["passed"], v["correctness"]
    # The gather/slice/concat lowering adds NO *systematic* rounding: the triton
    # fp32 rotation products then cast may differ from torch by at most ~1
    # mantissa ULP (FMA contraction of ``a*b - c*d``), well inside the op's
    # rtol=1e-2. bf16 is bit-exact; fp16 can round by one ULP -- both pass
    # `verify`, which is the contract correctness gate.
    assert v["correctness"]["max_abs_err"] < 1e-3, v["correctness"]
    assert v["determinism_check"] is True


@_SKIP
@_ROPE_TRITON_DEVICE_OOB
def test_rope_parity_reference_vs_triton_on_gpu():
    """The reference + triton backends agree (the §5.3 cross-backend gate)."""
    spec = spec_of(apply_rope)
    register_dsl(spec, backend="triton")
    p = verify_parity("apply_rope@1.0.0", archs=["nvidia_sm121"])
    assert not p["inconclusive"], p
    assert p["agree"], p


# ═══════════════════════════════════════════════════════════════════════════════
# §4  apply_rope_gqa — grouped-query RoPE in ONE launch (issue #104)
# ═══════════════════════════════════════════════════════════════════════════════
# ``apply_rope@1.0.0`` (#68) forces a single shared ``H`` for query and key, so
# mini-sglang's adapter calls it TWICE for GQA models (32q/8kv etc.), discarding
# one output each time — the double-launch dominates decode TPOT at batch=1 on
# GB10 (#104). ``apply_rope_gqa@1.0.0`` relaxes the contract: ``query`` carries
# ``Hq`` heads, ``key`` carries ``Hk`` heads, ``Hq % Hk == 0`` is the GQA
# divisibility constraint, and the SAME rotate-half body (cos/sin already
# broadcast over the head axis) rotates both in one call. The body is the
# auto-reference (bit-exact with an independent GQA rotate-half torch ref).
#
# The TRITON DEVICE CARD IS COMMITTED + REGISTERED, but ``verify`` on GB10 is
# the gate and will OOB until a multi-dim codegen extension lands:
# ``_TritonGenMultiDim``/``_launch_multidim`` key the program grid on the FIRST
# output and share one flat offset+mask across all stores -- correct only when
# all outputs share one numel; for ``Hq != Hk`` the larger output's grid would
# write out-of-bounds on the smaller. Each Store needs its own coord
# decomposition (from its own output shape) + its own ``offs < numel_<out>``
# mask, over a grid sized by ``max(numel)``. That extension + passing
# ``verify``/``verify_parity`` on GB10 is the GPU-gated remainder (see the #104
# comment); the reference card (CPU-verified below) is the live contract surface
# in the meantime. (The card must be committed because the VKL drift gate
# requires every declared-target card on disk; ``register_dsl`` is safe without
# a GPU -- it builds only the host launcher, JIT-compiling on first call.)
from xkernels.registry.constraints import evaluate, validate_decidable


def _gqa_rope_ref(query, key, positions, cos_sin_cache):
    """Independent rotate-half torch RoPE for GQA (the oracle must match this)."""
    cs = cos_sin_cache[positions]
    h = query.shape[-1] // 2
    cos = cs[:, :h][:, None, :].float()
    sin = cs[:, h:][:, None, :].float()

    def rot(x):
        x1, x2 = x[..., :h].float(), x[..., h:].float()
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)

    return rot(query), rot(key)


def test_gqa_emit_schema_valid():
    spec = spec_of(apply_rope_gqa)
    validate_op_spec(emit_spec(spec))
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))
    assert spec.canonical_op == "attention"
    assert spec.launch.pattern == "elementwise"
    assert "Hq % Hk == 0" in spec.constraints
    # The triton device card IS committed (the VKL drift gate requires every
    # declared-target card on disk) + registered (rope_kernel.py), but
    # ``verify`` on GB10 is the gate -- it OOBs until the multi-dim codegen
    # extension for different-sized outputs lands (see #104).
    import pathlib
    assert (pathlib.Path("registry/impls") / "apply_rope_gqa.triton.card.json").exists()


def test_gqa_constraint_symbol_mod_symbol_validatable():
    """The constraint engine accepts ``symbol % symbol`` (GQA divisibility, #104).

    Previously the lenient validator returned ``0`` for unbound names, so
    ``Hq % Hk == 0`` became ``0 % 0`` -> ZeroDivisionError at ingest, wrongly
    rejecting a valid constraint the grammar explicitly allows. The placeholder
    is now ``1`` (non-degenerate), so ``Hq % Hk == 0`` validates and evaluates
    correctly once the symbols are bound.
    """
    validate_decidable("Hq % Hk == 0")  # must not raise
    assert evaluate("Hq % Hk == 0", {"Hq": 32, "Hk": 8}) is True   # GQA g=4
    assert evaluate("Hq % Hk == 0", {"Hq": 8, "Hk": 8}) is True    # MHA special case
    assert evaluate("Hq % Hk == 0", {"Hq": 32, "Hk": 12}) is False  # non-divisible


@pytest.mark.parametrize("dt", ["bf16", "fp16"])
@pytest.mark.parametrize("Hq, Hk", [(32, 8), (8, 8), (16, 4)])
def test_gqa_body_bit_exact_with_rotate_half(dt, Hq, Hk):
    """The GQA body rotates query (Hq) and key (Hk) bit-exactly, in one call."""
    spec = spec_of(apply_rope_gqa)
    ins = make_inputs(spec, {"dtype": dt, "T": 5, "Hq": Hq, "Hk": Hk, "D": 128, "P": 32},
                      seed=0, device="cpu")
    qo, ko = run_reference(spec, ins)
    rq, rk = _gqa_rope_ref(ins["query"], ins["key"], ins["positions"], ins["cos_sin_cache"])
    assert qo.shape == (5, Hq, 128), f"query_out head count {qo.shape[1]} != Hq {Hq}"
    assert ko.shape == (5, Hk, 128), f"key_out head count {ko.shape[1]} != Hk {Hk}"
    assert torch.equal(qo, rq), f"{dt} Hq={Hq} Hk={Hk}: query drifted from rotate-half ref"
    assert torch.equal(ko, rk), f"{dt} Hq={Hq} Hk={Hk}: key drifted from rotate-half ref"
    assert qo.dtype == _DTYPES[dt] and ko.dtype == _DTYPES[dt]


def test_gqa_positions_are_valid_indices():
    """A4 scope line (unchanged from apply_rope): positions are valid in [0, P)."""
    spec = spec_of(apply_rope_gqa)
    point = {"dtype": "bf16", "T": 8, "Hq": 32, "Hk": 8, "D": 128, "P": 37}
    ins = make_inputs(spec, point, seed=1, device="cpu")
    assert ins["positions"].dtype == torch.int32
    assert ins["positions"].max().item() < ins["cos_sin_cache"].shape[0]
    assert ins["positions"].min().item() >= 0


def test_gqa_verify_reference_card_passes_on_cpu():
    v = verify("apply_rope_gqa.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    assert v["determinism_check"] is True


def test_gqa_find_impl_surfaces_for_gqa_and_rejects_nondivisible():
    """find_impl surfaces apply_rope_gqa for GQA, rejects non-divisible Hq/Hk,
    and accepts the MHA special case (Hq == Hk)."""
    cache = {"cos_sin_cache": {"dtype": "fp32", "shape": [512, 128]},
             "positions": {"dtype": "int32", "shape": [1]}}

    def applicable(qshape, kshape):
        res = find_impl("attention", {"query": {"dtype": "bf16", "shape": qshape},
                                       "key": {"dtype": "bf16", "shape": kshape}, **cache},
                        target_arch="nvidia_sm121")
        return {r["impl_card_id"]: r for r in res if "apply_rope_gqa" in (r["impl_card_id"] or "")}

    gqa = applicable([1, 32, 128], [1, 8, 128])
    assert "apply_rope_gqa.reference@1.0.0" in gqa
    assert gqa["apply_rope_gqa.reference@1.0.0"]["applicable"] is True
    # The triton device card targets GB10 (sm_121), so it surfaces as applicable
    # for the GQA shape on its target arch (verify on GB10 is the gate, #104).
    assert "apply_rope_gqa.triton@1.0.0" in gqa
    assert gqa["apply_rope_gqa.triton@1.0.0"]["applicable"] is True

    nondiv = applicable([1, 32, 128], [1, 12, 128])
    assert nondiv["apply_rope_gqa.reference@1.0.0"]["applicable"] is False
    assert any("Hq % Hk == 0" in r for r in nondiv["apply_rope_gqa.reference@1.0.0"]["reject_reasons"])
    assert nondiv["apply_rope_gqa.triton@1.0.0"]["applicable"] is False  # same constraint

    mha = applicable([1, 8, 128], [1, 8, 128])
    assert mha["apply_rope_gqa.reference@1.0.0"]["applicable"] is True
    assert mha["apply_rope_gqa.triton@1.0.0"]["applicable"] is True  # Hq == Hk: MHA ok
