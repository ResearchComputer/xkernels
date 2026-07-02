# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""DSL gate for the N-D-index gather + paged-KV-gather (issue #71 building block).

RoPE (#68) was the data-addressing family's *first* showcase — a 1-D index gather
(``cache[positions]``). This file pins the *second* showcase: a **2-D index**
gather (``pool[page_table]``), which is the unpage primitive behind paged
attention (issue #71's building block) and the one that exercised the N-D gather
generalization across the builder / torch oracle / Triton codegen.

What this test pins:

  * the N-D gather is bit-exact with raw torch advanced indexing (the oracle
    claim) across index ranks — a 1-D index (RoPE shape) and a 2-D index (paged
    shape) both lower to the exact same tensor;
  * ``paged_kv_gather`` (issue #71's DSL-expressible slice) is bit-exact with
    ``kv_cache[page_table]`` across dtypes / shapes, and its reference card
    passes ``verify`` on CPU (``max_abs == 0`` — the sharpest gate, since the op
    has zero arithmetic, so any drift is a codegen bug, never rounding);
  * ``make_inputs`` emits a VALID int ``page_table`` (indices in
    ``[0, num_pages)``) — the A4 scope line: the gather index is data, but a
    well-formed one;
  * on a GPU, the generated multi-dim addressing kernel is bit-exact with the
    oracle and the reference↔triton parity gate passes (the docs/brainstorm/06
    A4 loop closed on real hardware, now for the 2-D-index case).
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
from xkernels.vkl.examples import paged_kv_gather
from xkernels.vkl.ir.math import TensorRef
from xkernels.vkl.lower.mathbody import build_body, eval_torch

_GPU_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ═══════════════════════════════════════════════════════════════════════════════
# §1  The N-D gather primitive — bit-exact with raw torch (the oracle claim)
# ═══════════════════════════════════════════════════════════════════════════════


def _gather_ir(base_shape, idx_shape, axis=0):
    """Build a bare gather IR (base[<int idx>] along `axis`) and its inputs."""
    in_decls = {
        "base": TensorRef("base", "fp32", tuple(base_shape)),
        "idx": TensorRef("idx", "int32", tuple(idx_shape)),
    }
    # expected out shape: index's full shape replaces the gathered axis (in-place).
    out_shape = tuple(base_shape[:axis]) + tuple(idx_shape) + tuple(base_shape[axis + 1 :])
    out_decls = {"out": TensorRef("out", "fp32", out_shape)}

    def body(ctx):
        ctx.store("out", ctx.gather(ctx.load("base"), ctx.load("idx"), axis=axis))

    return build_body(body, in_decls, out_decls), out_shape


@pytest.mark.parametrize(
    "base_shape, idx_shape, axis",
    [
        ((8, 4), (3,), 0),          # 1-D index — the RoPE case (regression guard)
        ((8, 2, 4, 16), (3, 5), 0),  # 2-D index — the paged-KV case (NEW)
        ((16,), (4, 3), 0),          # gather into a 1-D base by a 2-D index
        ((2, 8, 5), (3,), 1),        # gather along a non-leading axis (1-D)
    ],
)
def test_nd_gather_bit_exact_with_torch(base_shape, idx_shape, axis):
    """The N-D gather lowering is bit-exact with torch advanced indexing."""
    body, _ = _gather_ir(base_shape, idx_shape, axis=axis)
    torch.manual_seed(0)
    n_pages = base_shape[axis]
    ins = {
        "base": torch.randn(*base_shape),
        "idx": torch.randint(0, n_pages, idx_shape, dtype=torch.int32),
    }
    out = eval_torch(body, ins, out_dtype="fp32")
    out = out["out"] if isinstance(out, dict) else out[0]
    if axis == 0:
        ref = ins["base"][ins["idx"].long()]
    else:
        ref = ins["base"].index_select(axis, ins["idx"].long())
    assert out.shape == ref.shape, f"{out.shape} != {ref.shape}"
    assert torch.equal(out, ref)


# ═══════════════════════════════════════════════════════════════════════════════
# §2  paged_kv_gather (issue #71 building block) — the full op
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("dt", ["bf16", "fp16", "fp32"])
def test_paged_gather_bit_exact_with_torch(dt):
    """The op is bit-exact with kv_cache[page_table] across dtypes."""
    spec = spec_of(paged_kv_gather)
    point = {"dtype": dt, "num_pages": 16, "page_size": 2, "num_kv_heads": 4,
             "head_dim": 64, "num_seqs": 3, "max_num_pages": 5}
    ins = make_inputs(spec, point, seed=0, device="cpu")
    out = run_reference(spec, ins)
    out = out[0] if isinstance(out, tuple) else out
    ref = ins["kv_cache"][ins["page_table"].long()]
    assert out.shape == ref.shape, f"{out.shape} != {ref.shape}"
    assert torch.equal(out, ref), f"{dt}: gather drifted from torch oracle"
    assert out.dtype == _DTYPES[dt]


def test_paged_gather_bit_exact_across_shapes():
    """The gather is bit-exact across page_size / heads / seq counts."""
    spec = spec_of(paged_kv_gather)
    for ps, nh, hd, ns, mp in [(1, 8, 128, 1, 7), (4, 4, 64, 4, 5), (2, 16, 256, 6, 11)]:
        point = {"dtype": "bf16", "num_pages": 32, "page_size": ps, "num_kv_heads": nh,
                 "head_dim": hd, "num_seqs": ns, "max_num_pages": mp}
        ins = make_inputs(spec, point, seed=1, device="cpu")
        out = run_reference(spec, ins)
        out = out[0] if isinstance(out, tuple) else out
        ref = ins["kv_cache"][ins["page_table"].long()]
        assert torch.equal(out, ref), f"shape {point}: drifted"


def test_paged_gather_page_table_is_valid_index():
    """A4 scope line: the gather index (page_table) is a VALID int index."""
    spec = spec_of(paged_kv_gather)
    point = {"dtype": "bf16", "num_pages": 16, "page_size": 2, "num_kv_heads": 4,
             "head_dim": 64, "num_seqs": 3, "max_num_pages": 5}
    ins = make_inputs(spec, point, seed=2, device="cpu")
    assert ins["page_table"].dtype == torch.int32
    assert ins["page_table"].min().item() >= 0
    assert ins["page_table"].max().item() < ins["kv_cache"].shape[0]


def test_paged_gather_schema_valid():
    """The emitted spec + cards are schema-valid (the contract-layer gate)."""
    spec = spec_of(paged_kv_gather)
    validate_op_spec(emit_spec(spec))
    validate_impl_card(emit_reference_card(spec))
    validate_impl_card(emit_card(spec, spec.targets["triton"]))


def test_paged_gather_verify_reference_card_passes_on_cpu():
    v = verify("paged_kv_gather.reference@1.0.0", arch="any")
    assert v["compiled"] is True, v["artifacts"].get("error")
    assert v["correctness"]["passed"] is True
    assert v["correctness"]["n_points"] >= 3
    # pure gather: zero arithmetic -> the gate is bit-exact (max_abs == 0).
    assert v["correctness"]["max_abs_err"] == 0.0
    assert v["determinism_check"] is True


def test_paged_gather_find_impl_surfaces_under_gather():
    res = find_impl(
        "gather",
        {"kv_cache": {"dtype": "bf16", "shape": [16, 2, 4, 64]},
         "page_table": {"dtype": "int32", "shape": [3, 5]}},
        target_arch="amd_cdna3",
    )
    ids = {r["impl_card_id"] for r in res}
    assert "paged_kv_gather.reference@1.0.0" in ids
    assert "paged_kv_gather.triton@1.0.0" in ids
    assert all(r["applicable"] for r in res if "paged_kv_gather" in r["impl_card_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# §3  GPU gate — the N-D gather codegen on real hardware
# ═══════════════════════════════════════════════════════════════════════════════


@_SKIP
@pytest.mark.parametrize("dt", ["bf16", "fp16"])
def test_paged_gather_triton_card_matches_reference_on_gpu(dt):
    """The generated N-D-index gather kernel is bit-exact with the oracle."""
    spec = spec_of(paged_kv_gather)
    register_dsl(spec, backend="triton")
    point = {"dtype": dt, "num_pages": 16, "page_size": 2, "num_kv_heads": 8,
             "head_dim": 128, "num_seqs": 3, "max_num_pages": 5}
    v = verify("paged_kv_gather.triton@1.0.0", arch="nvidia_sm121", shapes=[point])
    assert v["compiled"], v["artifacts"].get("error")
    assert v["correctness"]["passed"], v["correctness"]
    # pure gather: bit-exact (any drift is a codegen bug, never rounding).
    assert v["correctness"]["max_abs_err"] == 0.0
    assert v["determinism_check"] is True


@_SKIP
def test_paged_gather_parity_reference_vs_triton_on_gpu():
    """The reference + triton backends agree (the cross-backend gate)."""
    spec = spec_of(paged_kv_gather)
    register_dsl(spec, backend="triton")
    p = verify_parity("paged_kv_gather@1.0.0", archs=["nvidia_sm121"])
    assert not p["inconclusive"], p
    assert p["agree"], p
