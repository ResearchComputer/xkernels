# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 2.0a GPU gate: the DSL lowers a 2D-tiled GEMM (the go/no-go).

The Phase 1.5 row-reduce IR cannot express a GEMM (no 2D tiles, no K-loop, no
MMA). This test proves the **math-IR convergence** (docs/brainstorm/11 §11):
one ``@kernel`` source (``examples/gemm_bf16.py``) builds the doc-10 math IR
(``MMA``/``Pointwise``), which lowers to BOTH torch ``matmul`` (the bit-exact
auto-reference) AND a generated tiled ``@triton.jit`` K-loop kernel.

Three assertions (mirroring ``test_vkl_lower_triton.py``):

  1. The generated kernel COMPILES + RUNS on H100.
  2. Its output matches the torch auto-reference within the op's tolerance.
  3. End-to-end: ``verify`` on the DSL-emitted card PASSES against the
     auto-reference — the substrate gate, driven by one ``@kernel`` source.

Skipped when no CUDA device. CPU-side math-IR round-trip is covered separately.
"""

from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402

from xkernels.vkl import (  # noqa: E402
    lower_to_triton,
    make_inputs,
    run_reference,
    spec_of,
    trace_ir,
)
from xkernels.vkl.examples import gemm_bf16  # noqa: E402

_GPU_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")
_DEV = "cuda" if _GPU_OK else "cpu"

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent


def _atol_rtol(dtype: str) -> tuple[float, float]:
    spec = spec_of(gemm_bf16)
    by = spec.numerics.by_dtype[dtype]
    return float(by["atol"]), float(by["rtol"])


def test_mathbody_torch_reference_bit_exact():
    """The math-IR torch evaluator is bit-exact with a.float() @ b.float().

    No GPU needed; this is the reference-equivalence gate (the structural promise
    that the body IS the reference). Runs everywhere so a CPU box still exercises
    the math-IR build + eval path.
    """
    spec = spec_of(gemm_bf16)
    body = trace_ir(spec)
    assert [type(n).__name__ for n in body.ir.nodes] == [
        "Load",
        "Load",
        "MMA",
        "Pointwise",
        "Store",
    ], "gemm_bf16 should lower to exactly [Load,Load,MMA,Pointwise,Store]"
    for dtype, M, N, K in [
        ("bf16", 128, 128, 128),
        ("fp32", 128, 256, 256),
        ("bf16", 512, 512, 512),
    ]:
        dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        a = (torch.rand(M, K, dtype=torch.float32) * 2 - 1).to(dt)
        b = (torch.rand(K, N, dtype=torch.float32) * 2 - 1).to(dt)
        (out,) = run_reference(spec, {"a": a, "b": b})
        ref = (a.float() @ b.float()).to(dt)
        assert torch.equal(out, ref), f"{dtype} M={M} N={N} K={K}: math-IR ref drifted"


@_SKIP
def test_dsl_gemm_matches_reference():
    """The generated tiled Triton GEMM matches the auto-reference on H100."""
    spec = spec_of(gemm_bf16)
    launch = lower_to_triton(spec)
    sweep = json.loads((_REPO / "registry/shape_sweeps/gemm_bf16.sweep.json").read_text())
    for p in sweep["points"]:
        dt = torch.bfloat16 if p["dtype"] == "bf16" else torch.float32
        a = (torch.rand(p["M"], p["K"], dtype=torch.float32) * 2 - 1).to(dt).to(_DEV)
        b = (torch.rand(p["K"], p["N"], dtype=torch.float32) * 2 - 1).to(dt).to(_DEV)
        (out,) = launch(a, b)
        # the reference is the EXACT oracle (run_reference disables TF32 so a
        # GPU matmul is true fp32 — a true-fp32 kernel must match it bit-for-bit,
        # not just within the loose fp32 tolerance). Inline ``a.float() @ b.float()``
        # on GPU would silently be TF32 and diverge by the TF32 rounding gap.
        (ref,) = run_reference(spec, {"a": a, "b": b})
        atol, rtol = _atol_rtol(p["dtype"])
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@_SKIP
def test_dsl_gemm_matches_run_reference():
    """The two-lowerings guarantee: same math IR, torch eval vs Triton codegen."""
    spec = spec_of(gemm_bf16)
    launch = lower_to_triton(spec)
    for dtype, M, N, K in [("bf16", 256, 256, 256), ("fp32", 128, 128, 128)]:
        p = {"dtype": dtype, "M": M, "N": N, "K": K}
        inputs = make_inputs(spec, p, seed=11)
        inputs = {k: v.to(_DEV) for k, v in inputs.items()}
        (out,) = launch(**inputs)
        (ref,) = run_reference(spec, inputs)
        atol, rtol = _atol_rtol(dtype)
        torch.testing.assert_close(out, ref.to(_DEV), atol=atol, rtol=rtol)


@_SKIP
def test_register_then_verify_passes():
    """Registering the DSL GEMM + ``verify`` on the emitted card passes correctness."""
    from xkernels.verify import verify
    from xkernels.vkl import register_dsl

    spec = spec_of(gemm_bf16)
    register_dsl(spec, backend="triton")
    res = verify("gemm_bf16.triton@1.0.0", arch="nvidia_sm90")
    assert res["compiled"], f"kernel did not compile: {res.get('artifacts', {}).get('error')}"
    assert res["correctness"]["passed"], f"correctness failed: {res['correctness']}"
