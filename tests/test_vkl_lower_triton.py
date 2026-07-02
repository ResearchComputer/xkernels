# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 1.5 GPU gate: the DSL-generated Triton kernel runs on a real GPU.

Closes the docs/brainstorm/04 Ex.1 loop. One ``@kernel`` source (``examples/
dual_rmsnorm.py``) lowers to a generated ``@triton.jit`` kernel via
``lower_to_triton``. We assert three things:

  1. The generated kernel COMPILES + RUNS on H100 (sm_90).
  2. Its outputs match the torch auto-reference within the op's tolerance
     (``Numerics.by_dtype``: fp32 1e-5/1e-6; bf16 1.6e-2/1e-2).
  3. End-to-end: registering the DSL kernel + running ``verify`` on the
     DSL-emitted card PASSES the substrate's own correctness gate (against the
     SAME auto-reference the card declared) — zero JSON hand-editing.

Skipped when no CUDA device is available (the local CPU box). The CPU-side
contract round-trip + auto-reference equivalence are covered by
``test_vkl_roundtrip`` / ``test_vkl_reference``.
"""
from __future__ import annotations

import json
import pathlib

import pytest

pytest.importorskip("triton")

import torch  # noqa: E402

from xkernels.ops.norm.reference import dual_rmsnorm_ref  # noqa: E402
from xkernels.vkl import (  # noqa: E402
    lower_to_triton,
    make_inputs,
    register_dsl,
    run_reference,
    spec_of,
)
from xkernels.vkl.examples import dual_rmsnorm  # noqa: E402

_GPU_OK = torch.cuda.is_available()
_SKIP = pytest.mark.skipif(not _GPU_OK, reason="no CUDA device")
_DEV = "cuda" if _GPU_OK else "cpu"

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent


def _atol_rtol(dtype: str) -> tuple[float, float]:
    """Pull the op's per-dtype tolerance from the emitted numerics block."""
    spec = spec_of(dual_rmsnorm)
    by = spec.numerics.by_dtype[dtype]
    return float(by["atol"]), float(by["rtol"])


@_SKIP
def test_dsl_triton_matches_reference():
    """The DSL-generated Triton kernel matches the auto-reference on H100."""
    spec = spec_of(dual_rmsnorm)
    launch = lower_to_triton(spec)
    sweep = json.loads((_REPO / "registry/shape_sweeps/dual_rmsnorm.sweep.json").read_text())
    for p in sweep["points"]:
        dt = torch.bfloat16 if p["dtype"] == "bf16" else torch.float32
        x1 = torch.randn(p["T"], p["d1"], dtype=dt, device=_DEV)
        w1 = torch.randn(p["d1"], dtype=dt, device=_DEV)
        x2 = torch.randn(p["T"], p["d2"], dtype=dt, device=_DEV)
        w2 = torch.randn(p["d2"], dtype=dt, device=_DEV)
        o1, o2 = launch(x1, w1, x2, w2)
        r1, r2 = dual_rmsnorm_ref(x1, w1, x2, w2, eps=1e-6)
        atol, rtol = _atol_rtol(p["dtype"])
        torch.testing.assert_close(o1, r1, atol=atol, rtol=rtol)
        torch.testing.assert_close(o2, r2, atol=atol, rtol=rtol)


@_SKIP
def test_dsl_triton_matches_run_reference():
    """The DSL kernel also matches the trace-built auto-reference (``run_reference``).

    This is the stronger statement: the same trace IR lowered two ways (torch
    evaluator + Triton codegen) agrees on the GPU, which is the structural
    guarantee docs/brainstorm/02 §1 promises (one computation, two lowerings).
    """
    spec = spec_of(dual_rmsnorm)
    launch = lower_to_triton(spec)
    for dtype, T, d1, d2 in [("bf16", 256, 1536, 512), ("fp32", 128, 128, 128)]:
        p = {"dtype": dtype, "T": T, "d1": d1, "d2": d2}
        inputs = make_inputs(spec, p, seed=7)
        inputs = {k: v.to(_DEV) for k, v in inputs.items()}
        o1, o2 = launch(**inputs)
        r1, r2 = run_reference(spec, inputs)
        atol, rtol = _atol_rtol(dtype)
        torch.testing.assert_close(o1, r1.to(_DEV), atol=atol, rtol=rtol)
        torch.testing.assert_close(o2, r2.to(_DEV), atol=atol, rtol=rtol)


@_SKIP
def test_register_then_verify_passes():
    """Registering the DSL kernel + ``verify`` on the emitted card passes correctness.

    This is the full Phase 1.5 loop: the DSL lowers a body to Triton, registers
    it under the op's Triton backend, and the unchanged substrate ``verify``
    runs the sweep and reports PASS — the same gate every hand-written card
    passes, now driven by one ``@kernel`` source.
    """
    from xkernels.verify import verify

    spec = spec_of(dual_rmsnorm)
    register_dsl(spec, backend="triton")
    card_id = "dual_rmsnorm.triton@1.0.0"
    res = verify(card_id, arch="nvidia_sm90")
    assert res["compiled"], f"kernel did not compile: {res.get('artifacts', {}).get('error')}"
    assert res["correctness"]["passed"], (
        f"correctness failed: {res['correctness']}"
    )
