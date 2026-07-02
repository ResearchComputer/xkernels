# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 1 gate B: the auto-reference matches the hand-written one, bit-exact.

This is the structural guarantee (docs/brainstorm/05 §4) made into a CI gate:
the ``@kernel`` body — written *independently*, kernel-flavored
(``sum(x*x)/d + eps``) — must match the existing hand-written reference
(``pow(2).mean``) bit-for-bit on the FULL sweep. If this ever breaks, the body
is no longer a faithful oracle (the whole point of "the body IS the reference").

We assert ``torch.equal`` (bit-exact), not just within tolerance — the two
formulations are arithmetically identical, so any drift is a bug, not noise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xkernels.ops.norm.reference import dual_rmsnorm_ref
from xkernels.vkl import make_inputs, run_reference, spec_of
from xkernels.vkl.examples import dual_rmsnorm

_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spec():
    return spec_of(dual_rmsnorm)


@pytest.fixture(scope="module")
def sweep_points():
    sweep = json.loads((_REPO / "registry/shape_sweeps/dual_rmsnorm.sweep.json").read_text())
    return sweep["points"]


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_auto_ref_bit_exact_hand_ref(spec, sweep_points, seed):
    """For every sweep point + seed, auto-ref == hand-ref, bit-exact."""
    for point in sweep_points:
        inputs = make_inputs(spec, point, seed=seed)
        # Both references see IDENTICAL inputs by construction (same seed/generator).
        auto_out = run_reference(spec, inputs)
        hand_out = dual_rmsnorm_ref(
            inputs["x1"], inputs["w1"], inputs["x2"], inputs["w2"], eps=1e-6
        )
        assert len(auto_out) == len(hand_out) == 2
        for i, (a, h) in enumerate(zip(auto_out, hand_out, strict=True)):
            assert torch.equal(a, h), (
                f"output {i} drifted at point={point} seed={seed}: "
                f"max|diff|={(a - h).abs().max().item()}"
            )


def test_auto_ref_output_shapes_match_contract(spec, sweep_points):
    """The auto-reference output shapes match the declared output tensor symbols."""
    point = sweep_points[0]
    inputs = make_inputs(spec, point, seed=0)
    out1, out2 = run_reference(spec, inputs)
    assert out1.shape == (point["T"], point["d1"])
    assert out2.shape == (point["T"], point["d2"])


def test_auto_ref_handles_awkward_dims(spec):
    """The next_pow2 path works for non-power-of-2 dims (the d2=33 sweep point)."""
    point = {"dtype": "bf16", "T": 37, "d1": 48, "d2": 33}
    inputs = make_inputs(spec, point, seed=7)
    out1, out2 = run_reference(spec, inputs)
    hand1, hand2 = dual_rmsnorm_ref(
        inputs["x1"], inputs["w1"], inputs["x2"], inputs["w2"], eps=1e-6
    )
    assert torch.equal(out1, hand1) and torch.equal(out2, hand2)
