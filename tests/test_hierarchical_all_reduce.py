# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Distributed correctness smoke for the hierarchical all-reduce (issue #12).

Launches an 8-rank logical topology (2 nodes x 4) on CPU/gloo via torchrun and
checks ``hierarchical_all_reduce == flat_all_reduce``. Real RCCL numerics +
latency on MI300A are exercised by scripts/archive/issues/bench_allreduce_beverin.sbatch.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_hierarchical_all_reduce.py"


@pytest.mark.skipif(shutil.which("torchrun") is None, reason="torchrun not installed")
def test_hierarchical_matches_flat_8rank_gloo():
    # 8 logical ranks, 4 per node -> exercises both intra (4) and cross (2) legs.
    cmd = [
        "torchrun",
        "--nproc-per-node=8",
        "--master-port=29555",
        str(_BENCH),
        "--ranks-per-node",
        "4",
        "--iters",
        "3",
        "--warmup",
        "1",
        "--sizes",
        "1",
        "4",
        "16",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"torchrun failed:\n{out}"
    assert "correctness (all ranks): PASS" in out, f"correctness not reported PASS:\n{out}"
    print(out, file=sys.stderr)
