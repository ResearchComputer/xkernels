# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Single source of truth for the architecture vocabulary.

The ``arch.family`` enum lives in ``registry/schema/impl_card.schema.json``
because it is part of the contract. This module is its Python mirror, plus the
vendor mapping that retrieval / verify use to keep cuda cards off amd targets
(and vice-versa).

Keeping two copies (JSON + Python) is unavoidable — JSON Schema can't import a
Python module — so ``tests/test_registry.py::test_arch_vocab_matches_schema``
asserts the two stay in sync. The next arch addition is therefore: one line in
the schema enum + one line here + the test catches any drift.
"""
from __future__ import annotations

AMD_ARCHS = frozenset({"amd_cdna2", "amd_cdna3"})
NVIDIA_ARCHS = frozenset({"nvidia_sm80", "nvidia_sm90", "nvidia_sm100", "nvidia_sm121"})

# Every concrete (non-"any") arch family the contract knows about. MUST match the
# ``arch.family`` enum in impl_card.schema.json (minus the "any" entry).
ALL_ARCHS = AMD_ARCHS | NVIDIA_ARCHS


def vendor_of(arch: str) -> str:
    """Return the vendor of an arch id: 'amd' | 'nvidia' | 'any'."""
    if arch in AMD_ARCHS:
        return "amd"
    if arch in NVIDIA_ARCHS:
        return "nvidia"
    return "any"
