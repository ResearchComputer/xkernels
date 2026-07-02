# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Auto-reference registry for DSL-authored kernels.

When a kernel declares ``numerics.reference = AUTO_REFERENCE`` (the default),
``emit.py`` writes ``xkernels.vkl.auto:<short_name>`` into the spec. This module
is where those bodies live, so the import path resolves like any hand-written
reference (``xkernels.ops.norm.reference:dual_rmsnorm_ref``).

``@kernel`` auto-registers its body here by short name. A DSL kernel and a
hand-written kernel MUST NOT share a short name (the Op Spec id distinguishes
them, but the auto-reference namespace is flat by design — one body per name).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_auto(short_name: str, body: Callable[..., Any]) -> None:
    """Register a ``@kernel`` body as the auto-reference for ``short_name``."""
    if short_name in _REGISTRY and _REGISTRY[short_name] is not body:
        raise ValueError(
            f"auto-reference name {short_name!r} already registered to a different body; "
            f"DSL kernel short names must be unique"
        )
    _REGISTRY[short_name] = body


def get_auto(short_name: str) -> Callable[..., Any]:
    """Resolve the auto-reference body for ``short_name`` (the spec's reference path).

    Lazily imports ``xkernels.vkl.examples`` on a miss: importing the package is
    side-effect-free by design (``__init__`` does not register bodies), but the
    reference path (``xkernels.vkl.auto:<short_name>``) must resolve from a fresh
    process that only did ``import xkernels``. The examples subpackage
    self-registers each ``@kernel`` body, so importing it on first lookup makes
    DSL ops symmetric with hand ops (whose reference module self-registers on
    the import ``_import_attr`` performs).
    """
    try:
        return _REGISTRY[short_name]
    except KeyError:
        try:
            from . import examples  # noqa: F401  side-effect: register auto-refs
        except Exception:
            pass
        if short_name in _REGISTRY:
            return _REGISTRY[short_name]
        raise KeyError(
            f"no auto-reference registered for {short_name!r}; "
            f"was the @kernel module imported?"
        ) from None


def __getattr__(name: str) -> Callable[..., Any]:
    """Module-level attribute access resolves auto-references (so `auto:dual_rmsnorm` works)."""
    if name.startswith("_"):
        raise AttributeError(name)
    return get_auto(name)
