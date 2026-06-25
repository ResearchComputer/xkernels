"""Load and validate all Op Specs / Impl Cards / shape sweeps from the registry.

Ties each Impl Card to the existing ``@register``-ed dispatch callable via
``(op_spec.kernel, card.backend)``, and each Op Spec to its backend-neutral
reference callable via ``numerics.reference`` (an ``import.path:attr`` string).
The existing PyTorch dispatch path is untouched — this layer adds agent-facing
machine legibility on top of it.
"""
from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from .._dispatch import _REGISTRY, registered_backends
from .constraints import UndecidableConstraintError, validate_decidable
from .models import ImplCard, OpSpec, op_spec_from_doc
from .schemas import registry_root, validate_impl_card, validate_op_spec


class RegistryError(ValueError):
    """Raised for any ingest-time problem with an artifact."""


def _read_json(path: Path) -> dict:
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise RegistryError(f"invalid JSON in {path}: {e}") from e


def _load_op_spec(path: Path) -> OpSpec:
    doc = _read_json(path)
    validate_op_spec(doc)
    # Enforce invariant §2.4: constraints must be decidable, or reject at ingest.
    for c in doc.get("constraints", []):
        try:
            validate_decidable(c)
        except UndecidableConstraintError as e:
            raise RegistryError(f"{path.name}: {e}") from e
    return op_spec_from_doc(doc)


def _load_impl_card(path: Path) -> ImplCard:
    doc = _read_json(path)
    validate_impl_card(doc)
    # Enforce invariant §2.4: un-sourced or arch-less perf entries are dropped.
    measured = doc.get("perf", {}).get("measured", [])
    kept = [m for m in measured if m.get("source") and m.get("arch")]
    if len(kept) != len(measured):
        doc = {**doc, "perf": {**doc["perf"], "measured": kept}}
    return ImplCard.from_doc(doc)


def _shape_sweep_dir() -> Path:
    return registry_root() / "shape_sweeps"


@lru_cache(maxsize=1)
def load() -> tuple[dict[str, OpSpec], dict[str, dict[str, ImplCard]]]:
    """Load every artifact once. Returns (specs_by_id, cards_by_op_id_by_backend).

    ``cards_by_op_id_by_backend[op_id][backend_name] = ImplCard``.
    """
    root = registry_root()
    specs: dict[str, OpSpec] = {}
    cards: dict[str, dict[str, ImplCard]] = {}

    op_dir = root / "ops"
    if op_dir.is_dir():
        for path in sorted(op_dir.glob("*.spec.json")):
            spec = _load_op_spec(path)
            if spec.id in specs:
                raise RegistryError(f"duplicate Op Spec id {spec.id!r}")
            specs[spec.id] = spec

    impl_dir = root / "impls"
    if impl_dir.is_dir():
        for path in sorted(impl_dir.glob("*.card.json")):
            card = _load_impl_card(path)
            if card.implements not in specs:
                raise RegistryError(
                    f"{path.name}: implements {card.implements!r} which has no Op Spec"
                )
            bucket = cards.setdefault(card.implements, {})
            if card.backend.name in bucket:
                raise RegistryError(
                    f"duplicate card for {card.implements!r} backend {card.backend.name}"
                )
            bucket[card.backend.name] = card

    return specs, cards


def reset_cache() -> None:
    """Force artifacts to be re-read from disk (tests / write-back)."""
    load.cache_clear()


# --- accessors ----------------------------------------------------------------

def all_specs() -> dict[str, OpSpec]:
    return load()[0]


def all_cards() -> dict[str, dict[str, ImplCard]]:
    return load()[1]


def get_spec(op_id: str) -> OpSpec:
    specs = all_specs()
    if op_id not in specs:
        raise KeyError(f"unknown Op Spec {op_id!r}; have {sorted(specs)}")
    return specs[op_id]


def cards_for(op_id: str) -> dict[str, ImplCard]:
    return all_cards().get(op_id, {})


def get_card(card_id: str) -> ImplCard:
    for bucket in all_cards().values():
        for card in bucket.values():
            if card.id == card_id:
                return card
    raise KeyError(f"unknown Impl Card {card_id!r}")


def card_by_short_name(op_id: str, backend: str) -> ImplCard:
    """Lookup a card for an op by backend name (e.g. 'ffn' + 'triton')."""
    bucket = cards_for(op_id)
    if backend in bucket:
        return bucket[backend]
    # tolerant: allow short card id like 'ffn.triton'
    short = f"{op_id.split('@')[0]}.{backend}"
    for card in bucket.values():
        if card.short_name == short:
            return card
    raise KeyError(f"no {backend!r} card for {op_id!r}")


# --- callable resolution (ties metadata -> existing dispatch registry) --------

def _import_attr(path: str) -> Callable:
    module, _, attr = path.partition(":")
    if not attr:
        raise RegistryError(f"reference path must be 'module:attr', got {path!r}")
    return getattr(importlib.import_module(module), attr)


def reference_callable(op_id: str) -> Callable:
    """The backend-neutral reference callable for an op (from numerics.reference)."""
    return _import_attr(get_spec(op_id).numerics.reference)


def backend_callable(op_id: str, backend: str) -> Callable:
    """The registered dispatch callable for (op.kernel, backend).

    Lazily importing the op package populates ``_REGISTRY``; we import the
    top-level package which triggers all backend self-registration.
    """
    importlib.import_module("xkernels")  # ensures all ops register
    spec = get_spec(op_id)
    impls = _REGISTRY.get(spec.kernel, {})
    from .._backends import Backend as _Backend
    try:
        key = _Backend(backend.lower())
    except ValueError as e:
        raise KeyError(f"unknown backend {backend!r}: {e}") from e
    if key not in impls:
        raise KeyError(
            f"backend {backend!r} not registered for kernel {spec.kernel!r}; "
            f"have {sorted(b.name for b in registered_backends(spec.kernel))}"
        )
    return impls[key]


# --- shape sweeps -------------------------------------------------------------

def load_shape_sweep(sweep_id: str) -> list[dict[str, Any]]:
    """Load a shape sweep manifest as a list of point dicts.

    Each point may carry ``dtype`` (str) and symbolic shape keys, plus an
    optional ``knobs`` override. Points without a dtype inherit the sweep's
    default (or 'fp32').
    """
    path = _shape_sweep_dir() / f"{sweep_id}.sweep.json"
    if not path.exists():
        raise KeyError(f"no shape sweep {sweep_id!r} at {path}")
    doc = _read_json(path)
    points = doc.get("points", [])
    default_dtype = doc.get("default_dtype", "fp32")
    out = []
    for p in points:
        p = dict(p)
        p.setdefault("dtype", default_dtype)
        out.append(p)
    return out
