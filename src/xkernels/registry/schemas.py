"""JSON Schema loading + caching for Op Specs and Implementation Cards."""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

# registry/ lives at the repo root; the package is under src/xkernels/.
_REGISTRY_ROOT = Path(__file__).resolve().parents[3] / "registry"
_SCHEMA_DIR = _REGISTRY_ROOT / "schema"

try:
    import jsonschema  # type: ignore[import-untyped]

    _HAVE_JSONSCHEMA = True
except Exception:  # pragma: no cover - optional dep
    _HAVE_JSONSCHEMA = False


def registry_root() -> Path:
    return _REGISTRY_ROOT


@cache
def _load_schema_file(name: str) -> dict:
    path = _SCHEMA_DIR / name
    with path.open() as f:
        return json.load(f)


def op_spec_schema() -> dict:
    return _load_schema_file("op_spec.schema.json")


def impl_card_schema() -> dict:
    return _load_schema_file("impl_card.schema.json")


def have_validator() -> bool:
    """Whether the optional ``jsonschema`` validator is importable.

    When False, the loader still parses and structurally checks artifacts, but
    does not run full JSON Schema validation. Install with ``pip install jsonschema``.
    """
    return _HAVE_JSONSCHEMA


def validate_op_spec(doc: dict) -> None:
    if not _HAVE_JSONSCHEMA:
        _structural_check_op_spec(doc)
        return
    jsonschema.validate(instance=doc, schema=op_spec_schema())


def validate_impl_card(doc: dict) -> None:
    if not _HAVE_JSONSCHEMA:
        _structural_check_impl_card(doc)
        return
    jsonschema.validate(instance=doc, schema=impl_card_schema())


# --- minimal structural fallbacks (used when jsonschema is absent) -------------

_REQUIRED_OP_SPEC = {
    "id", "name", "version", "kernel", "op", "inputs", "outputs",
    "constraints", "numerics", "shape_sweep",
}
_REQUIRED_IMPL_CARD = {
    "id", "implements", "backend", "arch", "specialization_knobs", "perf", "provenance",
}


def _structural_check_op_spec(doc: dict) -> None:
    missing = _REQUIRED_OP_SPEC - doc.keys()
    if missing:
        raise ValueError(f"Op Spec missing required fields: {sorted(missing)}")
    for f in ("inputs", "outputs"):
        if not isinstance(doc.get(f), dict) or not doc[f]:
            raise ValueError(f"Op Spec '{f}' must be a non-empty object")


def _structural_check_impl_card(doc: dict) -> None:
    missing = _REQUIRED_IMPL_CARD - doc.keys()
    if missing:
        raise ValueError(f"Impl Card missing required fields: {sorted(missing)}")
