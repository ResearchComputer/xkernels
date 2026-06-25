"""Agent-facing registry of Op Specs, Implementation Cards, and shape sweeps.

Public surface (see docs/library.md §2, §8):
    - load / all_specs / all_cards / get_spec / get_card / cards_for
    - reference_callable / backend_callable  (metadata -> existing dispatch)
    - load_shape_sweep
"""
from __future__ import annotations

from .constraints import (
    UndecidableConstraintError,
    evaluate,
    validate_decidable,
)
from .loader import (
    RegistryError,
    all_cards,
    all_specs,
    backend_callable,
    card_by_short_name,
    cards_for,
    get_card,
    get_spec,
    load,
    load_shape_sweep,
    reference_callable,
    reset_cache,
)
from .models import ArchSpec, ImplCard, Numerics, OpSpec, op_spec_from_doc
from .outcomes import all_outcomes, record_outcome, reset_outcomes, skill_metrics
from .schemas import have_validator, impl_card_schema, op_spec_schema, registry_root
from .writeback import measurement_view, record_measurement

__all__ = [
    "ArchSpec",
    "ImplCard",
    "Numerics",
    "OpSpec",
    "RegistryError",
    "UndecidableConstraintError",
    "all_cards",
    "all_outcomes",
    "all_specs",
    "backend_callable",
    "card_by_short_name",
    "cards_for",
    "evaluate",
    "get_card",
    "get_spec",
    "have_validator",
    "impl_card_schema",
    "load",
    "load_shape_sweep",
    "measurement_view",
    "op_spec_from_doc",
    "op_spec_schema",
    "record_measurement",
    "record_outcome",
    "reference_callable",
    "registry_root",
    "reset_cache",
    "reset_outcomes",
    "skill_metrics",
    "validate_decidable",
]
