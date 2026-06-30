"""Agent-facing registry of Op Specs, Implementation Cards, and shape sweeps.

Public surface (see meta/docs/library.md §2, §8):
    - load / all_specs / all_cards / get_spec / get_card / cards_for
    - reference_callable / backend_callable  (metadata -> existing dispatch)
    - load_shape_sweep
"""
from __future__ import annotations

from .archs import ALL_ARCHS, AMD_ARCHS, NVIDIA_ARCHS, vendor_of
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
from .skills import (
    BACKEND_AGNOSTIC,
    Skill,
    SkillError,
    SkillMeta,
    all_skills,
    get_skill,
    has_skill,
    load_skills,
    skills_by_trigger,
    skills_dir,
    skills_for_backend,
    validate_skill_id,
)
from .writeback import measurement_view, record_measurement

__all__ = [
    "AMD_ARCHS",
    "ALL_ARCHS",
    "ArchSpec",
    "BACKEND_AGNOSTIC",
    "ImplCard",
    "NVIDIA_ARCHS",
    "Numerics",
    "OpSpec",
    "RegistryError",
    "Skill",
    "SkillError",
    "SkillMeta",
    "UndecidableConstraintError",
    "all_cards",
    "all_outcomes",
    "all_specs",
    "all_skills",
    "backend_callable",
    "card_by_short_name",
    "cards_for",
    "evaluate",
    "get_card",
    "get_skill",
    "get_spec",
    "has_skill",
    "have_validator",
    "impl_card_schema",
    "load",
    "load_shape_sweep",
    "load_skills",
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
    "skills_by_trigger",
    "skills_dir",
    "skills_for_backend",
    "validate_decidable",
    "validate_skill_id",
    "vendor_of",
]
