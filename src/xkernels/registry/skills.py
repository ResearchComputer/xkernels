"""Load and serve the SKILL.md skills library (meta/docs/library.md §7).

Skills are reusable procedural knowledge authored in the open SKILL.md format
(a folder with a ``SKILL.md``: YAML frontmatter + markdown body), so any
skills-compatible agent (Claude Code, Codex, Gemini CLI, Cursor, Cline, …) can
consume them. The standard fields (``name``, ``description``, ``license``) are
the universal layer; our library-specific metadata lives under a namespaced
``x-kernel-lib`` block that non-standard consumers ignore (§7.1).

This module is the library's own reader for that corpus: it parses the
frontmatter, exposes ergonomic dataclasses, filters by ``backend_scope`` (§7.2),
and is consulted by the outcome store so a ``skill_id`` must name a real skill
before its metrics can roll (integrity for the §7.3 governance loop).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loader import registry_root

# Marker that the skill operates at the Op Spec / contract level, not a backend.
BACKEND_AGNOSTIC = "agnostic"


class SkillError(ValueError):
    """Raised for any problem loading a SKILL.md file."""


@dataclass(frozen=True)
class SkillMeta:
    """The namespaced ``x-kernel-lib`` block (library-specific)."""

    id: str                                   # e.g. "tune-for-cdna@1.0.0"
    backend_scope: tuple[str, ...]            # ("agnostic",) | ("cuda","hip") | ...
    triggers: tuple[str, ...]                 # when_to_use.triggers
    preconditions: tuple[str, ...]            # when_to_use.preconditions
    inputs_required: tuple[str, ...]
    tools: tuple[str, ...]                    # MCP tools the procedure calls (§8)
    validation_must_pass: tuple[str, ...]
    references: tuple[str, ...]
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics_hint: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Skill:
    """A parsed SKILL.md skill (standard fields + namespaced metadata + body)."""

    name: str                                 # standard: the trigger-selection handle
    description: str                          # standard: what every agent reads to decide
    license: str | None                       # standard
    meta: SkillMeta | None                    # namespaced x-kernel-lib block (None if absent)
    body: str                                 # the markdown procedure (universal layer)
    path: Path                                # repo-relative path to SKILL.md

    @property
    def id(self) -> str:
        """Canonical skill id. Prefers the namespaced ``x-kernel-lib.id``;
        falls back to the standard ``name`` (so a plain SKILL.md still works)."""
        return self.meta.id if self.meta else self.name

    @property
    def version(self) -> str:
        """Skill version (from the namespaced id 'name@semver'), else ''."""
        if self.meta and "@" in self.meta.id:
            return self.meta.id.split("@", 1)[1]
        return ""

    def applies_to_backend(self, backend: str) -> bool:
        """Does this skill's ``backend_scope`` include ``backend``?

        ``agnostic`` applies to all backends (§7.2). Otherwise the backend must
        be listed (e.g. a ``[hip]`` skill does not fire for cuda).
        """
        if not self.meta:
            return True  # a plain SKILL.md with no x-kernel-lib block is agnostic
        scope = self.meta.backend_scope
        if not scope or BACKEND_AGNOSTIC in scope:
            return True
        return backend in scope


# --- parsing ------------------------------------------------------------------

def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a SKILL.md into (frontmatter_dict, body). Frontmatter is YAML
    between leading ``---`` fences."""
    if not text.startswith("---"):
        return {}, text
    try:
        import yaml  # type: ignore[import-untyped]
    except Exception as e:  # pragma: no cover - optional dep
        raise SkillError(
            "PyYAML is required to load skills; install with `pip install pyyaml`"
        ) from e
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise SkillError("malformed SKILL.md: missing closing '---' fence")
    fm = yaml.safe_load(parts[1])
    if not isinstance(fm, dict):
        raise SkillError("SKILL.md frontmatter must be a YAML mapping")
    return fm, parts[2].lstrip("\n")


def _coerce_scope(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return (BACKEND_AGNOSTIC,)
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    raise SkillError(f"backend_scope must be a string or list, got {type(raw).__name__}")


def _as_tuple(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(x) for x in raw)


def _parse_skill(path: Path) -> Skill:
    text = path.read_text()
    fm, body = _split_frontmatter(text)
    if "name" not in fm:
        raise SkillError(f"{path}: SKILL.md frontmatter missing required 'name'")
    description = str(fm.get("description", "")).strip()
    if not description:
        raise SkillError(f"{path}: 'description' is required (the trigger field)")

    raw_xkl = fm.get("x-kernel-lib")
    meta: SkillMeta | None = None
    if raw_xkl is not None:
        if not isinstance(raw_xkl, dict):
            raise SkillError(f"{path}: 'x-kernel-lib' must be a mapping")
        if "id" not in raw_xkl:
            raise SkillError(f"{path}: x-kernel-lib.id is required")
        when = raw_xkl.get("when_to_use", {}) or {}
        meta = SkillMeta(
            id=str(raw_xkl["id"]),
            backend_scope=_coerce_scope(raw_xkl.get("backend_scope", BACKEND_AGNOSTIC)),
            triggers=_as_tuple(when.get("triggers")),
            preconditions=_as_tuple(when.get("preconditions")),
            inputs_required=_as_tuple(raw_xkl.get("inputs_required")),
            tools=_as_tuple(raw_xkl.get("tools")),
            validation_must_pass=_as_tuple(
                (raw_xkl.get("validation", {}) or {}).get("must_pass")
            ),
            references=_as_tuple(raw_xkl.get("references")),
            provenance=dict(raw_xkl.get("provenance", {}) or {}),
            metrics_hint=dict(raw_xkl.get("metrics", {}) or {}),
        )
    return Skill(
        name=str(fm["name"]),
        description=description,
        license=str(fm["license"]) if fm.get("license") else None,
        meta=meta,
        body=body,
        path=path,
    )


# --- loading ------------------------------------------------------------------

def skills_dir() -> Path:
    """The SKILL.md corpus lives at the cross-harness standard location
    ``.agents/skills/`` (repo root), which Pi, Claude Code, Codex, and other
    skills-compatible agents discover automatically once the project is
    trusted. Keeping the library's reader pointed here means the same files
    are both agent-discoverable *and* queryable via ``xkernels.registry``."""
    return registry_root().parent / ".agents" / "skills"


def _iter_skill_files() -> list[Path]:
    root = skills_dir()
    if not root.is_dir():
        return []
    return sorted(root.glob("*/SKILL.md"))


def load_skills() -> dict[str, Skill]:
    """Load every ``.agents/skills/*/SKILL.md``. Returns {skill_id: Skill}.

    The key is the canonical skill id (``x-kernel-lib.id`` if present, else the
    standard ``name``). Duplicate ids are rejected.
    """
    skills: dict[str, Skill] = {}
    for path in _iter_skill_files():
        skill = _parse_skill(path)
        if skill.id in skills:
            raise SkillError(f"duplicate skill id {skill.id!r} ({path} vs {skills[skill.id].path})")
        skills[skill.id] = skill
    return skills


# --- accessors ----------------------------------------------------------------

def all_skills() -> dict[str, Skill]:
    return load_skills()


def get_skill(skill_id: str) -> Skill:
    skills = all_skills()
    if skill_id in skills:
        return skills[skill_id]
    # tolerate the bare 'name' form (no @version) via unique prefix match
    matches = [s for sid, s in skills.items() if sid.split("@", 1)[0] == skill_id]
    if len(matches) == 1:
        return matches[0]
    raise KeyError(
        f"unknown skill {skill_id!r}; have {sorted(skills)}"
    )


def has_skill(skill_id: str) -> bool:
    return skill_id in all_skills()


def skills_for_backend(backend: str) -> dict[str, Skill]:
    """Skills whose ``backend_scope`` applies to ``backend`` (§7.2).

    ``agnostic`` skills are always included. Used by an agent to pick a skill
    that actually applies to the target backend — a procedure can be reliable on
    one vendor and weak on another, so we score them separately (§7.3).
    """
    return {sid: s for sid, s in all_skills().items() if s.applies_to_backend(backend)}


def skills_by_trigger(trigger: str) -> list[Skill]:
    """Skills whose ``when_to_use.triggers`` contains ``trigger`` (substring match)."""
    out = []
    for s in all_skills().values():
        if s.meta and any(trigger in t for t in s.meta.triggers):
            out.append(s)
    return out


def validate_skill_id(skill_id: str) -> str:
    """Confirm ``skill_id`` names a real skill, or raise.

    Called by the outcome store (§7.3) so metrics can't roll for a skill that
    doesn't exist — integrity for the governance loop.
    """
    if not has_skill(skill_id):
        # tolerate the bare 'name' form (no @version) by trying a unique prefix match
        matches = [sid for sid in all_skills() if sid.split("@", 1)[0] == skill_id]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"unknown skill {skill_id!r}; have {sorted(all_skills())}"
        )
    return skill_id
