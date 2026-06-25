"""Ergonomic dataclass wrappers around Op Spec / Impl Card documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .._backends import Backend


@dataclass(frozen=True)
class Numerics:
    reference: str
    rtol: float
    atol: float
    cross_backend_rtol: float
    reduce_dtype: str | None = None
    notes: str | None = None
    by_dtype: dict[str, dict[str, float]] = field(default_factory=dict)

    def tolerance_for(self, dtype: str) -> tuple[float, float]:
        """Return (rtol, atol) for a short dtype name, falling back to defaults."""
        entry = self.by_dtype.get(dtype)
        if entry:
            return float(entry["rtol"]), float(entry["atol"])
        return self.rtol, self.atol


@dataclass(frozen=True)
class OpSpec:
    id: str
    name: str
    version: str
    kernel: str                       # dispatch key, e.g. "ffn"
    signature: str
    canonical_op: str
    fusions: tuple[str, ...]
    inputs: dict[str, dict]
    outputs: dict[str, dict]
    constraints: tuple[str, ...]
    preconditions: tuple[str, ...]
    numerics: Numerics
    shape_sweep: str
    composes_with: tuple[str, ...]
    doc: dict                         # the raw validated document

    @property
    def short_name(self) -> str:
        return self.id.split("@", 1)[0]


@dataclass(frozen=True)
class ArchSpec:
    family: str
    requires: tuple[str, ...] = ()
    wave_size: int = 0
    scratch: dict[str, Any] = field(default_factory=lambda: {"kind": "none", "bytes": 0})


@dataclass(frozen=True)
class Measurement:
    arch: str
    shape: dict[str, int]
    dtype: str
    knobs: dict[str, Any]
    source: str
    tflops: float | None = None
    achieved_bw_pct: float | None = None
    ms: float | None = None


@dataclass(frozen=True)
class ImplCard:
    id: str
    implements: str                   # Op Spec id
    backend: Backend
    arch: ArchSpec
    specialization_knobs: dict[str, dict]
    perf_regime: str
    roofline: str
    measured: tuple[Measurement, ...]
    uses_primitives: tuple[str, ...]
    supersedes: tuple[str, ...]
    provenance: dict[str, Any]
    doc: dict                         # the raw validated document

    @classmethod
    def from_doc(cls, doc: dict) -> ImplCard:
        arch_doc = doc["arch"]
        scratch = arch_doc.get("scratch") or {"kind": "none", "bytes": 0}
        measured = tuple(
            Measurement(
                arch=m["arch"], shape=m["shape"], dtype=m["dtype"], knobs=m["knobs"],
                source=m["source"], tflops=m.get("tflops"),
                achieved_bw_pct=m.get("achieved_bw_pct"), ms=m.get("ms"),
            )
            for m in doc.get("perf", {}).get("measured", [])
        )
        return cls(
            id=doc["id"],
            implements=doc["implements"],
            backend=Backend(doc["backend"]),
            arch=ArchSpec(
                family=arch_doc["family"],
                requires=tuple(arch_doc.get("requires", [])),
                wave_size=arch_doc.get("wave_size", 0),
                scratch=scratch,
            ),
            specialization_knobs=doc.get("specialization_knobs", {}),
            perf_regime=doc.get("perf", {}).get("regime", ""),
            roofline=doc.get("perf", {}).get("roofline", "unknown"),
            measured=measured,
            uses_primitives=tuple(doc.get("uses_primitives", [])),
            supersedes=tuple(doc.get("supersedes", [])),
            provenance=doc["provenance"],
            doc=doc,
        )

    @property
    def short_name(self) -> str:
        return self.id.split("@", 1)[0]

    def matches_measurement(self, arch: str, shape: dict, dtype: str) -> Measurement | None:
        """Return the most specific measured entry for (arch, shape, dtype), or None."""
        shape_norm = {str(k): v for k, v in shape.items()}
        for m in self.measured:
            if m.arch == arch and m.dtype == dtype and dict(m.shape) == shape_norm:
                return m
        return None


def op_spec_from_doc(doc: dict) -> OpSpec:
    num_doc = doc["numerics"]
    rtol = float(num_doc["rtol"])
    numerics = Numerics(
        reference=num_doc["reference"],
        rtol=rtol,
        atol=float(num_doc["atol"]),
        cross_backend_rtol=float(num_doc.get("cross_backend_rtol", rtol)),
        reduce_dtype=num_doc.get("reduce_dtype"),
        notes=num_doc.get("notes"),
        by_dtype=num_doc.get("by_dtype", {}) or {},
    )
    op_doc = doc["op"]
    return OpSpec(
        id=doc["id"],
        name=doc["name"],
        version=doc["version"],
        kernel=doc["kernel"],
        signature=op_doc["signature"],
        canonical_op=op_doc["canonical_op"],
        fusions=tuple(op_doc.get("fusions", [])),
        inputs=doc["inputs"],
        outputs=doc["outputs"],
        constraints=tuple(doc.get("constraints", [])),
        preconditions=tuple(doc.get("preconditions", [])),
        numerics=numerics,
        shape_sweep=doc["shape_sweep"],
        composes_with=tuple(doc.get("composes_with", [])),
        doc=doc,
    )
