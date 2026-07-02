# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""The authoring surface: ``@kernel`` + ``@targets`` decorators (docs/brainstorm/04).

Phase 1 model (made explicit, per docs/brainstorm/11 §2 CPU-satisfiable subset):

  * The ``@kernel`` **header** is a declarative spelling of the Op Spec. It is a
    1:1 projection (``emit.py``) — if the projection ever needs a "translation",
    the contract-native thesis (docs/brainstorm/02 §1) is broken.
  * The ``@kernel`` **body** is the computation. On CPU it IS the reference
    (``reference.py`` runs it on torch). This structurally guarantees the
    auto-reference cannot drift from the author's intent — there is one
    computation, not two.
  * The per-program tiling + Triton/CUDA/HIP lowering is the GPU-gated path
    (Phase 1.5 / Phase 2). On CPU we cannot lower to a device kernel anyway, so
    Phase 1's body is vectorized torch: the same arithmetic as the device kernel,
    bit-exact against the hand-written reference.

This is an honest scope line: body-parsing into the math IR (docs/brainstorm/10
§6) and device lowering are deferred, not hand-waved. The math IR dataclasses
(``ir/math.py``) exist as the frozen oracle edits respect; Phase 1's reference is
the body, and a future test will check body == math-IR-derived reference.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .ir.math import MathIR

# A sentinel meaning "the reference is the @kernel body itself" — emit.py turns
# this into the conventional reference import path the DSL owns.
AUTO_REFERENCE = "<auto>"


@dataclass(frozen=True)
class TensorDecl:
    """Mirror of the schema's tensorContract (op_spec.schema.json $defs).

    ``reduces_over`` is vkl-only metadata (for OUTPUTS): the input name whose
    last axis this output shares. Used by the row-reduce lowering to size output
    tiles. It is NOT emitted into the JSON contract (the schema's tensorContract
    is closed); it lives on the authoring surface only.
    """

    rank: int
    dtype: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    layout: str = "row_major"
    reduces_over: str | None = None  # vkl-only; outputs name the input sharing their d

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "dtype": list(self.dtype),
            "rank": self.rank,
            "layout": self.layout,
        }
        if self.symbols:
            d["shape_symbols"] = list(self.symbols)
        return d


@dataclass(frozen=True)
class Numerics:
    """Mirror of the schema's numerics block (tolerances defined once, centrally)."""

    rtol: float
    atol: float
    reference: str = AUTO_REFERENCE
    reduce_dtype: str | None = None
    cross_backend_rtol: float | None = None
    by_dtype: dict[str, dict[str, float]] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self, reference_path: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "reference": reference_path,
            "rtol": self.rtol,
            "atol": self.atol,
        }
        if self.reduce_dtype is not None:
            d["reduce_dtype"] = self.reduce_dtype
        if self.cross_backend_rtol is not None:
            d["cross_backend_rtol"] = self.cross_backend_rtol
        if self.by_dtype:
            d["by_dtype"] = self.by_dtype
        if self.notes is not None:
            d["notes"] = self.notes
        return d


@dataclass(frozen=True)
class Target:
    """One Impl Card's worth of target info (the ``@targets`` entry).

    Projects 1:1 into the Impl Card's ``arch`` block + ``specialization_knobs``
    (docs/brainstorm/10 §0). ``wave_size`` is bound by the target, never by a
    human remembering "32 vs 64" — defaults to 0 (agnostic) unless the target
    uses a wave-level primitive.
    """

    backend: str  # "triton" | "cuda" | "hip"
    arch: str = "any"  # "any" | "nvidia_sm90" | "amd_cdna3" | ...
    requires: tuple[str, ...] = ()
    wave_size: int = 0  # 0 (agnostic) | 32 (NVIDIA) | 64 (AMD)
    scratch_kind: str = "none"  # "smem" | "lds" | "registers" | "none"
    knobs: dict[str, tuple[int, ...]] = field(default_factory=dict)  # name -> choices
    roofline: str = "unknown"  # compute_bound | memory_bound | latency_bound | unknown
    regime: str = ""


@dataclass(frozen=True)
class OverrideBody:
    """A per-target override body (docs/brainstorm/04 Ex.2, Axis H1).

    The override reaches a specific (backend, arch) ceiling using MORE NATIVE code
    than the portable body (TMA descriptors, wgmma intrinsics, clusters on
    sm_90; MFMA on cdna3) — but it builds the SAME math IR, so the oracle
    property holds: ``verify`` checks it against the SAME auto-reference (the
    portable body on torch). The override is *more* native code, not a *different*
    op (docs/brainstorm/04 §"What this example establishes").

    The override body is a trace-builder taking ``ctx`` (like the portable body);
    its build is recorded on the ``Target`` so the emitter produces one Impl Card
    per (portable, *override) target. Native compilation of the override is
    GPU-gated (Phase 2.1); the decorator + the math-IR-invariant check are the
    CPU-doable mechanism shipped here.
    """

    backend: str  # "cuda" | "hip"
    arch: str  # "nvidia_sm90" | "amd_cdna3" | ...
    body: Callable[..., Any]  # the override trace-builder (ctx)
    provenance_kind: str = "full_body"  # H1 (full-body) vs H2 (primitive-swap) — Axis H


@dataclass(frozen=True)
class KernelSpec:
    """A fully-declared kernel: header + body + targets (+ optional math IR).

    Built by ``@kernel``. The frozen record ``emit.py`` projects to JSON and
    ``reference.py`` runs on torch.
    """

    id: str
    kernel: str
    name: str
    version: str
    signature: str
    canonical_op: str
    inputs: dict[str, TensorDecl]
    outputs: dict[str, TensorDecl]
    constraints: tuple[str, ...]
    numerics: Numerics
    shape_sweep: str
    body: Callable[..., Any]
    targets: dict[str, Target]
    fusions: tuple[str, ...] = ()
    preconditions: tuple[str, ...] = ()
    math: MathIR | None = None  # optional declarative oracle (Phase 1.5 derives from body)
    launch: Launch | None = None  # device lowering pattern; None = body is direct torch
    overrides: tuple[OverrideBody, ...] = ()  # per-(backend,arch) ceiling bodies (Phase 2.1)

    @property
    def short_name(self) -> str:
        return self.id.split("@", 1)[0]

    @property
    def reference_path(self) -> str:
        """The import path emit.py writes into numerics.reference.

        ``AUTO_REFERENCE`` means "the @kernel body"; emit.py turns that into the
        conventional path ``xkernels.vkl.<short_name>:<short_name>`` so the
        auto-reference is resolvable like any hand-written reference.
        """
        if self.numerics.reference != AUTO_REFERENCE:
            return self.numerics.reference
        return f"xkernels.vkl.auto:{self.short_name}"

    def override_for(self, backend: str, arch: str | None = None) -> OverrideBody | None:
        """The override body for ``(backend[, arch])``, or None (use the portable).

        Arch-specific overrides win over backend-wide ones; within an identical
        ``(backend, arch)`` key, the latest decorator wins. ``None`` arch means
        "any arch of this backend" (a backend-wide ceiling body).
        """
        # Prefer an arch-specific match; fall back to a backend-wide override.
        exact = [o for o in reversed(self.overrides) if o.backend == backend and o.arch == arch]
        if exact:
            return exact[0]
        wide = [o for o in reversed(self.overrides) if o.backend == backend and o.arch == "any"]
        return wide[0] if wide else None


def _attach_targets(fn: Callable, targets: dict[str, Target]) -> None:
    fn._vkl_targets = targets  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Launch:
    """Declares the device lowering pattern (the grid / parallelism).

    Two patterns lower from the SAME doc-10 math IR (``vkl.lower.mathbody``):
    ``rowwise`` — one program per leading-dim row, reducing over the last axis
    (the ``dual_rmsnorm`` shape); ``tiled_2d`` — a 2D grid with a K-loop (the GEMM
    shape). CUDA/HIP native overrides are Phase 2.1.
    """

    pattern: str  # "rowwise" | "tiled_2d" | "elementwise"

    @staticmethod
    def rowwise() -> Launch:
        return Launch(pattern="rowwise")

    @staticmethod
    def tiled_2d() -> Launch:
        """A 2D-tiled launch: one program per output tile (the GEMM / conv2d shape).

        Lowers as a 2D grid over the output's first two dims; an MMA node's
        contracted dim becomes the K-loop (docs/brainstorm/08 §3). Phase 2.0a
        lowering target; the math IR's ``subscript`` (Einstein labels) drives the
        tiling (docs/brainstorm/11 §11).
        """
        return Launch(pattern="tiled_2d")

    @staticmethod
    def elementwise() -> Launch:
        """A flat-1D launch: one program per tile of the flattened output (pure pointwise).

        Lowers as a 1D grid over ``numel = prod(output shape)``; every node is a
        Load/Pointwise/Store over the same element grid — NO reduction axis. Use
        for standalone elementwise ops with no reduction (gated activations like
        ``silu_and_mul``/``gelu_and_mul``, issue #67). Does not support rank-1
        broadcast (use ``rowwise`` for a weight broadcast).
        """
        return Launch(pattern="elementwise")


def targets(**tgt_kwargs: Any) -> Callable:
    """Decorator: declare the Impl Card set (docs/brainstorm/04).

    ``@targets(triton=Target(...), cuda=Target(...))`` attaches the targets to
    the body so ``@kernel`` can read them. Each kwarg key is a backend name; the
    value is a ``Target`` (or a dict ``Target``-constructor-compatible).
    """
    def deco(fn: Callable) -> Callable:
        resolved: dict[str, Target] = {}
        for backend, t in tgt_kwargs.items():
            resolved[backend] = t if isinstance(t, Target) else Target(backend=backend, **t)
        _attach_targets(fn, resolved)
        return fn

    return deco


def launch(launch_decl: Launch) -> Callable:
    """Decorator: declare the device lowering pattern for the body.

    ``@launch(Launch.rowwise())`` attaches the pattern so ``@kernel`` can read
    it. When present, the body is a trace-builder (takes ``ctx``); the lowering
    builds the IR + lowers to torch (reference) and Triton (device). When absent,
    the body is a direct torch computation (the Phase 1 form).
    """
    def deco(fn: Callable) -> Callable:
        fn._vkl_launch = launch_decl  # type: ignore[attr-defined]
        return fn
    return deco


def kernel(
    *,
    id: str,
    kernel: str,
    inputs: dict[str, TensorDecl],
    outputs: dict[str, TensorDecl],
    numerics: Numerics,
    shape_sweep: str,
    canonical_op: str,
    name: str | None = None,
    version: str | None = None,
    signature: str = "",
    constraints: tuple[str, ...] | list[str] = (),
    fusions: tuple[str, ...] | list[str] = (),
    preconditions: tuple[str, ...] | list[str] = (),
    targets: dict[str, Target] | None = None,
    math: MathIR | None = None,
    launch: Launch | None = None,
) -> Callable:
    """Decorator: declare an op's contract + capture its reference body.

    The header fields mirror op_spec.schema.json 1:1. The decorated function is
    the reference computation (run on torch by ``reference.py``). Returns the
    body unchanged but attaches the ``KernelSpec`` as ``body._vkl_spec`` so the
    emitter / reference runner can find it.
    """
    def deco(fn: Callable) -> Callable:
        # version + name default from the id ("<name>@<version>").
        _name, _, _ver = id.partition("@")
        spec = KernelSpec(
            id=id,
            kernel=kernel,
            name=name or _name,
            version=version or _ver,
            signature=signature,
            canonical_op=canonical_op,
            inputs=dict(inputs),
            outputs=dict(outputs),
            constraints=tuple(constraints),
            numerics=numerics,
            shape_sweep=shape_sweep,
            body=fn,
            targets=dict(targets or getattr(fn, "_vkl_targets", {})),
            fusions=tuple(fusions),
            preconditions=tuple(preconditions),
            math=math,
            launch=launch or getattr(fn, "_vkl_launch", None),
        )
        fn._vkl_spec = spec  # type: ignore[attr-defined]
        # Auto-register the reference so the emitted ``numerics.reference``
        # (``xkernels.vkl.auto:<short_name>``) resolves. For a DIRECT body the
        # reference IS the body (called with inputs). For a TRACE body the
        # reference is a wrapper that builds the IR + evaluates on torch
        # (``run_reference``); the body itself takes a ``ctx``, not inputs.
        from .auto import register_auto
        if spec.launch is None:
            auto_ref: Callable[..., Any] = fn
        else:
            from .reference import run_reference as _run_reference

            def _ref(**inputs):  # type: ignore[no-untyped-def]
                return _run_reference(spec, inputs)

            auto_ref = _ref
        register_auto(spec.short_name, auto_ref)
        # Auto-wire the op into the substrate so ``verify()`` runs its reference
        # on CPU with zero per-op boilerplate (input generator + REFERENCE
        # dispatch). First-writer-wins: a hand op imported eagerly at
        # ``import xkernels`` (e.g. dual_rmsnorm ships its own reference.py +
        # input_gen entry) is never clobbered — ``vkl.examples`` loads lazily
        # AFTER the hand ops, so the hand slots are already filled and the guard
        # skips. A DSL-only op (no hand counterpart) gets wired here.
        _wire_substrate(spec, auto_ref)
        # Attach the per-target override decorator as an attribute so authors
        # spell ``@gemm.target("cuda", arch="nvidia_sm90")`` (docs/brainstorm/04
        # Ex.2). Each call appends an ``OverrideBody`` to ``spec.overrides``.
        fn.target = _make_target_decorator(spec, fn)  # type: ignore[attr-defined]
        return fn

    return deco


def _make_target_decorator(spec: KernelSpec, owner: Callable) -> Callable[..., Callable]:
    """Build the ``@<kernel>.target(backend, arch=...)`` decorator factory.

    Returns a decorator that captures an override body and rebuilds the spec with
    it appended to ``overrides`` (frozen → ``dataclasses.replace``). The override
    body is a trace-builder (``ctx``) like the portable body; it must build the
    SAME math IR (the oracle property — checked at emit/lower time).

    ``owner`` is the original ``@kernel`` function; the rebuilt spec is propagated
    back to ``owner._vkl_spec`` so ``spec_of(<kernel>)`` sees every override
    attached via ``@<kernel>.target(...)`` (the ergonomic discovery contract —
    tests/examples do ``spec_of(gemm_bf16)`` and expect the cuda override).
    """

    def target_factory(
        backend: str, *, arch: str = "any", provenance_kind: str = "full_body"
    ) -> Callable[[Callable], Callable]:
        def deco(body: Callable) -> Callable:
            from dataclasses import replace

            override = OverrideBody(backend=backend, arch=arch, body=body,
                                    provenance_kind=provenance_kind)
            new_spec = replace(spec, overrides=spec.overrides + (override,))
            # propagate to BOTH the override body AND the owning kernel fn so
            # ``spec_of(gemm_bf16)`` and ``spec_of(gemm_bf16_cuda)`` both see it.
            body._vkl_spec = new_spec  # type: ignore[attr-defined]
            owner._vkl_spec = new_spec  # type: ignore[attr-defined]
            # re-attach the decorator so chained ``@gemm.target(...)`` stacking works
            target = _make_target_decorator(new_spec, owner)
            body.target = target  # type: ignore[attr-defined]
            owner.target = target  # type: ignore[attr-defined]
            return body

        return deco

    return target_factory


def spec_of(fn: Callable) -> KernelSpec:
    """Recover the ``KernelSpec`` attached by ``@kernel``."""
    s = getattr(fn, "_vkl_spec", None)
    if s is None:
        raise AttributeError(f"{fn!r} is not an @kernel-decorated function")
    return s


def _wire_substrate(spec: KernelSpec, auto_ref: Callable[..., Any]) -> None:
    """Auto-wire a DSL-authored op into the substrate (input gen + REFERENCE dispatch).

    Makes a DSL op a first-class seeded op that ``verify()`` runs on CPU with no
    per-op boilerplate: the seeded input generator comes from the spec's shape
    symbols (``vkl.reference.make_inputs``), and the REFERENCE dispatch callable
    IS the auto-reference body. Both are guarded first-writer-wins so a hand op
    (imported eagerly at ``import xkernels``, before ``vkl.examples``) keeps its
    own wiring. Imports are lazy so ``surface.py`` stays side-effect-free at
    module load (no circular import with ``xkernels._dispatch`` / ``registry``).
    """
    try:
        from .._dispatch import Backend, register, registered_backends
        from ..registry.input_gen import has_generator, register_input_gen
        from .reference import make_inputs
    except Exception:  # pragma: no cover - substrate not importable in isolation
        return
    # REFERENCE dispatch: the body IS the reference oracle.
    if Backend.REFERENCE not in registered_backends(spec.kernel):
        register(spec.kernel, Backend.REFERENCE)(auto_ref)
    # Seeded input generator derived from the spec's shape symbols.
    if not has_generator(spec.id):
        def _gen(point, seed, device, _spec=spec):  # type: ignore[no-untyped-def]
            return make_inputs(_spec, point, seed=seed, device=device)

        register_input_gen(spec.id, _gen)
