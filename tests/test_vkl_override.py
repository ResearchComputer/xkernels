# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Phase 2.1 gate: the per-target override MECHANISM (the native-ceiling path).

The full Phase 2.1 (native CUDA/CUTE + HIP/CK codegen reaching the vendor
ceiling) is GPU-gated and environment-blocked on this node (no cu12 nvcc). What
IS CPU-doable — and load-bearing for the whole thesis — is the *mechanism*:

  1. ``@gemm.target("cuda", arch="nvidia_sm90")`` attaches an override body.
  2. ``check_override_math_ir`` enforces the oracle property: the override must
     build the SAME math IR (same op-kind signature) as the portable body. A
     body that computes a *different* op is rejected — it's a new op, not an
     override (route to author-an-op-spec).
  3. ``emit_override_card`` projects the override to its own schema-valid Impl
     Card (cuda/hip backend, arch.requires the native features).

This is the foundation the GPU codegen lands on top of. The native lowering
(wgmma/MFMA intrinsics, TMA, clusters) is the future Phase 2.1 GPU work.
"""
from __future__ import annotations

from xkernels.registry.schemas import validate_impl_card
from xkernels.vkl import (
    OverrideCheck,
    check_override_math_ir,
    emit_override_card,
    spec_of,
)
from xkernels.vkl.examples import gemm_bf16

# ─── the oracle-property invariant ───────────────────────────────────────────


def _correct_cuda_override():
    """Re-derive a fresh spec + a correct cuda override (same math IR)."""

    @gemm_bf16.target("cuda", arch="nvidia_sm90")
    def gemm_cuda(ctx):  # noqa: D401
        a = ctx.load("a")
        b = ctx.load("b")
        acc = ctx.mma(a, b, accum_dtype="fp32")
        out = acc.cast(ctx.out_dtype())
        ctx.store("out", out)

    return spec_of(gemm_cuda)


def _override(spec, backend: str, arch: str):
    ov = spec.override_for(backend, arch)
    assert ov is not None, f"missing override for {backend}/{arch}"
    return ov


class TestOverrideDecorator:
    def test_target_decorator_attaches_override(self):
        spec = _correct_cuda_override()
        # The example ships with its real ds5 sm_121 override; this test adds an
        # sm_90 override and must not clobber the existing one.
        assert spec.override_for("cuda", "nvidia_sm121") is not None
        ov = _override(spec, "cuda", "nvidia_sm90")
        assert ov.backend == "cuda"
        assert ov.arch == "nvidia_sm90"
        assert ov.provenance_kind == "full_body"

    def test_override_for_lookup_prefers_arch_specific(self):
        spec = _correct_cuda_override()
        assert spec.override_for("cuda", "nvidia_sm90") is _override(spec, "cuda", "nvidia_sm90")
        assert spec.override_for("cuda", "nvidia_sm121") is not None
        assert spec.override_for("cuda", "nvidia_sm80") is None  # no backend-wide
        assert spec.override_for("triton", "any") is None  # portable, not an override


class TestOracleInvariant:
    def test_correct_override_passes(self):
        spec = _correct_cuda_override()
        chk = check_override_math_ir(spec, _override(spec, "cuda", "nvidia_sm90"))
        assert isinstance(chk, OverrideCheck)
        assert chk.ok
        assert chk.portable_signature == chk.override_signature
        assert "MMA" in chk.portable_signature

    def test_wrong_op_override_is_rejected(self):
        """An override that drops the MMA computes a different op → reject."""

        @gemm_bf16.target("cuda", arch="nvidia_sm90")
        def gemm_cuda_wrong(ctx):  # noqa: D401
            a = ctx.load("a")
            out = a.cast(ctx.out_dtype())  # NO mma — different signature
            ctx.store("out", out)

        spec = spec_of(gemm_cuda_wrong)
        chk = check_override_math_ir(spec, _override(spec, "cuda", "nvidia_sm90"))
        assert not chk.ok
        assert "signature mismatch" in chk.reason
        assert "author-an-op-spec" in chk.reason
        assert "MMA" in chk.portable_signature
        assert "MMA" not in chk.override_signature

    def test_non_native_backend_rejected(self):
        """An override for 'triton' is meaningless — Triton is the portable body."""

        @gemm_bf16.target("triton", arch="any")
        def gemm_triton(ctx):  # noqa: D401
            a = ctx.load("a")
            b = ctx.load("b")
            acc = ctx.mma(a, b, accum_dtype="fp32")
            ctx.store("out", acc.cast(ctx.out_dtype()))

        spec = spec_of(gemm_triton)
        chk = check_override_math_ir(spec, _override(spec, "triton", "any"))
        assert not chk.ok
        assert "not a native target" in chk.reason


# ─── override card emission ──────────────────────────────────────────────────


class TestOverrideCardEmission:
    def test_emits_schema_valid_cuda_card(self):
        spec = _correct_cuda_override()
        card = emit_override_card(
            spec,
            _override(spec, "cuda", "nvidia_sm90"),
            knobs={"BLOCK_M": (128, 256), "num_stages": (3, 4)},
        )
        validate_impl_card(card)  # raises on schema violation
        assert card["id"] == "gemm_bf16.cuda@1.0.0"
        assert card["backend"] == "cuda"
        assert card["arch"]["family"] == "nvidia_sm90"
        # native features the override declares (tensor_cores + tma + clusters on sm_90)
        assert "tensor_cores" in card["arch"]["requires"]
        assert "tma" in card["arch"]["requires"]
        assert "clusters" in card["arch"]["requires"]
        # provenance records the derivation
        assert card["provenance"]["derived_from"] == "gemm_bf16.triton@1.0.0"
        # the declared knob space projects to specialization_knobs
        assert set(card["specialization_knobs"]) == {"BLOCK_M", "num_stages"}

    def test_emits_schema_valid_hip_card(self):
        spec = spec_of(gemm_bf16)

        @gemm_bf16.target("hip", arch="amd_cdna3")
        def gemm_hip(ctx):  # noqa: D401
            a = ctx.load("a")
            b = ctx.load("b")
            ctx.store("out", ctx.mma(a, b, accum_dtype="fp32").cast(ctx.out_dtype()))

        spec = spec_of(gemm_hip)
        card = emit_override_card(spec, _override(spec, "hip", "amd_cdna3"))
        validate_impl_card(card)
        assert card["backend"] == "hip"
        assert card["arch"]["family"] == "amd_cdna3"
        assert "mfma" in card["arch"]["requires"]
        assert card["arch"]["wave_size"] == 64  # AMD
        assert card["arch"]["scratch"]["kind"] == "lds"

    def test_override_card_supersedes_portable_in_provenance(self):
        """The override's derived_from points at the portable Triton card."""
        spec = _correct_cuda_override()
        card = emit_override_card(spec, _override(spec, "cuda", "nvidia_sm90"))
        assert card["provenance"]["derived_from"] == "gemm_bf16.triton@1.0.0"
        assert card["provenance"]["authored_by"] == "dsl"
