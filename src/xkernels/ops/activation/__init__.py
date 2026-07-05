# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Standalone gated-activation ops (issue #67): SwiGLU / GELU-gated multiply.

The bare ``act(gate) * up`` kernels factored out of ``fused_ffn`` — the ops
mini-sglang / vLLM / flashinfer call as ``silu_and_mul`` / ``gelu_and_mul`` when
the FFN is split across a separately-fused GEMM (no flashinfer ROCm wheel). The
contract is authored once in the vkl DSL (``xkernels.vkl.examples.activation``);
this package is the thin dispatch surface that exposes them as
``xkernels.silu_and_mul(gate, up)`` etc.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from . import reference  # noqa: F401  (registers the REFERENCE backend for `xielu`)
from .interface import (
    gelu_and_mul,
    packed_gelu_and_mul,
    packed_silu_and_mul,
    silu_and_mul,
    xielu,
)

# Register the DSL-generated Triton backend for its dispatch side effect. Unlike
# the hand-written triton kernel modules (``import triton`` at top level, so they
# route through ``triton_import_ctx``), the DSL path builds only a lazy host
# launcher at import time — no triton import happens until the kernel is actually
# called — so no import-context redirect is needed here.
with backend_registration_guard(
    ("silu_and_mul", "gelu_and_mul", "packed_silu_and_mul", "packed_gelu_and_mul"),
    Backend.TRITON,
    source="xkernels.ops.activation.triton.activation_kernel",
):  # pragma: no cover - hardware dependent
    from .triton import activation_kernel  # noqa: F401

# Register the HAND-WRITTEN Triton backend for the parametric `xielu` activation
# (issue #80). Unlike the DSL gated activations, xIELU's where(x>0,...) sign
# branch and the param softplus are not expressible in the vkl math IR (no
# value-comparison primitive, no log), so this is the author-an-op-spec hand
# fallback — same pattern as the hand-written ``dual_rmsnorm_kernel``. The import
# binds stock ``triton`` (the kernel JIT-imports it), so it routes through
# ``triton_import_ctx`` (tokenspeed redirect when running inside tokenspeed).
with backend_registration_guard(
    "xielu",
    Backend.TRITON,
    source="xkernels.ops.activation.triton.xielu_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import xielu_kernel  # noqa: F401

__all__ = ["silu_and_mul", "gelu_and_mul", "packed_silu_and_mul", "packed_gelu_and_mul", "xielu"]
