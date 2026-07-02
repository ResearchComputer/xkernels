"""Decoding-time sampling ops (issue #69): the stochastic half of the sampler.

Seeds the ``sampling`` canonical_op category with two deterministic-given-inputs
ops -- the RNG is external (``uniform_samples`` is an input tensor), so each op is
bit-exact ``verify``-able:

  * ``sampling_from_probs(probs, uniform_samples)``            -- inverse-CDF multinomial draw
  * ``top_k_sampling_from_probs(probs, uniform_samples, top_k)`` -- mask top-k, renorm, inverse-CDF

These compose with the DSL ``temperature_softmax`` op (which produces the prob
distribution these ops draw from). The remaining flashinfer family members
(``top_p_sampling_from_probs``, ``top_k_top_p_sampling_from_probs``) need a
device-side sort (the nucleus cutoff) and are a follow-up.
"""
from ..._backends import Backend
from ..._dispatch import backend_registration_guard
from .sampling import (
    sampling_from_probs,
    sampling_from_probs_ref,
    top_k_sampling_from_probs,
    top_k_sampling_from_probs_ref,
)

# Triton device backends -- register by import side effect. Optional: guarded so
# the package imports without Triton installed. Routed through the optional
# ``_triton_compat`` redirect so the kernels bind ``tokenspeed_triton`` inside
# tokenspeed (see ``xkernels/_triton_compat.py``).
with backend_registration_guard(
    "sampling_from_probs", Backend.TRITON,
    source="xkernels.ops.sampling.triton.sampling_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import sampling_kernel  # noqa: F401

with backend_registration_guard(
    "top_k_sampling_from_probs", Backend.TRITON,
    source="xkernels.ops.sampling.triton.sampling_kernel",
):  # pragma: no cover - requires triton
    from ..._triton_compat import triton_import_ctx

    with triton_import_ctx():
        from .triton import sampling_kernel  # noqa: F401  (same module, both regs)

__all__ = [
    "sampling_from_probs",
    "sampling_from_probs_ref",
    "top_k_sampling_from_probs",
    "top_k_sampling_from_probs_ref",
]
