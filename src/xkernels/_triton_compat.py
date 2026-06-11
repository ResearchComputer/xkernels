# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
"""Optional Triton-package redirection hook.

``xkernels`` Triton backends are written against the stock ``triton``
distribution (``import triton`` / ``import triton.language as tl``). A host that
builds against a *different* Triton package — e.g. tokenspeed, whose runtime
kernels use a vendored ``tokenspeed_triton`` exposed via
``tokenspeed_kernel._triton`` — needs these backends to bind that package
instead. Otherwise ``tl.dot`` rejects operands whose dtype objects come from the
foreign package ("Unsupported rhs dtype"), because it asserts that both operands
share the *same* dtype object and ``triton.bfloat16 is not
tokenspeed_triton.bfloat16``.

:func:`triton_import_ctx` returns a context manager to wrap the import of a
Triton-backend module so its module-level ``import triton`` binds the host's
package. It degrades to a no-op when no host redirect is importable — the
standalone case, where the kernels simply import and compile against whatever
``triton`` is installed. This keeps the kernel sources themselves package-clean;
only the ``ops/*/__init__`` import sites route through here.
"""

from __future__ import annotations

import contextlib

__all__ = ["triton_import_ctx"]


def triton_import_ctx():
    """Return a context manager that binds Triton imports to the host's package.

    Use around the import of any Triton-backend module so its module-level
    ``import triton`` / ``import triton.language as tl`` resolve to the host's
    Triton package. No-op when no host redirect helper is importable (standalone
    use with stock Triton).
    """
    try:
        from tokenspeed_kernel._triton import (
            redirect_triton_to_tokenspeed_triton,
        )
    except Exception:
        return contextlib.nullcontext()
    return redirect_triton_to_tokenspeed_triton()
