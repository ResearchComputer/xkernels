#!/usr/bin/env python
"""Inspect the CUTE DSL compile options surface to learn how to target sm_121.

Reads signatures/docstrings (not memory) so the smoke kernel uses the real API.
"""
from __future__ import annotations

import inspect


def main() -> int:
    import cutlass
    import cutlass.cute as cute
    import cutlass.cutlass_dsl as cdsl

    print("=== cutlass.cutlass_dsl names ===")
    print([n for n in dir(cdsl) if not n.startswith("_")][:40])

    print("\n=== cute.compile signature ===")
    try:
        print(str(inspect.signature(cute.compile))[:1000])
    except Exception as e:  # noqa: BLE001
        print("sig err", e)
    print((inspect.getdoc(cute.compile) or "")[:700])

    # GPUArch usage: how do callers pass it?
    print("\n=== where is GPUArch / cubin-chip passed? grep cute top-level ===")
    import os
    root = os.path.dirname(cute.__file__)
    print("cute pkg root:", root)

    print("\n=== Compile options: default arch derivation ===")
    # look for how the current device cc becomes the default chip
    import subprocess
    out = subprocess.run(
        ["grep", "-rln", "cubin-chip\|GPUArch\|get_device_cc\|cuda_get_device", root],
        capture_output=True, text=True,
    )
    print(out.stdout[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
