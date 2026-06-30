#!/usr/bin/env python
"""Probe the installed CUTE DSL surface + locate bundled examples.

We do NOT guess the DSL API from memory (AGENTS.md: name things from the
skills + `meta/wiki/05-cutedsl-authoring.md`, not intuition). This prints what `cutlass.cute` actually exports
and finds example files shipped with nvidia-cutlass-dsl so we can model the
first xkernels CUTE card on a real, working kernel.
"""
from __future__ import annotations

import importlib
import os


def dump_surface(modname: str, limit: int = 60) -> None:
    mod = importlib.import_module(modname)
    public = [n for n in dir(mod) if not n.startswith("_")]
    print(f"  [{modname}] {len(public)} public names")
    print("   ", ", ".join(public[:limit]))


def main() -> int:
    import cutlass
    import cutlass.cute as cute

    print("=== versions ===")
    print("  cutlass", getattr(cutlass, "__version__", "?"),
          "| cute at", os.path.dirname(cute.__file__))

    print("\n=== cutlass top-level surface ===")
    dump_surface("cutlass")

    print("\n=== cutlass.cute surface ===")
    dump_surface("cutlass.cute")

    # Walk the installed nvidia_cutlass_dsl tree for examples / smoke tests.
    import cutlass.cute as cute  # noqa: F811
    root = os.path.dirname(os.path.dirname(os.path.dirname(cute.__file__)))
    # root ~= .../nvidia_cutlass_dsl/python_packages/cutlass  -> walk up to pkg root
    for _ in range(4):
        root = os.path.dirname(root)
    print("\n=== nvidia_cutlass_dsl package root ===")
    print("  ", root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__"}]
        for f in filenames:
            if f.endswith(".py") and ("example" in f.lower() or "smoke" in f.lower()
                                      or "gemm" in f.lower()):
                rel = os.path.relpath(os.path.join(dirpath, f), root)
                print("   ", rel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
