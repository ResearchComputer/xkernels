#!/usr/bin/env python
"""Introspect the CUTE DSL compile/launch surface to author the first sm_121 kernel
WITHOUT guessing the API from memory (AGENTS.md hard rule).

Prints signatures of the key entrypoints: cute.jit / cute.kernel / cute.compile /
LaunchConfig, plus how `arch` (the GPUArch compile option) is configured.
"""
from __future__ import annotations

import inspect


def show(modname: str, names: list[str]) -> None:
    mod = __import__(modname, fromlist=["*"])
    print(f"\n=== {modname} ===")
    for n in names:
        obj = getattr(mod, n, None)
        if obj is None:
            print(f"  {n}: MISSING")
            continue
        kind = type(obj).__name__
        doc = (inspect.getdoc(obj) or "").split("\n")[0][:90]
        sig = ""
        try:
            if callable(obj) or isinstance(obj, type):
                sig = str(inspect.signature(obj))[:500]
        except (TypeError, ValueError):
            sig = "<no sig>"
        print(f"  {n} [{kind}] {sig}")
        if doc:
            print(f"      doc: {doc}")


def main() -> int:
    import cutlass.cute as cute

    show("cutlass.cute", ["jit", "kernel", "struct", "compile", "LaunchConfig",
                          "GPUArch", "dsl_user_op", "arch", "range", "copy"])
    # GPUArch compile option: find how to set the target chip.
    print("\n=== GPUArch detail ===")
    ga = cute.GPUArch
    for n in dir(ga):
        if n.startswith("_"):
            continue
        try:
            print(f"  {n} = {getattr(ga, n)!r}"[:160])
        except Exception as e:  # noqa: BLE001
            print(f"  {n}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
