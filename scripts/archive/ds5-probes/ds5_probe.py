#!/usr/bin/env python
"""One-shot probe: what CUTE DSL surface does the installed nvidia-cutlass give us?

Run on ds5 (or any box) inside the project venv. Prints importability + the
top-level module surface so we know whether to target `pycute` or `cutlass.cute`.
"""
from __future__ import annotations

import importlib
import json
import os
import sys


def surface(modname: str) -> list[str]:
    mod = importlib.import_module(modname)
    d = os.path.dirname(getattr(mod, "__file__", "") or "")
    if not d:
        return []
    return sorted(x[:-3] for x in os.listdir(d) if x.endswith(".py") and not x.startswith("_"))


def main() -> int:
    print("=== python / torch ===")
    import torch  # noqa: F401

    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| cuda avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device", torch.cuda.get_device_name(0))

    print("\n=== candidate imports ===")
    for m in ("pycute", "cutlass", "cutlass.cute", "cutlass_cute", "cutlass_cppgen"):
        try:
            mod = importlib.import_module(m)
            print(f"  OK   {m:16s} -> {getattr(mod, '__file__', '?')}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {m:16s} -> {type(e).__name__}: {e}")

    print("\n=== pycute surface ===")
    try:
        print("  ", surface("pycute"))
    except Exception as e:  # noqa: BLE001
        print("  (no pycute)", e)

    print("\n=== version info ===")
    try:
        import importlib.metadata as md

        for d in ("nvidia-cutlass",):
            try:
                print(f"  {d}: {md.version(d)}")
            except Exception as e:  # noqa: BLE001
                print(f"  {d}: {e}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
