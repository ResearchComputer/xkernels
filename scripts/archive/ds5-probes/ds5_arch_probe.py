#!/usr/bin/env python
"""Decisive reachability check: does the installed CUTE DSL claim sm_120 (GB10)?

Dumps the `cutlass.cute.GPUArch` enum and greps the installed
nvidia-cutlass-dsl package for sm_100/120/90 token coverage, so we know whether
ds5 (GB10 = sm_120) can JIT a CUTE kernel at all — independent of the
torch-vendored blockscaled kernel which is hard-gated to sm_100/101/103.
"""
from __future__ import annotations

import os
import re


def main() -> int:
    import cutlass
    import cutlass.cute as cute

    print("=== cutlass.cute.GPUArch members ===")
    arch = cute.GPUArch
    members = [n for n in dir(arch) if not n.startswith("_")]
    for n in members:
        v = getattr(arch, n)
        print(f"   {n:28s} = {v!r}")

    # Locate the installed package and scan for sm_xxx tokens.
    pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(cute.__file__)))
    print("\n=== package root ===")
    print("  ", pkg_root)

    hits: dict[str, int] = {}
    for dirpath, dirnames, filenames in os.walk(pkg_root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in filenames:
            if not f.endswith((".py", ".cc", ".h", ".cu", ".txt")):
                continue
            try:
                txt = open(os.path.join(dirpath, f), encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for tok in re.findall(r"sm_(\d+)", txt):
                hits[tok] = hits.get(tok, 0) + 1
    print("\n=== sm_xxx token occurrences across the package ===")
    for tok in sorted(hits, key=int):
        marker = "  <-- ds5 (GB10)" if tok == "120" else ""
        print(f"   sm_{tok:3s}: {hits[tok]:5d}{marker}")

    # Current device's cc, for the record.
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"\n=== this device cc = {major}{minor} (sm_{major}{minor}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
