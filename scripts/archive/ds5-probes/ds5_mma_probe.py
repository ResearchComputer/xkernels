#!/usr/bin/env python
"""Discover the CUTE DSL MMA-atom catalog + arch coverage for sm_121 (GB10).

Decides the kernel strategy WITHOUT guessing:
  - If sm_121 supports a tensor-core MMA atom (m16n8k16 / wgmma / tcgen05), the
    card can use a real matrix engine.
  - If only the data-center tcgen05 path is exposed (and it's sm_100-gated), the
    honest ds5 card is the portable dequant-then-fp32-matmul (the op's parity
    target), correctly verified + parity-passed but not claiming native blockscaled.
"""
from __future__ import annotations

import os
import re


def main() -> int:
    import cutlass
    import cutlass.cute as cute

    print("=== cute.nvgpu submodules (MMA / copy engines) ===")
    import cutlass.cute.nvgpu as nvgpu
    print("  ", [n for n in dir(nvgpu) if not n.startswith("_")])

    print("\n=== tcgen05 (Blackwell data-center) present? ===")
    try:
        from cutlass.cute.nvgpu import tcgen05
        print("   tcgen05 OK:", [n for n in dir(tcgen05) if not n.startswith("_")][:25])
    except Exception as e:  # noqa: BLE001
        print("   tcgen05:", e)

    print("\n=== MMA atom helpers (make_trivial_tiled_mma / make_mma / TiledMma) ===")
    for n in ("make_trivial_tiled_mma", "make_mma", "TiledMma", "MmaAtom"):
        print(f"   {n}: {'yes' if hasattr(cute, n) else 'MISSING'}")

    # Scan the installed cute.nvgpu tree for which sm_xxx each MMA kind targets.
    root = os.path.dirname(cute.__file__)
    nvgpu_dir = os.path.join(root, "nvgpu")
    print(f"\n=== scanning {nvgpu_dir} for MMA-arch coverage ===")
    arch_hits: dict[str, set[str]] = {}
    mma_kinds = {"mma_m16n8": [], "wgmma": [], "tcgen05": []}
    for dp, dn, fns in os.walk(nvgpu_dir):
        dn[:] = [d for d in dn if d != "__pycache__"]
        for f in fns:
            if not f.endswith(".py"):
                continue
            try:
                txt = open(os.path.join(dp, f), encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for tok in re.findall(r"sm_(\d+)", txt):
                arch_hits.setdefault(tok, set()).add(f)
            if "wgmma" in txt:
                mma_kinds["wgmma"].append(f)
            if "tcgen05" in txt:
                mma_kinds["tcgen05"].append(f)
            if "mma.m16n8" in txt or "m16n8k" in txt:
                mma_kinds["mma_m16n8"].append(f)
    print("   files per MMA kind:")
    for k, fs in mma_kinds.items():
        print(f"     {k:12s}: {fs[:6]}")
    print("   sm_xxx -> #files touching it (MMA-relevant):")
    for tok in sorted(arch_hits, key=int):
        mark = "  <-- GB10" if tok in ("120", "121") else ""
        print(f"     sm_{tok:4s}: {len(arch_hits[tok]):2d} files{mark}")

    # The decisive test: can we BUILD a trivial tiled MMA on this device?
    print("\n=== build a trivial tiled MMA (m16n8k16, fp16/fp32) ===")
    try:
        if hasattr(cute, "make_trivial_tiled_mma"):
            mma = cute.make_trivial_tiled_mma((16, 8, 16))
            print("   make_trivial_tiled_mma((16,8,16)) OK:", type(mma).__name__)
        else:
            print("   no make_trivial_tiled_mma on cute top-level")
    except Exception as e:  # noqa: BLE001
        print(f"   FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
