#!/usr/bin/env python
"""Find a SIMPLE tiled GEMM example in the installed cutlass DSL package, to learn
the smem/copy-atom/tile idioms for a tiled fp32 GEMM on sm_121. The torch-vendored
dense_blockscaled template is too advanced (SM100 tcgen05 + TMA + pipeline)."""
from __future__ import annotations
import os, re

ROOT = ".venv/lib/python3.12/site-packages/cutlass"

# Walk and score files by how 'simple tiled gemm' they look.
hits = []
for dirpath, _, files in os.walk(ROOT):
    for f in files:
        if not f.endswith(".py"):
            continue
        p = os.path.join(dirpath, f)
        try:
            txt = open(p, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        has_gemm = bool(re.search(r"\bgemm\b|\bmatmul\b|gemm_kernel|GemmKernel", txt, re.I))
        has_smem = bool(re.search(r"smem|cp_async|cpasync|make_tensor.*shared|TmaAsync|make_copy_atom", txt, re.I))
        has_mma = bool(re.search(r"mma|MMA|tiled_mma|make_mma|MatMul", txt, re.I))
        if has_gemm and (has_smem or has_mma):
            # skip the sm100 tcgen05 monster (we know it)
            score = (1 if has_smem else 0) + (1 if has_mma else 0)
            nlines = txt.count("\n")
            hits.append((score, nlines, p))

hits.sort(key=lambda h: (h[0], h[1]))  # simple (low score, short) first
print(f"{'lines':>6}  path")
for _, n, p in hits[:25]:
    print(f"{n:>6}  {p.replace(ROOT, '<cutlass>')}")
