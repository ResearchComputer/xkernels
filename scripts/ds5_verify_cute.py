#!/usr/bin/env python
"""Run the real xkernels verify + verify_parity harness on the CUTE card, on ds5.

This is the actual gate (docs/library.md §5): verify() checks the card vs the op's
one reference across the mandatory shape sweep; verify_parity() checks the CUTE
card agrees with the triton card within cross_backend_rtol.
"""
from __future__ import annotations

from xkernels import verify, verify_parity

CARD = "mm_fp8_blockscale.cuda@1.0.0"
OP = "mm_fp8_blockscale@1.0.0"

print("=" * 70)
print(f"verify({CARD!r}, arch='nvidia_sm121')")
print("=" * 70)
r = verify(CARD, arch="nvidia_sm121")
c = r["correctness"]
print(f"  compiled          = {r['compiled']}")
print(f"  correctness.passed= {c['passed']}")
print(f"  max_abs_err       = {c['max_abs_err']:.3e}")
print(f"  max_rel_err       = {c['max_rel_err']:.3e}")
print(f"  n_points          = {c['n_points']}, failing = {len(c['failing_shapes'])}")
for f in c["failing_shapes"]:
    print(f"    FAIL point={f['point']} abs={f['abs_err']:.3e} rel={f['rel_err']:.3e} "
          f"(rtol={f['rtol']} atol={f['atol']})")
if "error" in r.get("artifacts", {}):
    print("  error:", r["artifacts"]["error"])

print()
print("=" * 70)
print(f"verify_parity({OP!r})")
print("=" * 70)
p = verify_parity(OP)
print(f"  agree                  = {p['agree']}")
print(f"  cross_backend_rtol     = {p['cross_backend_rtol']}")
print(f"  max_pairwise_rel_err   = {p['max_pairwise_rel_err']:.3e}")
print(f"  runnable backends      = {[k for k,v in p['per_backend_runnable'].items() if v]}")
for d in p["diverging"]:
    print(f"    DIVERGE pair={d['pair']} point={d['point']} rel={d['rel_err']:.3e}")
print(f"  errors                 = {p['errors']}")
