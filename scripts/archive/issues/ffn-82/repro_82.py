"""Standalone seeded repro for issue #82: fused_ffn + moe_sum_reduce numerics.

Runs the reference and each backend card per sweep point, dumps per-point
abs/rel err + NaN/Inf status + a few sample values. No pytest, no autotune pin.
"""
import sys
import torch
import torch.nn.functional as F

from xkernels.registry import get_spec, cards_for
from xkernels.registry.input_gen import generate_inputs
from xkernels.registry import load_shape_sweep


def _err(actual, expected):
    af = actual.detach().float()
    ef = expected.detach().float()
    diff = (af - ef).abs()
    denom = ef.abs().clamp_min(1e-8)
    return float(diff.max().item()), float((diff / denom).max().item())


def run_op(op_id, seed=1729):
    print("=" * 64)
    print(f"OP {op_id}")
    print("=" * 64)
    spec = get_spec(op_id)
    sweep = load_shape_sweep(spec.shape_sweep)
    bucket = cards_for(op_id)
    # reference callable
    from xkernels.registry import reference_callable
    ref_fn = reference_callable(op_id)
    print("backends:", list(bucket.keys()))
    for be, card in bucket.items():
        from xkernels.registry import backend_callable
        try:
            fn = backend_callable(op_id, be)
        except Exception as e:
            print(f"  [{be}] no callable: {e}")
            continue
        print(f"\n-- backend {be} (card {card.id}) --")
        for i, p in enumerate(sweep):
            inp = generate_inputs(op_id, p, seed, "cuda")
            try:
                ref = ref_fn(**inp)
                got = fn(**inp)
            except Exception as e:
                print(f"    pt{i} {p}: RAISED {type(e).__name__}: {e}")
                continue
            # normalize to list
            ref_l = ref if isinstance(ref, list) else [ref]
            got_l = got if isinstance(got, list) else [got]
            if len(ref_l) != len(got_l):
                print(f"    pt{i} {p}: OUTPUT COUNT ref={len(ref_l)} got={len(got_l)}")
                continue
            allabs, allrel = 0.0, 0.0
            naninf = []
            for j,(r,g) in enumerate(zip(ref_l,got_l)):
                a, rel = _err(g, r)
                allabs = max(allabs, a); allrel = max(allrel, rel)
                naninf.append(f"out{j}:nan={bool(g.isnan().any())}/inf={bool(g.isinf().any())}")
            print(f"    pt{i} {p}: max_abs={allabs:.4e} max_rel={allrel:.4e} {' '.join(naninf)}")


for op in sys.argv[1:]:
    run_op(op)
