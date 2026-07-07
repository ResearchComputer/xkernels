import torch
import triton
import triton.language as tl


@triton.jit
def _argmax_tie_test(out_ptr, BLOCK: tl.constexpr):
    # all-equal values: which index does tl.argmax return?
    p = tl.full([BLOCK], 0.5, dtype=tl.float32)
    idx = tl.argmax(p, axis=0)
    tl.store(out_ptr, idx.to(tl.int32))


@triton.jit
def _argmax_tie_test_neg(out_ptr, BLOCK: tl.constexpr):
    # tie between index 0 (value 0.5) and index 13 (value 0.5), rest 0.4
    ar = tl.arange(0, BLOCK)
    p = tl.where((ar == 0) | (ar == 13), 0.5, 0.4)
    idx = tl.argmax(p, axis=0)
    tl.store(out_ptr, idx.to(tl.int32))


if __name__ == "__main__":
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    _argmax_tie_test[(1,)](out, BLOCK=16)
    print("all-tied argmax:", out.item(), "(0=lowest-wins, 15=highest-wins)")
    _argmax_tie_test_neg[(1,)](out, BLOCK=16)
    print("tie@0,13 argmax:", out.item(), "(0=lowest-wins, 13=highest-wins)")
