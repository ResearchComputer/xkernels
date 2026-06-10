# xkernels

Customized compute kernels across hardware vendors (NVIDIA, AMD, …) and kernel
types (FFN, MoE, comm, …), with a uniform PyTorch API, automatic backend
dispatch, and a correctness + benchmark harness.

## Install

```bash
pip install -e ".[dev]"          # pure-Python (reference + triton if present)
XKERNELS_FORCE_BUILD=1 pip install -e .   # also build CUDA/HIP extensions
```

Triton/CUDA backends are optional; the package runs on the pure-torch reference
path anywhere.

## Usage

```python
import torch
from xkernels import fused_ffn

y = fused_ffn(x, w_gate, w_up, w_down)            # backend="auto"
y = fused_ffn(x, w_gate, w_up, w_down, backend="triton")  # force a backend
```

```python
from xkernels import fused_moe_int4_w4a16  # INT4 W4A16 grouped fused-MoE GEMM

out = fused_moe_int4_w4a16(A, packed, scale, topk_ids, topk_w, group_size=32)
```

Override globally with `XKERNELS_BACKEND=reference|triton|cuda|hip`.

## Layout

- `src/xkernels/ops/<type>/` — kernels by type; each has `reference.py`,
  `interface.py`, and per-backend subdirs (`triton/`, `cuda/`).
- `src/xkernels/_dispatch.py` — backend registry + selection.
- `tests/`, `benchmarks/`, `examples/` — harness and demos.

See `docs/adding-a-kernel.md` to extend. Design: `docs/superpowers/specs/`.
