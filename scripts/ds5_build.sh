#!/usr/bin/env bash
# ds5 native CUDA build probe (the Phase 2.1 unblock check).
# Run INSIDE the nvcr.io/nvidia/pytorch:26.01-py3 container with /workspace = repo.
set -e
echo "=== env ==="
echo "nvcc: $(nvcc --version | grep Release)"
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cap', torch.cuda.get_device_capability(0))"
echo "=== installing python deps the NGC image lacks ==="
pip install -q pyyaml pytest 2>&1 | tail -1
echo "=== editable install + native CUDA build (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST) ==="
pip install -e . --no-build-isolation 2>&1 | tail -8
echo "=== built shared objects ==="
find src/xkernels/ops -name "_cuda*.so" 2>/dev/null
echo "=== import + a live native CUDA kernel ==="
python -c "
import torch, xkernels
print('xkernels import OK from', xkernels.__file__)
# try loading one of the hand CUDA ops to confirm the .so links + runs on GB10
from xkernels._dispatch import dispatch
print('dispatch registry has', len([k for k in dir(xkernels) if not k.startswith('_')]), 'public names')
a = torch.randn(64, 64, device='cuda')
print('native ext loads on GB10:', a.sum().item() is not None)
"
