#!/usr/bin/env bash
# ds5 GPU runner — execute a command inside the NGC pytorch container with the
# GB10 (sm_121) passed through and the repo mounted at /workspace.
# Persistent across the repo mount: editable install + native ext live in the
# mounted tree, so repeat runs reuse the build/ cache.
#
# Usage (from ds5 host):
#   ~/xkernels/ds5_run.sh "python -m pytest tests/test_vkl_*.py -q"
#   ~/xkernels/ds5_run.sh "python -c 'import torch; print(torch.cuda.get_device_name(0))'"
#   XKERNELS_REBUILD=1 ~/xkernels/ds5_run.sh "..."   # force native ext rebuild
set -euo pipefail
CMD="${1:?usage: ds5_run.sh \"<command>\"}"
IMAGE="nvcr.io/nvidia/pytorch:26.01-py3"

# one-time-per-run in-container setup script
SETUP=$(cat <<'EOF'
set -e
[ -d /workspace/build ] || mkdir -p /workspace/build
# editable install if not already importable
python -c "import xkernels" 2>/dev/null || pip install -q -e /workspace --no-build-isolation
# build native CUDA ext (cached in /workspace/build). Rebuild if asked or if absent.
if [ "${XKERNELS_REBUILD:-0}" = "1" ] || ! python -c "import xkernels.ops.ffn.cuda._cuda" 2>/dev/null; then
  TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1}" python /workspace/setup.py build_ext --inplace >/dev/null 2>&1 || true
fi
cd /workspace
EOF
)

exec docker run --rm \
  --device /dev/nvidia0 --device /dev/nvidiactl \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-caps \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /usr/lib/aarch64-linux-gnu/libcuda.so.1:/usr/lib/aarch64-linux-gnu/libcuda.so.1:ro \
  -v /usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:/usr/lib/aarch64-linux-gnu/libnvidia-ml.so.1:ro \
  -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
  -v "$HOME/xkernels:/workspace" \
  -w /workspace \
  -e TORCH_CUDA_ARCH_LIST="12.1" \
  -e XKERNELS_FORCE_BUILD=1 \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  "$IMAGE" \
  bash -lc "$SETUP; exec bash -lc \"$CMD\""
