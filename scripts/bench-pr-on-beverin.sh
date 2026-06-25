#!/usr/bin/env bash
# Push the local tree to beverin and run a single benchmark on an MI300A node.
#
#   scripts/bench-pr-on-beverin.sh benchmarks/pr/pr61_sparse_mla_decode.py [TIME]
#
# Assumes the relevant PR branch is already checked out locally.
set -euo pipefail

PY="${1:?usage: $0 <benchmarks/xxx.py> [minutes]}"
MINS="${2:-15}"
REPO_REMOTE="/capstor/scratch/cscs/xyao/xkernels"
ENV_NAME="tokenspeed-rocm-aiter-myofi"

# Normalize to a path relative to the repo root, which is what exists on the remote.
REPO_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
PY_REL="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"
PY_REL="${PY_REL#$REPO_LOCAL/}"

echo "[rcc] pushing local tree (branch: $(git -C "$REPO_LOCAL" rev-parse --abbrev-ref HEAD)) to beverin..."
rcc push

echo "[rcc] running on MI300A: $PY_REL"
rcc run -- srun --environment="$ENV_NAME" --partition=mi300 --gpus-per-node=1 \
       --time="00:${MINS}:00" --chdir="$REPO_REMOTE" bash -c "
  unset ROCR_VISIBLE_DEVICES || true
  export LD_LIBRARY_PATH=/opt/rocm/lib:\${LD_LIBRARY_PATH:-}
  export PYTHONPATH=$REPO_REMOTE/src:\${PYTHONPATH:-}
  cd '$REPO_REMOTE'
  python3 -u '$PY_REL'
"
