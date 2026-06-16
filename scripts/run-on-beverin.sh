#!/usr/bin/env bash
# Push the local tree to beverin and run an interactive command inside the
# tokenspeed-rocm-aiter-myofi container environment.
#
# Usage:
#   scripts/run-on-beverin.sh <command>
#
# Examples:
#   scripts/run-on-beverin.sh python3 -u benchmarks/bench_moe_sum_reduce.py
#   scripts/run-on-beverin.sh python3 -u tests/test_mhc_pre_post.py
#   scripts/run-on-beverin.sh srun --environment=tokenspeed-rocm-aiter-myofi --partition=mi300 --gpus-per-node=1 --time=00:10:00 bash -c 'cd /capstor/scratch/cscs/xyao/xkernels && python3 -u benchmarks/bench_all.py'

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: scripts/run-on-beverin.sh <command> [args...]" >&2
    exit 1
fi

REPO_REMOTE="/capstor/scratch/cscs/xyao/xkernels"

echo "[rcc] pushing to beverin..."
rcc push

echo "[rcc] running: $*"
# rcc execs the command directly, so we explicitly invoke bash and cd to the repo.
# bash -lc '...' bash "$@" passes the original args through $1, $2, ...
rcc run -- bash -lc "cd '$REPO_REMOTE' && export REPO='$REPO_REMOTE' PYTHONPATH='$REPO_REMOTE/src:\${PYTHONPATH:-}' && \"\$@\"" bash "$@"
