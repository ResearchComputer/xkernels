#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# Push the local tree to bristen and run an interactive command on an A100
# compute node INSIDE the NGC PyTorch container (torch + triton + ncu + nsys).
#
# bristen's login node has no GPU and no CUDA toolkit, so unlike beverin every
# command must `srun` to a compute node inside a container — this wrapper does
# that for you.
#
# Usage:
#   scripts/run-on-bristen.sh <command> [args...]
#
# Examples:
#   scripts/run-on-bristen.sh python3 -u benchmarks/bench_all.py
#   scripts/run-on-bristen.sh python3 -u tests/test_mhc_pre_post.py
#   scripts/run-on-bristen.sh python3 -c "import torch;print(torch.cuda.get_device_name(0), torch.__version__)"
#
# Env overrides:
#   BRISTEN_IMAGE      container image      (default nvcr.io/nvidia/pytorch:24.10-py3)
#   BRISTEN_PARTITION  slurm partition      (default normal)
#   BRISTEN_ACCOUNT    slurm account        (default a-infra02)
#   BRISTEN_GPU        --gpus-per-node      (default 1)
#   BRISTEN_TIME       wall time            (default 00:10:00)

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: scripts/run-on-bristen.sh <command> [args...]" >&2
    exit 1
fi

REPO_REMOTE="/capstor/scratch/cscs/xyao/xkernels"
IMAGE="${BRISTEN_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
PARTITION="${BRISTEN_PARTITION:-normal}"
ACCOUNT="${BRISTEN_ACCOUNT:-a-infra02}"
GPUS="${BRISTEN_GPU:-1}"
TIME="${BRISTEN_TIME:-00:10:00}"

echo "[rcc] pushing to bristen..."
rcc --profile bristen push

echo "[rcc] srun on bristen (image=$IMAGE partition=$PARTITION gpu=$GPUS time=$TIME)"
# rcc execs under a remote login shell; the trailing `bash "$@"` repasses the
# original args through the in-container `bash -lc` as "$@".
rcc --profile bristen run -- \
    srun -A "$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 \
         --gpus-per-node="$GPUS" --time="$TIME" \
         --container-image="docker://$IMAGE" \
         --container-mounts="/capstor:/capstor,/iopsstor:/iopsstor" \
         bash -lc "cd '$REPO_REMOTE' && export PYTHONPATH='$REPO_REMOTE/src:\${PYTHONPATH:-}' && \"\$@\"" bash "$@"
