#!/bin/bash
# SPDX-License-Identifier: MIT
# 2-node x 4-GPU MI300A (gfx942) eval of the hierarchical all-reduce (issue #12)
# on CSCS Alps "beverin". Real RCCL across the Slingshot/CXI fabric.
#
# Submit from the repo on beverin (env vars override the defaults):
#   sbatch scripts/slurm/bench_allreduce_beverin.sbatch
#   ENV_NAME=tokenspeed-rocm-aiter sbatch scripts/slurm/bench_allreduce_beverin.sbatch   # Socket fallback
#
#SBATCH --job-name=xk-allreduce
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=allreduce-%j.out
#SBATCH --error=allreduce-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"
# From-source aws-ofi-rccl plugin (built --with-rocm=7.2). RCCL's NCCL_NET_PLUGIN=ofi
# must find librccl-net-ofi.so here; the myofi EDF no longer sets this LD path.
OFI_PLUGIN_DIR="${OFI_PLUGIN_DIR:-/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib}"
# Tunables (override at submit time via --export):
SIZES="${SIZES:-1 2 4 8 16}"          # batch sizes (rows of `hidden`); big -> bandwidth regime
ITERS="${ITERS:-100}"
WARMUP="${WARMUP:-30}"
GRAPH="${GRAPH:-0}"                    # 1 -> also time HIP-graph-captured replay
# Watchdog-disable env for graph capture (see scripts/slurm/probe_graph_beverin.sbatch).
# Harmless when GRAPH=0. Empty by default so eager runs match the EDF exactly.
CAPTURE_ENV="${CAPTURE_ENV:-}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29577}"
# Confirm RCCL splits into the 4+4 xGMI islands instead of a flat 8-way.
export NCCL_TOPO_DUMP_FILE="${REPO}/rccl-topo-${SLURM_JOB_ID}.xml"

GRAPH_FLAG=""; [ "$GRAPH" = "1" ] && GRAPH_FLAG="--graph"
echo "REPO=$REPO ENV=$ENV_NAME MASTER_ADDR=$MASTER_ADDR nodes=$SLURM_NNODES tasks=$SLURM_NTASKS"
echo "SIZES=[$SIZES] ITERS=$ITERS WARMUP=$WARMUP GRAPH=$GRAPH CAPTURE_ENV=[$CAPTURE_ENV]"

srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
  set -e
  unset ROCR_VISIBLE_DEVICES || true
  # Point RCCL at the from-source ROCm-7 OFI plugin + the ROCm runtime it links.
  # (The myofi EDF currently omits this LD path, so RCCL cannot find the ofi
  #  plugin without it -> "Failed to initialize any NET plugin".)
  export LD_LIBRARY_PATH="'"$OFI_PLUGIN_DIR"':/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
  [ -n "'"$CAPTURE_ENV"'" ] && eval "export '"$CAPTURE_ENV"'"
  python -u "'"$REPO"'/meta/benchmarks/bench_hierarchical_all_reduce.py" \
      --ranks-per-node 4 --hidden 7168 --sizes '"$SIZES"' \
      --iters '"$ITERS"' --warmup '"$WARMUP"' '"$GRAPH_FLAG"'
'
