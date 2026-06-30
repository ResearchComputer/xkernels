#!/bin/bash
# SPDX-License-Identifier: MIT
# HIP-graph capture probe for RCCL collectives on beverin (issue #12 deep-dive).
# Runs meta/benchmarks/probe_graph_capture.py once per PROBE_CFG in $CFGS, all inside
# one allocation. Override NODES / CFGS / PROBE_MODE / NUMEL at submit time:
#
#   # watchdog isolation (1 node, xGMI only, no OFI):
#   sbatch --nodes=1 --export=ALL,CFGS="baseline wd_off" scripts/slurm/probe_graph_beverin.sbatch
#   # OFI isolation (2 nodes, cross-node CXI):
#   sbatch --nodes=2 --export=ALL,CFGS="wd_off wd_off_mrcache wd_off_no_devrdma wd_off_all" \
#          scripts/slurm/probe_graph_beverin.sbatch
#
#SBATCH --job-name=xk-probe
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --time=00:20:00
#SBATCH --output=probe-%j.out
#SBATCH --error=probe-%j.out

set -uo pipefail

REPO="${REPO:-/capstor/scratch/cscs/xyao/kernels}"
ENV_NAME="${ENV_NAME:-tokenspeed-rocm-aiter-myofi}"
OFI_PLUGIN_DIR="${OFI_PLUGIN_DIR:-/capstor/store/cscs/swissai/infra02/xyao/tokenspeed-beverin/aws-ofi-rccl/lib}"
CFGS="${CFGS:-baseline wd_off}"
PROBE_MODE="${PROBE_MODE:-full}"
NUMEL="${NUMEL:-7168}"

export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
BASE_PORT="${MASTER_PORT:-29577}"

echo "REPO=$REPO ENV=$ENV_NAME nodes=$SLURM_NNODES tasks=$SLURM_NTASKS MODE=$PROBE_MODE NUMEL=$NUMEL"
echo "CONFIGS=[$CFGS]"

port_i=0
for cfg in $CFGS; do
  # Fresh port per config: sequential sruns must not reuse the prior TCPStore
  # port (it lingers in TIME_WAIT and the next rendezvous hangs).
  export MASTER_PORT=$((BASE_PORT + port_i))
  port_i=$((port_i + 1))
  echo ""
  echo "################## PROBE_CFG=$cfg  (port $MASTER_PORT) ##################"
  timeout 180 srun --environment="$ENV_NAME" --cpu-bind=none bash -c '
    set -e
    unset ROCR_VISIBLE_DEVICES || true
    export LD_LIBRARY_PATH="'"$OFI_PLUGIN_DIR"':/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="'"$REPO"'/src:${PYTHONPATH:-}"
    export PROBE_CFG="'"$cfg"'"
    export PROBE_MODE="'"$PROBE_MODE"'"
    export NUMEL="'"$NUMEL"'"
    export MASTER_ADDR="'"$MASTER_ADDR"'"
    export MASTER_PORT="'"$MASTER_PORT"'"
    python -u "'"$REPO"'/meta/benchmarks/probe_graph_capture.py"
  ' || echo "[srun exited non-zero / timed out for cfg=$cfg]"
done
echo ""
echo "ALL CONFIGS DONE"
