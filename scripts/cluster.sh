#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ResearchComputer
# cluster.sh — the ONE rcc wrapper: push the local tree to a cluster host and
# either run a command interactively or submit a SLURM job there.
#
# Replaces the four thin per-host shims run-on-{beverin,bristen}.sh +
# bench-on-{beverin,bristen}.sh, and adds ds5 coverage (it had no wrapper
# before). Host selection mirrors the rcc profiles in .rcc/config.toml:
#
#   beverin  AMD MI300A (gfx942)   — default; rocm uenv; direct rcc run
#   bristen  NVIDIA A100 (sm_80)   — NGC PyTorch container via srun --container-image
#   ds5      NVIDIA GB10 (sm_121)   — CUTE-DSL test bed; no SLURM (run only)
#
# Usage:
#   scripts/cluster.sh run   [--host beverin|bristen|ds5] [--] <command...>
#   scripts/cluster.sh submit [--host beverin|bristen]     [sbatch_script]
#
#   --host defaults to beverin. For `run`, `--` separates cluster.sh's flags
#   from your command; it is optional unless the command's first token looks
#   like a flag (e.g. starts with `-`).
#
# Examples:
#   # run interactively
#   scripts/cluster.sh run --host beverin -- python3 -u meta/benchmarks/bench_all.py
#   scripts/cluster.sh run --host bristen -- python3 -u meta/benchmarks/bench_all.py
#   scripts/cluster.sh run --host ds5     -- python -m xkernels.ops._cute_backend.smoke_vecadd
#
#   # submit a SLURM job (beverin / bristen only — ds5 has no SLURM)
#   scripts/cluster.sh submit --host beverin scripts/slurm/bench_all_beverin.sbatch
#   scripts/cluster.sh submit --host bristen                         # default: bench_all_bristen
#   KERNEL=dual_rmsnorm MODE=sq scripts/cluster.sh submit --host bristen scripts/slurm/profile_ncu_bristen.sbatch
#
# bristen `run` env overrides (the srun container + GPU allocation):
#   BRISTEN_IMAGE / BRISTEN_PARTITION / BRISTEN_ACCOUNT / BRISTEN_GPU / BRISTEN_TIME
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '/^set -euo/d'
}

# --- parse -----------------------------------------------------------------
SUBCMD="${1:-}"
[[ -n "$SUBCMD" ]] || { usage >&2; exit 1; }
shift

HOST="beverin"
CMD=()
SCRIPT=""
case "$SUBCMD" in
    run)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --host)    HOST="$2"; shift 2;;
                --host=*)  HOST="${1#--host=}"; shift;;
                --)        shift; CMD=("$@"); break;;
                *)         CMD=("$@"); break;;   # first non-flag starts the command
            esac
        done
        [[ ${#CMD[@]} -gt 0 ]] || { echo "run: no command given (use -- <command>)" >&2; exit 1; }
        ;;
    submit)
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --host)    HOST="$2"; shift 2;;
                --host=*)  HOST="${1#--host=}"; shift;;
                --)        shift; break;;
                -*)        echo "submit: unknown option: $1" >&2; exit 2;;
                *)         SCRIPT="$1"; shift;;
            esac
        done
        ;;
    help|-h|--help) usage; exit 0;;
    *) echo "unknown subcommand: $SUBCMD (use run|submit)" >&2; exit 2;;
esac

# --- per-host profile + repo + default job ---------------------------------
case "$HOST" in
    beverin) RCC=();                          REPO="/capstor/scratch/cscs/xyao/xkernels"; DEF_SCRIPT="scripts/slurm/bench_all_beverin.sbatch";;
    bristen) RCC=(--profile bristen);         REPO="/capstor/scratch/cscs/xyao/xkernels"; DEF_SCRIPT="scripts/slurm/bench_all_bristen.sbatch";;
    ds5)     RCC=(--profile ds5);             REPO="/local/home/xiayao/xkernels";          DEF_SCRIPT="";;
    *) echo "unknown host: $HOST (use beverin|bristen|ds5)" >&2; exit 2;;
esac

case "$SUBCMD" in
    # ---- run: push + interactive command on a compute node -----------------
    run)
        echo "[rcc] pushing to $HOST..."
        case "$HOST" in
            beverin)
                echo "[rcc] running on $HOST: ${CMD[*]}"
                rcc "${RCC[@]}" run -- bash -lc \
                    "cd '$REPO' && export REPO='$REPO' PYTHONPATH='$REPO/src:\${PYTHONPATH:-}' && \"\$@\"" \
                    bash "${CMD[@]}"
                ;;
            bristen)
                IMAGE="${BRISTEN_IMAGE:-nvcr.io/nvidia/pytorch:24.10-py3}"
                PARTITION="${BRISTEN_PARTITION:-normal}"
                ACCOUNT="${BRISTEN_ACCOUNT:-a-infra02}"
                GPUS="${BRISTEN_GPU:-1}"
                TIME="${BRISTEN_TIME:-00:10:00}"
                # bristen's login node has no GPU/CUDA toolkit, so every command
                # must srun onto a compute node inside the NGC container.
                echo "[rcc] srun on $HOST (image=$IMAGE partition=$PARTITION gpu=$GPUS time=$TIME)"
                rcc "${RCC[@]}" run -- \
                    srun -A "$ACCOUNT" --partition="$PARTITION" --nodes=1 --ntasks=1 \
                         --gpus-per-node="$GPUS" --time="$TIME" \
                         --container-image="docker://$IMAGE" \
                         --container-mounts="/capstor:/capstor,/iopsstor:/iopsstor" \
                         bash -lc "cd '$REPO' && export PYTHONPATH='$REPO/src:\${PYTHONPATH:-}' && \"\$@\"" \
                         bash "${CMD[@]}"
                ;;
            ds5)
                # ds5 is a bare Grace+Blackwell node: no SLURM, no container.
                # CUTE DSL JIT needs nvcc (CUDA_HOME); the project venv has the rest.
                echo "[rcc] running on $HOST: ${CMD[*]}"
                rcc "${RCC[@]}" run -- bash -lc \
                    "cd '$REPO' && export CUDA_HOME=/usr/local/cuda-13.0 && . .venv/bin/activate && \"\$@\"" \
                    bash "${CMD[@]}"
                ;;
        esac
        ;;

    # ---- submit: push + sbatch a job (beverin / bristen) -------------------
    submit)
        [[ -n "$DEF_SCRIPT" ]] || { echo "submit: $HOST has no SLURM (use 'run')" >&2; exit 1; }
        SCRIPT="${SCRIPT:-$DEF_SCRIPT}"
        [[ -f "$SCRIPT" ]] || { echo "Error: SLURM script not found: $SCRIPT" >&2; exit 1; }
        echo "[rcc] pushing to $HOST..."
        rcc "${RCC[@]}" push
        echo "[rcc] submitting $SCRIPT on $HOST (REPO=$REPO)..."
        JOBID=$(rcc "${RCC[@]}" run -- env REPO="$REPO" sbatch "$SCRIPT" | awk '{print $NF}')
        RCCP="${RCC[*]:-}"; RCCP="${RCCP:+$RCCP }"
        echo "[rcc] submitted job $JOBID"
        echo "[rcc] follow: rcc ${RCCP}run -- squeue -j $JOBID"
        echo "[rcc] output file is named per the sbatch's --output= (e.g. <prefix>-${JOBID}.out)"
        ;;
esac
