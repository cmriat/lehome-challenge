#!/bin/zsh
# Pi0.5 Implicit V5 Multi-node Training Script for LeHome Challenge
# Usage: pixi run submit-multi implicit_v5 2 scripts_mz/train_pi05_implicit_v5_multi.sh [NUM_GPUS_PER_NODE]

set -uo pipefail

NUM_GPUS="${1:-8}"
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-2}}"
NODE_RANK="${SLURM_NODEID:-${MACHINE_RANK:-}}"
if [[ -z "${NODE_RANK}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        NODE_RANK=$((SLURM_JOB_ID - 1))
    else
        NODE_RANK=0
    fi
fi
MASTER_ADDR="${SLURM_JOB_FIRST_NODE_IP:-${MAIN_PROCESS_IP:-${MASTER_ADDR:-}}}"
MASTER_PORT="${MAIN_PROCESS_PORT:-${MASTER_PORT:-29500}}"
CONFIG_PATH="configs/train_pi05_implicit_v5_raw_reject_v4.yaml"
TRAIN_SCRIPT="scripts/train_pi05_implicit_v5.py"
PROJECT_DIR="/home/jovyan/code/vla/lehome-challenge"

if [[ -z "$MASTER_ADDR" ]]; then
    echo "Error: SLURM_JOB_FIRST_NODE_IP not set. submit-multi must provide --first-node-ip." >&2
    exit 1
fi

LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_pi05_implicit_v5_multi_node${NODE_RANK}_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

exec > >(stdbuf -oL -eL tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Pi0.5 Implicit V5 Multi-node Training: LeHome Challenge"
echo "Job started at: $(date)"
echo "Node log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "Train script: $TRAIN_SCRIPT"
echo "Nodes: $NNODES | GPUs per node: $NUM_GPUS | Total GPUs: $((NNODES * NUM_GPUS))"
echo "Node rank: $NODE_RANK | Master: $MASTER_ADDR:$MASTER_PORT"
echo "=============================================="

export CUDA_LAUNCH_BLOCKING=0
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
export CONDA_OVERRIDE_CUDA=12.9
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export ACCELERATE_LOG_LEVEL="info"
export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export WORLD_SIZE="$(( NNODES * NUM_GPUS ))"

cd "$PROJECT_DIR"

OUTPUT_DIR=$(grep -A0 'output_dir:' "$CONFIG_PATH" | awk '{print $2}')
if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
    if [[ ! -d "$OUTPUT_DIR/checkpoints" ]]; then
        echo "Removing stale output dir from failed run: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    else
        echo "WARNING: $OUTPUT_DIR has checkpoints. Add --resume=true or change output_dir."
    fi
fi

set +u
pixi run torchrun \
    --nnodes="$NNODES" \
    --nproc-per-node="$NUM_GPUS" \
    --node-rank="$NODE_RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --config_path="$PROJECT_DIR/$CONFIG_PATH"
EXIT_CODE=$?
set -u

echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
