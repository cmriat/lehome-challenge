#!/bin/zsh
# Pi0.5 Implicit V4 Training Script for LeHome Challenge
# Usage: pixi run submit implicit_v4 scripts_mz/train_pi05_implicit_v4.sh [NUM_GPUS]

set -uo pipefail

NUM_GPUS="${1:-8}"
CONFIG_PATH="configs/train_pi05_implicit_v4.yaml"
TRAIN_SCRIPT="scripts/train_pi05_implicit_v4.py"
PROJECT_DIR="/home/jovyan/code/vla/lehome-challenge"

LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_pi05_implicit_v4_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Pi0.5 Implicit V4 Training: LeHome Challenge"
echo "Job started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

export CUDA_LAUNCH_BLOCKING=0
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
export CONDA_OVERRIDE_CUDA=12.9
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

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

EXIT_CODE=0
pixi run accelerate launch \
    --mixed_precision=bf16 \
    --num_processes="$NUM_GPUS" \
    --main_process_port="$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --config_path="$PROJECT_DIR/$CONFIG_PATH" \
    || EXIT_CODE=$?

echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
