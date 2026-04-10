#!/bin/zsh
# Pi0.5 Delta Action Training Script for LeHome Challenge
# Usage: pixi run submit delta scripts/train_pi05_delta.sh [NUM_GPUS]
#
# Examples:
#   pixi run submit delta scripts/train_pi05_delta.sh        # 8 GPUs (default)
#   pixi run submit delta scripts/train_pi05_delta.sh 4      # 4 GPUs

set -uo pipefail

# ╔════════════════════════════════════════════════════════════════════╗
# ║  >>> CONFIGURATION <<<                                            ║
# ╚════════════════════════════════════════════════════════════════════╝

NUM_GPUS="${1:-8}"
CONFIG_PATH="configs/train_pi05_delta.yaml"
PROJECT_DIR="/home/jovyan/code/vla/lehome-challenge"

# ============== Log Configuration ==============
LOG_DIR="${PROJECT_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_pi05_delta_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "Pi0.5 Delta Action Training: LeHome Challenge"
echo "Job started at: $(date)"
echo "Log file: $LOG_FILE"
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# ============== Environment Variables ==============
export CUDA_LAUNCH_BLOCKING=0
export TORCH_MULTIPROCESSING_START_METHOD=spawn
export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_QPS_PER_CONNECTION="2"
export NCCL_IB_TIME_OUT="22"
export CONDA_OVERRIDE_CUDA=12.9
export PYTORCH_ALLOC_CONF=expandable_segments:True
# export HF_ENDPOINT=https://hf-mirror.com
# export HF_HOME="/data/hf"
# export HF_HUB_OFFLINE=1  # Enable only if model is pre-cached in HF_HOME

cd "$PROJECT_DIR"

# ============== Clean stale output from failed runs ==============
OUTPUT_DIR=$(grep -A0 'output_dir:' "$CONFIG_PATH" | awk '{print $2}')
if [[ -n "$OUTPUT_DIR" && -d "$OUTPUT_DIR" ]]; then
    # Only remove if no valid checkpoint exists (i.e. previous run failed before saving)
    if [[ ! -d "$OUTPUT_DIR/checkpoints" ]]; then
        echo "Removing stale output dir from failed run: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    else
        echo "WARNING: $OUTPUT_DIR has checkpoints. Add --resume=true or change output_dir."
    fi
fi

# ============== Training ==============
# Uses custom train_pi05_delta.py instead of lerobot-train
# to inject DeltaActionProcessorStep into the preprocessor pipeline.
EXIT_CODE=0
pixi run accelerate launch \
    --mixed_precision=bf16 \
    --num_processes="$NUM_GPUS" \
    scripts/train_pi05_delta.py \
    --config_path="$CONFIG_PATH" \
    || EXIT_CODE=$?

echo "=============================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE

# Notes:
# - To resume training, add: --resume=true
# - To override batch_size: --batch_size=8
# - To override steps: --steps=30000
