#!/usr/bin/env bash
# Parallel Eval for a single implicit checkpoint.
# Usage:
#   bash scripts_mz/eval_implicit_parallel.sh \
#     --policy-path ckpts/pi05_implicit_v1/step50k \
#     --gpus 0,1,2,3 \
#     --dataset-root Datasets/example/four_types_merged \
#     --headless

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DEFAULT_DATASET_ROOT="Datasets/example/four_types_merged"
DEFAULT_GPUS="4,5,6,7"
DEFAULT_DEVICE="cpu"
DEFAULT_NUM_EPISODES="20"
DEFAULT_MAX_STEPS="600"
DEFAULT_TASK="LeHome-BiSO101-Direct-Garment-v2"
DEFAULT_TASK_DESCRIPTION="fold the garment on the table"
DEFAULT_GARMENT_CFG_BASE_PATH="Assets/objects/Challenge_Garment"
DEFAULT_PARTICLE_CFG_PATH="source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml"
DEFAULT_LOG_DIR="logs/eval_implicit"
DEFAULT_VIDEO_DIR="videos/eval_implicit"

POLICY_PATH="/home/maoz/code/lehome-challenge/ckpts/pi05_implicit_v1/step50k"
GPU_CSV="${GPUS:-$DEFAULT_GPUS}"
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
NUM_EPISODES="${NUM_EPISODES:-$DEFAULT_NUM_EPISODES}"
MAX_STEPS="${MAX_STEPS:-$DEFAULT_MAX_STEPS}"
TASK="${TASK:-$DEFAULT_TASK}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-$DEFAULT_TASK_DESCRIPTION}"
GARMENT_CFG_BASE_PATH="${GARMENT_CFG_BASE_PATH:-$DEFAULT_GARMENT_CFG_BASE_PATH}"
PARTICLE_CFG_PATH="${PARTICLE_CFG_PATH:-$DEFAULT_PARTICLE_CFG_PATH}"
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"
VIDEO_DIR="${VIDEO_DIR:-$DEFAULT_VIDEO_DIR}"
SAVE_VIDEO=0 # 0=不保存视频, 1=保存视频

EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  bash scripts_mz/eval_implicit_parallel.sh --policy-path <ckpt_dir> [options] [-- extra_eval_args]

Required:
  --policy-path PATH         Path to checkpoint directory to evaluate

Options:
  --gpus IDS                 Comma-separated 4 GPU ids (default: 0,1,2,3)
  --dataset-root PATH        Dataset root (default: Datasets/example/four_types_merged)
  --device DEVICE            Eval device passed to scripts.eval (default: cpu)
  --num-episodes N           Episodes per garment (default: 10)
  --max-steps N              Max steps per episode (default: 600)
  --task NAME                Task name
  --task-description TEXT    Task description
  --garment-cfg-base-path P  Garment config base path
  --particle-cfg-path P      Particle config path
  --log-dir PATH             Log directory (default: logs/eval_implicit_parallel)
  --save-video               Save evaluation videos (default: enabled)
  --no-save-video            Disable evaluation video saving
  --video-dir PATH           Video directory root (default: videos/eval_implicit)
  -h, --help                 Show this help

Examples:
  bash scripts_mz/eval_implicit_parallel.sh \
    --policy-path ckpts/pi05_implicit_v1/step50k \
    --gpus 0,1,2,3 \
    --dataset-root Datasets/example/four_types_merged \
    --headless

  bash scripts_mz/eval_implicit_parallel.sh \
    --policy-path ckpts/pi05_implicit_v2/step50k \
    --gpus 4,5,6,7 \
    --num-episodes 20 \
    --headless --seed 7
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)
            POLICY_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPU_CSV="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --task-description)
            TASK_DESCRIPTION="$2"
            shift 2
            ;;
        --garment-cfg-base-path)
            GARMENT_CFG_BASE_PATH="$2"
            shift 2
            ;;
        --particle-cfg-path)
            PARTICLE_CFG_PATH="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO=1
            shift
            ;;
        --no-save-video)
            SAVE_VIDEO=0
            shift
            ;;
        --video-dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$POLICY_PATH" ]]; then
    echo "[错误] --policy-path 是必填参数。" >&2
    usage
    exit 1
fi

if [[ ! -d "$POLICY_PATH" ]]; then
    echo "[错误] checkpoint 目录不存在: $POLICY_PATH" >&2
    exit 1
fi

if [[ ! -f "$POLICY_PATH/config.json" ]]; then
    echo "[错误] checkpoint 目录缺少 config.json: $POLICY_PATH" >&2
    exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "[错误] dataset_root 不存在: $DATASET_ROOT" >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
if [[ ${#GPU_IDS[@]} -ne 4 ]]; then
    echo "[错误] --gpus 必须提供恰好 4 张卡，例如: --gpus 0,1,2,3" >&2
    exit 1
fi

extract_implicit_version() {
    python - "$POLICY_PATH/config.json" <<'PY'
import json
import sys

config_path = sys.argv[1]
try:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    variant = data.get("implicit_conditioning", {}).get("variant")
    if variant:
        print(f"implicit_{variant}")
    else:
        print("implicit_unknown")
except Exception:
    print("implicit_unknown")
PY
}

sanitize_path_component() {
    local value="$1"
    value="${value//\//_}"
    value="${value// /_}"
    echo "$value"
}

GARMENTS=(top_long top_short pant_long pant_short)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
POLICY_NAME="$(basename "$POLICY_PATH")"
POLICY_PARENT="$(basename "$(dirname "$POLICY_PATH")")"
IMPLICIT_VERSION="$(extract_implicit_version)"
CHECKPOINT_NAME="$(sanitize_path_component "${IMPLICIT_VERSION}_${POLICY_PARENT}_${POLICY_NAME}")"
RUN_LOG_DIR="$LOG_DIR/$CHECKPOINT_NAME/$TIMESTAMP"
RUN_VIDEO_DIR="$VIDEO_DIR/$CHECKPOINT_NAME/$TIMESTAMP"
mkdir -p "$RUN_LOG_DIR"
if [[ "$SAVE_VIDEO" == "1" ]]; then
    mkdir -p "$RUN_VIDEO_DIR"
fi

# Stub zenity to prevent headless IsaacSim from blocking on dialog popups
ZENITY_STUB_DIR="$SCRIPT_DIR/.zenity_stub"
mkdir -p "$ZENITY_STUB_DIR"
if [[ ! -x "$ZENITY_STUB_DIR/zenity" ]]; then
    printf '%s\n' '#!/bin/sh' 'exit 0' > "$ZENITY_STUB_DIR/zenity"
    chmod +x "$ZENITY_STUB_DIR/zenity"
fi

PIDS=()
PID_LABELS=()
PID_LOGS=()

print_header() {
    echo "============================================================"
    echo "Parallel Eval - Single Checkpoint / 4 Garments / 4 GPUs"
    echo "  Checkpoint:   $POLICY_PATH"
    echo "  Dataset Root: $DATASET_ROOT"
    echo "  GPUs:         $GPU_CSV"
    echo "  Device:       $DEVICE"
    echo "  Episodes:     $NUM_EPISODES"
    echo "  Max Steps:    $MAX_STEPS"
    echo "  Checkpoint:   $CHECKPOINT_NAME"
    echo "  Log Dir:      $RUN_LOG_DIR"
    if [[ "$SAVE_VIDEO" == "1" ]]; then
        echo "  Video Dir:    $RUN_VIDEO_DIR"
    fi
    echo "============================================================"
    echo ""
    printf '%-12s %-8s %s\n' "Garment" "GPU" "Log"
    printf '%-12s %-8s %s\n' "------------" "--------" "------------------------------"
    for idx in "${!GARMENTS[@]}"; do
        local garment="${GARMENTS[$idx]}"
        local gpu="${GPU_IDS[$idx]}"
        local log_file="$RUN_LOG_DIR/${garment}.log"
        printf '%-12s %-8s %s\n' "$garment" "$gpu" "$log_file"
    done
    echo ""
}

launch_eval() {
    local garment="$1"
    local gpu_id="$2"
    local log_file="$3"

    (
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        export LD_LIBRARY_PATH="$PROJECT_DIR/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
        export LEHOME_DISABLE_KEYBOARD=1
        export TRANSFORMERS_OFFLINE=1
        export HF_HUB_OFFLINE=1
        export PATH="$ZENITY_STUB_DIR:$PATH"
        CMD=(
            pixi run python -m scripts.eval
            --policy_type lerobot
            --policy_path "$POLICY_PATH"
            --dataset_root "$DATASET_ROOT"
            --garment_type "$garment"
            --num_episodes "$NUM_EPISODES"
            --max_steps "$MAX_STEPS"
            --task "$TASK"
            --task_description "$TASK_DESCRIPTION"
            --garment_cfg_base_path "$GARMENT_CFG_BASE_PATH"
            --particle_cfg_path "$PARTICLE_CFG_PATH"
            --enable_cameras
            --device "$DEVICE"
            --headless
            --use_random_seed
        )
        if [[ "$SAVE_VIDEO" == "1" ]]; then
            CMD+=(--save_video --video_dir "$RUN_VIDEO_DIR/$garment")
        fi
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            CMD+=("${EXTRA_ARGS[@]}")
        fi
        exec "${CMD[@]}"
    ) >"$log_file" 2>&1 &

    PIDS+=("$!")
    PID_LABELS+=("$garment@gpu$gpu_id")
    PID_LOGS+=("$log_file")
}

wait_all() {
    local failed=0
    for idx in "${!PIDS[@]}"; do
        local pid="${PIDS[$idx]}"
        local label="${PID_LABELS[$idx]}"
        local log_file="${PID_LOGS[$idx]}"
        if wait "$pid"; then
            echo "[完成] $label"
        else
            echo "[失败] $label  日志: $log_file" >&2
            failed=1
        fi
    done
    return "$failed"
}

print_header

for idx in "${!GARMENTS[@]}"; do
    garment="${GARMENTS[$idx]}"
    gpu_id="${GPU_IDS[$idx]}"
    log_file="$RUN_LOG_DIR/${garment}.log"
    launch_eval "$garment" "$gpu_id" "$log_file"
    echo "[启动] $garment -> GPU $gpu_id"
done

echo ""
echo "等待所有评估任务完成..."
if ! wait_all; then
    echo "[错误] 存在评估任务失败。" >&2
    exit 1
fi

echo "[完成] 所有 garment 评估完成。"