#!/usr/bin/env bash
# 批量评估所有衣物类型的脚本
# 用法: CUDA_VISIBLE_DEVICES=0 bash scripts/eval_all_types.sh [可选参数覆盖默认值]
#
# 默认配置:
#   --policy_path   /home/maozan/code/data/ckpts/pi05_step80k/pretrained_model
#   --dataset_root  Datasets/example/four_types_merged
#   --num_episodes  5
#   --video_dir     outputs/eval_videos
#
# 示例:
#   # 使用默认配置直接运行
#   bash scripts/eval_all_types.sh
#
#   # 保存视频
#   bash scripts/eval_all_types.sh --save_video
#
#   # 覆盖部分参数
#   bash scripts/eval_all_types.sh --num_episodes 10 --save_video --video_dir my_videos

set -euo pipefail

# 修复 libstdc++ 版本问题：使用 pixi 环境的库
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIXI_LIB_PATH="$SCRIPT_DIR/../.pixi/envs/default/lib"
if [ -d "$PIXI_LIB_PATH" ]; then
    export LD_LIBRARY_PATH="$PIXI_LIB_PATH:$LD_LIBRARY_PATH"
fi

# 无头服务器：禁用键盘输入（必须在 Python 启动前设置）
export LEHOME_DISABLE_KEYBOARD=1

# 捕获 Ctrl+C 信号，确保能正常退出
trap 'echo -e "\n收到中断信号，正在退出..."; exit 130' INT TERM

# 默认参数
DEFAULT_POLICY_PATH="/home/maozan/code/data/ckpts/pi05_step80k/pretrained_model"
DEFAULT_DATASET_ROOT="Datasets/example/four_types_merged"
DEFAULT_NUM_EPISODES=5
DEFAULT_VIDEO_DIR="outputs/eval_videos_pi05_step80k"

# 解析命令行参数，允许覆盖默认值
POLICY_PATH="$DEFAULT_POLICY_PATH"
DATASET_ROOT="$DEFAULT_DATASET_ROOT"
NUM_EPISODES="$DEFAULT_NUM_EPISODES"
VIDEO_DIR="$DEFAULT_VIDEO_DIR"
SAVE_VIDEO=true  # 默认保存视频
HEADLESS=true  # 默认启用无头模式（适用于服务器）
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --policy_path)
            POLICY_PATH="$2"
            shift 2
            ;;
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --save_video)
            SAVE_VIDEO=true
            shift
            ;;
        --video_dir)
            VIDEO_DIR="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        --no_headless)
            HEADLESS=false
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

GARMENT_TYPES=("top_long" "top_short" "pant_long" "pant_short")

# 构建 video 和 headless 参数
VIDEO_ARGS=()
if [ "$SAVE_VIDEO" = true ]; then
    VIDEO_ARGS=(--save_video --video_dir "$VIDEO_DIR")
fi

HEADLESS_ARGS=()
if [ "$HEADLESS" = true ]; then
    HEADLESS_ARGS=(--headless)
fi

echo "========================================"
echo "  Pi0.5 全类型评估"
echo "  Policy: $POLICY_PATH"
echo "  数据集: $DATASET_ROOT"
echo "  Episodes: $NUM_EPISODES"
echo "  无头模式: $HEADLESS"
echo "  保存视频: $SAVE_VIDEO${VIDEO_DIR:+ (→ $VIDEO_DIR)}"
echo "  衣物类型: ${GARMENT_TYPES[*]}"
echo "  额外参数: ${EXTRA_ARGS[*]:-无}"
echo "========================================"

failed_types=()

for type in "${GARMENT_TYPES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "  开始评估: $type"
    echo "----------------------------------------"

    if python -m scripts.eval \
        --policy_type lerobot \
        --garment_type "$type" \
        --enable_cameras \
        --device cpu \
        --policy_path "$POLICY_PATH" \
        --dataset_root "$DATASET_ROOT" \
        --num_episodes "$NUM_EPISODES" \
        "${HEADLESS_ARGS[@]}" \
        "${VIDEO_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"; then
        echo "[OK] $type 评估完成"
    else
        echo "[FAIL] $type 评估失败 (exit code: $?)"
        failed_types+=("$type")
    fi
done

echo ""
echo "========================================"
echo "  评估汇总"
echo "========================================"
echo "  总计: ${#GARMENT_TYPES[@]} 种类型"
echo "  成功: $(( ${#GARMENT_TYPES[@]} - ${#failed_types[@]} )) 种"
if [ ${#failed_types[@]} -gt 0 ]; then
    echo "  失败: ${failed_types[*]}"
    exit 1
else
    echo "  全部通过"
fi
