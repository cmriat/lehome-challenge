#!/usr/bin/env bash
# 单类型评估脚本
# 用法: bash scripts/eval_single_type.sh <garment_type> [可选参数]
#
# 参数:
#   garment_type   必需，可选值: top_long, top_short, pant_long, pant_short
#
# 默认配置:
#   --policy_path   /home/maozan/code/data/ckpts/pi05_step80k/pretrained_model
#   --dataset_root  Datasets/example/four_types_merged
#   --num_episodes  5
#   --video_dir     outputs/eval_videos_pi05_step80k
#
# 示例:
#   # 评估 top_long 类型
#   bash scripts/eval_single_type.sh top_long
#
#   # 评估 pant_short 类型，指定 GPU
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_single_type.sh pant_short
#
#   # 覆盖其他参数
#   CUDA_VISIBLE_DEVICES=0 bash scripts/eval_single_type.sh top_long --no_headless

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

# 检查必需参数
if [ $# -lt 1 ]; then
    echo "错误: 缺少 garment_type 参数"
    echo "用法: bash scripts/eval_single_type.sh <garment_type> [可选参数]"
    echo ""
    echo "可选的 garment_type: top_long, top_short, pant_long, pant_short"
    exit 1
fi

# 第一个参数是 garment_type
GARMENT_TYPE="$1"
shift

# 验证 garment_type
VALID_TYPES=("top_long" "top_short" "pant_long" "pant_short")
VALID=false
for t in "${VALID_TYPES[@]}"; do
    if [ "$GARMENT_TYPE" = "$t" ]; then
        VALID=true
        break
    fi
done

if [ "$VALID" = false ]; then
    echo "错误: 无效的 garment_type '$GARMENT_TYPE'"
    echo "可选值: top_long, top_short, pant_long, pant_short"
    exit 1
fi

# 默认参数
DEFAULT_POLICY_PATH="/home/maozan/code/data/ckpts/pi05_step124k"
DEFAULT_DATASET_ROOT="Datasets/example/four_types_merged"
DEFAULT_NUM_EPISODES=5
DEFAULT_VIDEO_DIR="outputs/eval_videos_pi05_step124k"

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
echo "  Pi0.5 单类型评估"
echo "  Policy: $POLICY_PATH"
echo "  数据集: $DATASET_ROOT"
echo "  Episodes: $NUM_EPISODES"
echo "  无头模式: $HEADLESS"
echo "  保存视频: $SAVE_VIDEO${VIDEO_DIR:+ (→ $VIDEO_DIR)}"
echo "  衣物类型: $GARMENT_TYPE"
echo "  额外参数: ${EXTRA_ARGS[*]:-无}"
echo "========================================"

python -m scripts.eval \
    --policy_type lerobot \
    --garment_type "$GARMENT_TYPE" \
    --enable_cameras \
    --device cpu \
    --policy_path "$POLICY_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --num_episodes "$NUM_EPISODES" \
    "${HEADLESS_ARGS[@]}" \
    "${VIDEO_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo "  $GARMENT_TYPE 评估完成"
echo "========================================"
