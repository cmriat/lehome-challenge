#!/usr/bin/env bash
# Batch Eval: 对所有衣物类别执行模型评估，输出成功率等指标
# 用法:
#   bash scripts/batch_eval.sh [OPTIONS]
#
# 所有额外参数会透传给 `python -m scripts.eval`
# --garment_type 和 --num_episodes 按类别自动设置

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 默认配置
POLICY_PATH="${POLICY_PATH:-/home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject}"
DATASET_ROOT="${DATASET_ROOT:-Datasets/example/four_types_merged}"
DEVICE="${DEVICE:-cpu}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
HEADLESS="${HEADLESS:-}"
# 类别之间的等待秒数，避免 Isaac Sim 启动过于密集
COOLDOWN="${COOLDOWN:-10}"

# 透传给 eval 的额外参数
EXTRA_ARGS=("$@")

# 执行顺序
CATEGORY_ORDER=(top_long top_short pant_long pant_short)

# ---------- 辅助函数 ----------
run_category() {
    local garment_type="$1"
    local num_episodes="$2"

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始评估: ${garment_type}  (${num_episodes} eps/件)"
    echo "  模型:   ${POLICY_PATH}"
    echo "  数据集: ${DATASET_ROOT}"
    echo "  设备:   ${DEVICE}"
    echo "============================================================"

    local cmd=(
        python -m scripts.eval
        --policy_type lerobot
        --policy_path "${POLICY_PATH}"
        --dataset_root "${DATASET_ROOT}"
        --garment_type "${garment_type}"
        --num_episodes "${num_episodes}"
        --max_steps "${MAX_STEPS}"
        --task LeHome-BiSO101-Direct-Garment-v2
        --task_description "fold the garment on the table"
        --garment_cfg_base_path Assets/objects/Challenge_Garment
        --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml
        --enable_cameras
        --device "${DEVICE}"
    )

    if [ -n "${HEADLESS}" ]; then
        cmd+=(--headless)
    fi

    # 追加额外参数
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    "${cmd[@]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: ${garment_type}"
}

# ---------- 主流程 ----------
echo "============================================================"
echo "Batch Eval - 全衣物类别模型评估"
echo "  模型路径:   ${POLICY_PATH}"
echo "  数据集根:   ${DATASET_ROOT}"
echo "  每件ep数:   ${NUM_EPISODES}"
echo "  最大步数:   ${MAX_STEPS}"
echo "  设备:       ${DEVICE}"
echo "  Headless:   ${HEADLESS:-否}"
echo "============================================================"
echo ""

FAILED=()
for garment_type in "${CATEGORY_ORDER[@]}"; do
    if ! run_category "${garment_type}" "${NUM_EPISODES}"; then
        echo "[错误] ${garment_type} 评估失败，跳过继续执行..."
        FAILED+=("${garment_type}")
    fi

    if [ "${garment_type}" != "${CATEGORY_ORDER[-1]}" ]; then
        echo "等待 ${COOLDOWN} 秒后启动下一个类别..."
        sleep "${COOLDOWN}"
    fi
    echo ""
done

echo "============================================================"
echo "Batch Eval 执行完毕"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "所有类别均成功完成。"
else
    echo "失败类别: ${FAILED[*]}"
    exit 1
fi
echo "============================================================"
