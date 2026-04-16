#!/usr/bin/env bash
# Rejection Sampling: 依次对所有衣物类别执行 eval 采集数据
# 用法:
#   bash scripts/reject_sampling.sh [OPTIONS]
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
EVAL_BASE="${EVAL_BASE:-Datasets/reject_sampling}"
DEVICE="${DEVICE:-cpu}"
MAX_STEPS="${MAX_STEPS:-500}"
# 类别之间的等待秒数，避免 Isaac Sim 启动过于密集导致渲染问题
COOLDOWN="${COOLDOWN:-10}"

# 透传给 eval 的额外参数
EXTRA_ARGS=("$@")

# 每个类别的 episode 数（每件衣物），按成功率反比调整:
#   pant_short 80% -> 8,  top_long 70% -> 10,  pant_long 50% -> 15,  top_short 40% -> 20
declare -A EPISODES_MAP=(
    [pant_short]=8
    [top_long]=10
    [pant_long]=15
    [top_short]=20
)

# 执行顺序：成功率从高到低
CATEGORY_ORDER=(pant_short top_long pant_long top_short)

# ---------- 辅助函数 ----------
run_category() {
    local garment_type="$1"
    local num_episodes="$2"
    local out_dir="${EVAL_BASE}/${garment_type}"

    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始: ${garment_type}  (${num_episodes} eps/件)"
    echo "  输出目录: ${out_dir}"
    echo "============================================================"

    python -m scripts.eval \
        --policy_type lerobot \
        --policy_path "${POLICY_PATH}" \
        --dataset_root "${DATASET_ROOT}" \
        --garment_type "${garment_type}" \
        --use_random_seed \
        --save_datasets \
        --eval_dataset_path "${out_dir}" \
        --num_episodes "${num_episodes}" \
        --max_steps "${MAX_STEPS}" \
        --task LeHome-BiSO101-Direct-Garment-v2 \
        --task_description "fold the garment on the table" \
        --garment_cfg_base_path Assets/objects/Challenge_Garment \
        --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml \
        --device "${DEVICE}" \
        "${EXTRA_ARGS[@]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: ${garment_type}"

    # 等待一段时间再启动下一个类别，让 GPU 资源释放
    if [ "${garment_type}" != "${CATEGORY_ORDER[-1]}" ]; then
        echo "等待 ${COOLDOWN} 秒后启动下一个类别..."
        sleep "${COOLDOWN}"
    fi
    echo ""
}

# ---------- 主流程 ----------
echo "============================================================"
echo "Rejection Sampling - 全类别数据采集"
echo "  模型路径:   ${POLICY_PATH}"
echo "  数据集根:   ${DATASET_ROOT}"
echo "  输出目录:   ${EVAL_BASE}"
echo "  设备:       ${DEVICE}"
echo "============================================================"
echo ""

FAILED=()
for garment_type in "${CATEGORY_ORDER[@]}"; do
    num_episodes="${EPISODES_MAP[$garment_type]}"
    if ! run_category "${garment_type}" "${num_episodes}" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"; then
        echo "[错误] ${garment_type} 失败，跳过继续执行..."
        FAILED+=("${garment_type}")
    fi
done

echo "============================================================"
echo "Rejection Sampling 执行完毕"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "所有类别均成功完成。"
else
    echo "失败类别: ${FAILED[*]}"
    exit 1
fi
echo "============================================================"
