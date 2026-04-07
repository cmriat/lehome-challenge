#!/usr/bin/env bash
# Targeted Rejection Sampling v2: 按单件衣物粒度的针对性拒绝采样
# 用法:
#   bash scripts/targeted_reject_sampling.sh [OPTIONS]
#
# ====================== 采样策略 ======================
#
# 基于 pi05_step62k 评测结果 (10eps/件, seed=42)，按单件成功率分 5 档:
#
#   成功率    | 采样量 | 理由
#   ----------|--------|------
#   >= 90%    | 3 eps  | 已足够强，少量补充防止遗忘
#   80-89%    | 5 eps  | 较强但仍有失败模式，适度补充
#   50-79%    | 10 eps | 中等水平，需要较多成功轨迹引导
#   20-49%    | 15 eps | 弱项，重点补充 (实际有效轨迹 ≈ 成功率×15)
#   0%        | 跳过   | 策略完全失败，reject sampling 无法产出，
#             |        | 由 cross_garment_replay.sh 通过动作迁移覆盖
#
# 各品类预算分配 (总计 ~403 eps):
#   Pant_Short (84.17%): ~63 eps  — 削减高成功率预算
#   Top_Long   (72.50%): ~94 eps
#   Pant_Long  (57.50%): ~125 eps — 整体偏弱，全量投入
#   Top_Short  (50.00%): ~121 eps — 最弱品类，最大投入
#
# =======================================================
#
# 通过为每件衣物生成临时 txt 文件 + custom garment_type 实现单件粒度控制
# 所有额外参数会透传给 `python -m scripts.eval`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 配置 ----------
POLICY_PATH="${POLICY_PATH:-/home/nvidia/maoz/ckpts/lehome_pi05_step62k_reject}"
DATASET_ROOT="${DATASET_ROOT:-Datasets/example/four_types_merged}"
EVAL_BASE="${EVAL_BASE:-Datasets/reject_sampling_v2}"
DEVICE="${DEVICE:-cpu}"
MAX_STEPS="${MAX_STEPS:-500}"
COOLDOWN="${COOLDOWN:-10}"

# 透传给 eval 的额外参数
EXTRA_ARGS=("$@")

# ---------- 单件衣物采样配置 ----------
# 格式: "衣物名|episode数"
# 基于测评结果 (lehome_pi05_step62k_reject, seed=42, 10eps)
# 按成功率分级: >=90% -> 3eps, 80-89% -> 5eps, 50-79% -> 10eps, 20-49% -> 15eps, 0% -> skip

SAMPLING_JOBS=(
    # ===== Pant_Short (84.17% overall) =====
    "Pant_Short_Seen_0|3"       # 90%
    "Pant_Short_Seen_1|3"       # 100%
    "Pant_Short_Seen_2|10"      # 70%
    "Pant_Short_Seen_3|3"       # 100%
    "Pant_Short_Seen_4|3"       # 100%
    "Pant_Short_Seen_5|3"       # 90%
    "Pant_Short_Seen_6|3"       # 90%
    "Pant_Short_Seen_7|3"       # 100%
    "Pant_Short_Seen_8|10"      # 70%
    "Pant_Short_Seen_9|5"       # 80%
    "Pant_Short_Unseen_0|15"    # 40%
    "Pant_Short_Unseen_1|5"     # 80%

    # ===== Top_Long (72.50% overall) =====
    "Top_Long_Seen_0|5"         # 80%
    "Top_Long_Seen_1|3"         # 90%
    "Top_Long_Seen_2|5"         # 80%
    "Top_Long_Seen_3|15"        # 40%
    "Top_Long_Seen_4|10"        # 70%
    "Top_Long_Seen_5|3"         # 100%
    "Top_Long_Seen_6|10"        # 60%
    "Top_Long_Seen_7|10"        # 70%
    "Top_Long_Seen_8|3"         # 90%
    "Top_Long_Seen_9|5"         # 80%
    "Top_Long_Unseen_0|10"      # 70%
    "Top_Long_Unseen_1|15"      # 40%

    # ===== Pant_Long (57.50% overall) =====
    "Pant_Long_Seen_0|15"       # 20%
    "Pant_Long_Seen_1|10"       # 70%
    "Pant_Long_Seen_2|10"       # 50%
    "Pant_Long_Seen_3|10"       # 60%
    "Pant_Long_Seen_4|10"       # 60%
    "Pant_Long_Seen_5|10"       # 70%
    "Pant_Long_Seen_6|10"       # 50%
    "Pant_Long_Seen_7|10"       # 70%
    "Pant_Long_Seen_8|10"       # 50%
    "Pant_Long_Seen_9|10"       # 60%
    "Pant_Long_Unseen_0|10"     # 70%
    "Pant_Long_Unseen_1|10"     # 60%

    # ===== Top_Short (50.00% overall) =====
    "Top_Short_Seen_0|3"        # 90%
    "Top_Short_Seen_1|5"        # 80%
    "Top_Short_Seen_2|15"       # 40%
    "Top_Short_Seen_3|10"       # 60%
    "Top_Short_Seen_4|3"        # 90%
    "Top_Short_Seen_5|10"       # 70%
    "Top_Short_Seen_6|10"       # 50%
    "Top_Short_Seen_7|15"       # 30%
    "Top_Short_Seen_8|15"       # 40%
    "Top_Short_Seen_9|15"       # 20%
    # Top_Short_Unseen_0 跳过    # 0% -> 由 cross-garment replay 覆盖
    "Top_Short_Unseen_1|15"     # 30%
)

# ---------- 临时目录 ----------
TMP_CFG_DIR=$(mktemp -d "${PROJECT_DIR}/.tmp_garment_cfg_XXXXXX")
# 创建 Release 子目录结构
mkdir -p "${TMP_CFG_DIR}/Release"

cleanup_tmp() {
    rm -rf "${TMP_CFG_DIR}"
}
trap cleanup_tmp EXIT

# ---------- 辅助函数 ----------
run_single_garment() {
    local garment_name="$1"
    local num_episodes="$2"

    # 推导类别名: Top_Short_Seen_0 -> Top_Short
    local category
    category=$(echo "$garment_name" | sed 's/_Seen_.*//;s/_Unseen_.*//')

    local out_dir="${EVAL_BASE}/${category}/${garment_name}"

    # 生成临时单件衣物 txt 文件
    local tmp_list="${TMP_CFG_DIR}/Release/Release_test_list.txt"
    echo "${garment_name}" > "$tmp_list"

    echo "------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${garment_name} (${num_episodes} eps)"
    echo "  输出: ${out_dir}"
    echo "------------------------------------------------------------"

    local cmd=(
        python -m scripts.eval
        --policy_type lerobot
        --policy_path "${POLICY_PATH}"
        --dataset_root "${DATASET_ROOT}"
        --garment_type custom
        --garment_cfg_base_path "${TMP_CFG_DIR}"
        --use_random_seed
        --save_datasets
        --eval_dataset_path "${out_dir}"
        --num_episodes "${num_episodes}"
        --max_steps "${MAX_STEPS}"
        --task LeHome-BiSO101-Direct-Garment-v2
        --task_description "fold the garment on the table"
        --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml
        --device "${DEVICE}"
    )

    # 追加额外参数
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    "${cmd[@]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 完成: ${garment_name}"
}

# ---------- 主流程 ----------
TOTAL_JOBS=${#SAMPLING_JOBS[@]}
TOTAL_EPISODES=0
for job in "${SAMPLING_JOBS[@]}"; do
    IFS='|' read -r _ eps <<< "$job"
    TOTAL_EPISODES=$((TOTAL_EPISODES + eps))
done

echo "============================================================"
echo "Targeted Rejection Sampling v2 - 单件粒度精细化采样"
echo "  模型路径:   ${POLICY_PATH}"
echo "  数据集根:   ${DATASET_ROOT}"
echo "  输出目录:   ${EVAL_BASE}"
echo "  设备:       ${DEVICE}"
echo "  最大步数:   ${MAX_STEPS}"
echo "  总任务数:   ${TOTAL_JOBS} 件衣物"
echo "  总采样次数: ${TOTAL_EPISODES} episodes"
echo "============================================================"
echo ""

FAILED=()
SUCCEEDED=()
JOB_INDEX=0

for job in "${SAMPLING_JOBS[@]}"; do
    JOB_INDEX=$((JOB_INDEX + 1))
    IFS='|' read -r garment_name num_episodes <<< "$job"

    echo ""
    echo "[${JOB_INDEX}/${TOTAL_JOBS}] ${garment_name} (${num_episodes} eps)"

    if ! run_single_garment "$garment_name" "$num_episodes"; then
        echo "[ERROR] ${garment_name} 采样失败"
        FAILED+=("${garment_name}")
    else
        SUCCEEDED+=("${garment_name}")
    fi

    # 任务间等待
    if [ "$JOB_INDEX" -lt "$TOTAL_JOBS" ]; then
        sleep "${COOLDOWN}"
    fi
done

# ---------- 汇总报告 ----------
echo ""
echo "============================================================"
echo "Targeted Rejection Sampling v2 执行完毕"
echo "  成功: ${#SUCCEEDED[@]}/${TOTAL_JOBS}"
echo "  失败: ${#FAILED[@]}/${TOTAL_JOBS}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "失败衣物:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
fi
echo "============================================================"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
