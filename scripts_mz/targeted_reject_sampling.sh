#!/usr/bin/env bash
# Targeted Rejection Sampling v2: 按单件衣物粒度的针对性拒绝采样
# 用法:
#   bash scripts/targeted_reject_sampling.sh [OPTIONS]
#
# ====================== 采样策略 ======================
#
# 基于 pi05_step62k_reject 评测结果 (10eps/件, max_steps=600)
# 数据来源: submission/rollout_results.txt
#
# 按单件成功率分 6 档:
#
#   成功率    | 采样量 | 理由
#   ----------|--------|------
#   100%      | 5 eps  | 已满分，少量补充防遗忘
#   >= 90%    | 5 eps  | 很强，少量补充
#   80-89%    | 10 eps | 较强但仍有失败模式，适度补充
#   50-79%    | 20 eps | 中等水平，需要较多成功轨迹引导
#   40-49%    | 30 eps | 弱项，重点补充
#   20-39%    | 40 eps | 极弱，极限加量 (同时由人工遥操作补充)
#   0%        | 跳过   | 策略完全失败，纯靠人工遥操作
#
# 各品类预算分配 (总计 ~820 eps):
#   Top_Long   (61.67%): ~235 eps
#   Top_Short  (55.83%): ~225 eps
#   Pant_Long  (57.50%): ~235 eps
#   Pant_Short (85.00%): ~125 eps
#
# =======================================================
#
# 通过覆写 Release_test_list.txt + custom garment_type 实现单件粒度控制
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

# 合并配置
RUN_MERGE="${RUN_MERGE:-true}"
ORIGINAL_DATASET="${ORIGINAL_DATASET:-Datasets/example/four_types_merged}"
MERGE_OUTPUT="${MERGE_OUTPUT:-Datasets/example/four_types_with_reject_v2}"
MERGE_REPO_ID="${MERGE_REPO_ID:-four_types_with_reject_v2}"

# 透传给 eval 的额外参数
EXTRA_ARGS=("$@")

# ---------- 单件衣物采样配置 ----------
# 格式: "衣物名|episode数"
# 基于 submission/rollout_results.txt (pi05_step62k_reject, 10eps/件, max_steps=600)
# 分档: 100%→5, >=90%→5, 80-89%→10, 50-79%→20, 40-49%→30, 20-39%→40, 0%→skip

SAMPLING_JOBS=(
    # ===== Unseen 补采：目标拉齐到 ~35 eps/garment =====
    # 数字 = 需补eps / eval成功率，即实际需要roll的次数
    # 按 效率低→高 排序（成功率低的先跑，耗时长）

    # Top_Short_Unseen_1: 当前8eps, 需补27, eval 30% → 27/0.3=90
    "Top_Short_Unseen_1|90"

    # Pant_Short_Unseen_0: 当前7eps, 需补28, eval 40% → 28/0.4=70
    "Pant_Short_Unseen_0|70"

    # Top_Long_Unseen_1: 当前15eps, 需补20, eval 40% → 20/0.4=50
    "Top_Long_Unseen_1|50"

    # Pant_Long_Unseen_1: 当前7eps, 需补28, eval 60% → 28/0.6=47
    "Pant_Long_Unseen_1|47"

    # Top_Long_Unseen_0: 当前2eps, 需补33, eval 70% → 33/0.7=48
    "Top_Long_Unseen_0|48"

    # Pant_Long_Unseen_0: 当前2eps, 需补33, eval 70% → 33/0.7=48
    "Pant_Long_Unseen_0|48"

    # Pant_Short_Unseen_1: 当前4eps, 需补31, eval 80% → 31/0.8=39
    "Pant_Short_Unseen_1|39"

    # Top_Short_Unseen_0: eval 0%, reject采样大概率无法收集, 但尝试一下
    "Top_Short_Unseen_0|100"

    # ===== Seen 低成功率(<50%)补采：目标拉到 ~60 eps/garment =====

    # Pant_Long_Seen_0: 当前37eps, 需补23, eval 20% → 23/0.2=115
    "Pant_Long_Seen_0|115"

    # Top_Short_Seen_9: 当前35eps, 需补25, eval 20% → 25/0.2=125
    "Top_Short_Seen_9|125"

    # Top_Short_Seen_7: 当前32eps, 需补28, eval 30% → 28/0.3=94
    "Top_Short_Seen_7|94"

    # Top_Short_Seen_2: 当前37eps, 需补23, eval 40% → 23/0.4=58
    "Top_Short_Seen_2|58"

    # Top_Short_Seen_8: 当前32eps, 需补28, eval 40% → 28/0.4=70
    "Top_Short_Seen_8|70"

    # Top_Long_Seen_3: 当前37eps, 需补23, eval 40% → 23/0.4=58
    "Top_Long_Seen_3|58"
)

# ---------- 路径 ----------
GARMENT_ASSETS="Assets/objects/Challenge_Garment"
CUSTOM_LIST="${PROJECT_DIR}/${GARMENT_ASSETS}/Release/Release_test_list.txt"

# 备份原始衣物列表，退出时恢复
CUSTOM_LIST_BACKUP=""
if [ -f "${CUSTOM_LIST}" ]; then
    CUSTOM_LIST_BACKUP="${CUSTOM_LIST}.bak.$$"
    cp "${CUSTOM_LIST}" "${CUSTOM_LIST_BACKUP}"
fi

cleanup_custom_list() {
    if [ -n "${CUSTOM_LIST_BACKUP}" ] && [ -f "${CUSTOM_LIST_BACKUP}" ]; then
        mv "${CUSTOM_LIST_BACKUP}" "${CUSTOM_LIST}"
    else
        rm -f "${CUSTOM_LIST}"
    fi
}
trap cleanup_custom_list EXIT

# ---------- 辅助函数 ----------
run_single_garment() {
    local garment_name="$1"
    local num_episodes="$2"

    # 推导类别名: Top_Short_Seen_0 -> Top_Short
    local category
    category=$(echo "$garment_name" | sed 's/_Seen_.*//;s/_Unseen_.*//')

    local out_dir="${EVAL_BASE}/${category}/${garment_name}"

    # 覆写自定义衣物列表
    echo "${garment_name}" > "${CUSTOM_LIST}"

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
        --garment_cfg_base_path "${GARMENT_ASSETS}"
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
    if [ "${EXTRA_ARGS[@]+x}" ] && [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
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
SKIPPED=()
JOB_INDEX=0

for job in "${SAMPLING_JOBS[@]}"; do
    JOB_INDEX=$((JOB_INDEX + 1))
    IFS='|' read -r garment_name num_episodes <<< "$job"

    echo ""
    echo "[${JOB_INDEX}/${TOTAL_JOBS}] ${garment_name} (${num_episodes} eps)"

    # 断点续跑: 检查是否已有足够 episodes 的输出
    out_dir="${EVAL_BASE}/$(echo "$garment_name" | sed 's/_Seen_.*//;s/_Unseen_.*//')/${garment_name}"
    existing_eps=0
    while IFS= read -r info_file; do
        eps=$(python3 -c "import json; print(json.load(open('${info_file}')).get('total_episodes', 0))" 2>/dev/null || echo "0")
        existing_eps=$((existing_eps + eps))
    done < <(find "$out_dir" -name "info.json" -path "*/meta/*" 2>/dev/null)

    if [ "$existing_eps" -ge "$num_episodes" ]; then
        echo "[SKIP] ${garment_name} 已有 ${existing_eps} eps >= 目标 ${num_episodes} eps，跳过"
        SKIPPED+=("${garment_name}")
        continue
    elif [ "$existing_eps" -gt 0 ]; then
        remaining=$((num_episodes - existing_eps))
        echo "[INFO] ${garment_name} 已有 ${existing_eps} eps，还需 ${remaining} eps，补充采样"
    else
        remaining=$num_episodes
    fi

    if ! run_single_garment "$garment_name" "$remaining"; then
        echo "[ERROR] ${garment_name} 采样失败"
        FAILED+=("${garment_name}")
    else
        SUCCEEDED+=("${garment_name}")
    fi

done

# ---------- 汇总报告 ----------
echo ""
echo "============================================================"
echo "Targeted Rejection Sampling v2 执行完毕"
echo "  成功: ${#SUCCEEDED[@]}/${TOTAL_JOBS}"
echo "  跳过: ${#SKIPPED[@]}/${TOTAL_JOBS}"
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

# ============================================================
# Phase 2: 合并 reject sampling 数据与原始训练集
# ============================================================
if [ "$RUN_MERGE" = "true" ]; then
    echo ""
    echo "============================================================"
    echo "Phase 2: 合并数据"
    echo "  原始数据: ${ORIGINAL_DATASET}"
    echo "  Reject v2: ${EVAL_BASE}"
    echo "  输出: ${MERGE_OUTPUT}"
    echo "============================================================"

    # 收集所有有效的 reject sampling 输出
    REJECT_SOURCES=()
    while IFS= read -r info_file; do
        eps=$(python3 -c "import json; print(json.load(open('${info_file}')).get('total_episodes', 0))" 2>/dev/null || echo "0")
        if [ "$eps" -gt 0 ]; then
            ds_dir=$(dirname "$(dirname "$info_file")")
            REJECT_SOURCES+=("${ds_dir}")
        fi
    done < <(find "$EVAL_BASE" -name "info.json" -path "*/meta/*" 2>/dev/null | sort)

    echo "  发现 ${#REJECT_SOURCES[@]} 个有效 reject sampling 数据集"

    if [ ${#REJECT_SOURCES[@]} -eq 0 ]; then
        echo "[WARN] 无有效 reject sampling 输出，跳过合并"
    else
        # 也包含人工遥操作数据 (如果存在)
        TELEOP_DIR="Datasets/manual_teleop"
        if [ -d "$TELEOP_DIR" ]; then
            while IFS= read -r info_file; do
                eps=$(python3 -c "import json; print(json.load(open('${info_file}')).get('total_episodes', 0))" 2>/dev/null || echo "0")
                if [ "$eps" -gt 0 ]; then
                    ds_dir=$(dirname "$(dirname "$info_file")")
                    REJECT_SOURCES+=("${ds_dir}")
                    echo "  包含遥操作数据: ${ds_dir} (${eps} eps)"
                fi
            done < <(find "$TELEOP_DIR" -name "info.json" -path "*/meta/*" 2>/dev/null | sort)
        fi

        # 构建 Python 合并调用
        python3 -c "
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from lerobot.datasets.dataset_tools import merge_datasets as lerobot_merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scripts.utils.dataset_processing import merge_garment_info

original_root = Path('${ORIGINAL_DATASET}')
output_root = Path('${MERGE_OUTPUT}')
output_repo_id = '${MERGE_REPO_ID}'

# 收集所有数据源
source_roots = [original_root]
reject_paths = [$(printf "'%s', " "${REJECT_SOURCES[@]}")]
source_roots.extend([Path(p) for p in reject_paths])

print(f'合并 {len(source_roots)} 个数据源 (1 原始 + {len(source_roots)-1} 增强):')
datasets = []
for root in source_roots:
    ds = LeRobotDataset(repo_id=root.name, root=root)
    print(f'  {root}: {ds.meta.total_episodes} eps, {ds.meta.total_frames} frames')
    datasets.append(ds)

merged = lerobot_merge_datasets(
    datasets=datasets,
    output_repo_id=output_repo_id,
    output_dir=output_root,
)
print(f'合并完成: {merged.meta.total_episodes} eps, {merged.meta.total_frames} frames -> {output_root}')

# 合并 garment_info.json
merge_garment_info(source_roots, output_root)
print('garment_info.json 合并完成')
"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 合并完成: ${MERGE_OUTPUT}"
    fi
else
    echo ""
    echo "[INFO] 跳过合并 (RUN_MERGE=false)"
fi
