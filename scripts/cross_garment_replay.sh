#!/usr/bin/env bash
# Cross-Garment Replay: 将高成功率 Seen 衣物的动作序列重放到弱项/Unseen 衣物上
# 用法:
#   bash scripts/cross_garment_replay.sh [OPTIONS]
#
# 所有额外参数会透传给 `pixi run dataset-sim replay`
# 任务列表按尺度分组配对，确保同尺度的衣物才进行跨衣物 replay

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 配置 ----------
REPLAY_OUTPUT="${REPLAY_OUTPUT:-Datasets/cross_garment_replay}"
REPLAY_PER_EPISODE="${REPLAY_PER_EPISODE:-3}"
DEVICE="${DEVICE:-cpu}"
COOLDOWN="${COOLDOWN:-10}"

# 透传给 replay 的额外参数
EXTRA_ARGS=("$@")

# ---------- Replay 任务定义 ----------
# 格式: "TARGET_GARMENT|SOURCE_TYPE_MERGED|START_EP|END_EP|SOURCE_DESC"
# SOURCE_TYPE_MERGED: 使用 Datasets/example/{type}_merged 作为源数据集
# Episode 排布: Seen_0=0-24, Seen_1=25-49, ..., Seen_9=225-249
#
# 配对原则: 同尺度组内，从高成功率 Seen 衣物 replay 到弱项衣物
# 尺度分组:
#   Top_Short scale=0.45: Seen_0,1,2,5,6,7,8,9 + Unseen_0
#   Top_Short scale=0.65: Seen_3,4 + Unseen_1
#   Top_Long  scale=0.45: Seen_0-2,4-9 + Unseen_0,1
#   Top_Long  scale=0.40: Seen_3 (无同尺度源!)
#   Pant_Long scale=0.37: 全部12件
#   Pant_Short scale=0.45: 全部12件

REPLAY_JOBS=(
    # ===== Top_Short (scale=0.45 组) =====
    # Unseen_0 (0%) <- 3个高成功率源
    "Top_Short_Unseen_0|top_short|0|25|Seen_0(90%)"
    "Top_Short_Unseen_0|top_short|25|50|Seen_1(80%)"
    "Top_Short_Unseen_0|top_short|125|150|Seen_5(70%)"
    # Seen_9 (20%) <- Seen_0
    "Top_Short_Seen_9|top_short|0|25|Seen_0(90%)"
    # Seen_7 (30%) <- Seen_1
    "Top_Short_Seen_7|top_short|25|50|Seen_1(80%)"

    # ===== Top_Short (scale=0.65 组) =====
    # Unseen_1 (30%) <- Seen_3, Seen_4
    "Top_Short_Unseen_1|top_short|75|100|Seen_3(60%)"
    "Top_Short_Unseen_1|top_short|100|125|Seen_4(90%)"

    # ===== Top_Long (scale=0.45 组) =====
    # Unseen_1 (40%) <- Seen_5, Seen_1
    "Top_Long_Unseen_1|top_long|125|150|Seen_5(100%)"
    "Top_Long_Unseen_1|top_long|25|50|Seen_1(90%)"
    # Seen_3 (40%, scale=0.40) <- Seen_5 (scale=0.45, 跨尺度尝试)
    "Top_Long_Seen_3|top_long|125|150|Seen_5(100%)_cross_scale"

    # ===== Pant_Short (scale=0.45 组) =====
    # Unseen_0 (40%) <- Seen_1, Seen_3
    "Pant_Short_Unseen_0|pant_short|25|50|Seen_1(100%)"
    "Pant_Short_Unseen_0|pant_short|75|100|Seen_3(100%)"

    # ===== Pant_Long (scale=0.37 组) =====
    # Seen_0 (20%) <- Seen_1, Seen_5
    "Pant_Long_Seen_0|pant_long|25|50|Seen_1(70%)"
    "Pant_Long_Seen_0|pant_long|125|150|Seen_5(70%)"
)

# ---------- 辅助函数 ----------
check_existing_output() {
    local output_dir="$1"
    # 查找 output_dir 下的任何 meta/info.json (可能在子目录中)
    local info_files
    info_files=$(find "$output_dir" -name "info.json" -path "*/meta/*" 2>/dev/null || true)
    if [ -n "$info_files" ]; then
        local total_eps=0
        while IFS= read -r f; do
            local eps
            eps=$(python3 -c "import json; print(json.load(open('$f')).get('total_episodes', 0))" 2>/dev/null || echo "0")
            total_eps=$((total_eps + eps))
        done <<< "$info_files"
        if [ "$total_eps" -gt 0 ]; then
            echo "$total_eps"
            return 0
        fi
    fi
    echo "0"
    return 1
}

run_replay_job() {
    local target_garment="$1"
    local source_type="$2"
    local start_ep="$3"
    local end_ep="$4"
    local source_desc="$5"

    local source_dataset="Datasets/example/${source_type}_merged"
    # 按类别和 target 组织输出目录
    local output_dir="${REPLAY_OUTPUT}/${source_type}/${target_garment}"

    # 检查源数据集是否存在
    if [ ! -d "$source_dataset" ]; then
        echo "[ERROR] 源数据集不存在: ${source_dataset}"
        return 1
    fi

    echo "------------------------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Replay Job"
    echo "  Target: ${target_garment}"
    echo "  Source: ${source_desc} (${source_dataset}, ep ${start_ep}-${end_ep})"
    echo "  Output: ${output_dir}"
    echo "  Replays/Episode: ${REPLAY_PER_EPISODE}"
    echo "------------------------------------------------------------"

    local cmd=(
        pixi run dataset-sim replay
        --dataset_root "${source_dataset}"
        --output_root "${output_dir}"
        --override_garment "${target_garment}"
        --start_episode "${start_ep}"
        --end_episode "${end_ep}"
        --num_replays "${REPLAY_PER_EPISODE}"
        --save_successful_only
        --task LeHome-BiSO101-Direct-Garment-v2
        --task_description "fold the garment on the table"
        --garment_cfg_base_path Assets/objects/Challenge_Garment
        --particle_cfg_path source/lehome/lehome/tasks/bedroom/config_file/particle_garment_cfg.yaml
        --device "${DEVICE}"
    )

    # 追加额外参数
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    "${cmd[@]}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Replay 完成: ${target_garment} <- ${source_desc}"
}

# ---------- 主流程 ----------
TOTAL_JOBS=${#REPLAY_JOBS[@]}

echo "============================================================"
echo "Cross-Garment Replay - 跨衣物数据增强"
echo "  输出根目录:    ${REPLAY_OUTPUT}"
echo "  每集重放次数:  ${REPLAY_PER_EPISODE}"
echo "  总任务数:      ${TOTAL_JOBS}"
echo "============================================================"
echo ""

FAILED=()
SUCCEEDED=()
SKIPPED=()
JOB_INDEX=0

for job in "${REPLAY_JOBS[@]}"; do
    JOB_INDEX=$((JOB_INDEX + 1))
    IFS='|' read -r target source_type start end source_desc <<< "$job"

    echo ""
    echo "============================================================"
    echo "[${JOB_INDEX}/${TOTAL_JOBS}] ${target} <- ${source_desc}"
    echo "============================================================"

    if ! run_replay_job "$target" "$source_type" "$start" "$end" "$source_desc"; then
        echo "[ERROR] Replay 失败: ${target} <- ${source_desc}"
        FAILED+=("${target}<-${source_desc}")
    else
        SUCCEEDED+=("${target}<-${source_desc}")
    fi

    # 任务间等待
    if [ "$JOB_INDEX" -lt "$TOTAL_JOBS" ]; then
        echo "等待 ${COOLDOWN} 秒后启动下一个任务..."
        sleep "${COOLDOWN}"
    fi
done

# ---------- 汇总报告 ----------
echo ""
echo "============================================================"
echo "Cross-Garment Replay 执行完毕"
echo "  成功: ${#SUCCEEDED[@]}/${TOTAL_JOBS}"
echo "  失败: ${#FAILED[@]}/${TOTAL_JOBS}"
echo "  跳过: ${#SKIPPED[@]}/${TOTAL_JOBS}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "失败任务:"
    for f in "${FAILED[@]}"; do
        echo "  - ${f}"
    done
fi

if [ ${#SUCCEEDED[@]} -gt 0 ]; then
    echo ""
    echo "成功任务:"
    for s in "${SUCCEEDED[@]}"; do
        echo "  + ${s}"
    done
fi
echo "============================================================"

# 如有失败则退出码非零
if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
