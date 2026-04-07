#!/usr/bin/env bash
# Augmentation Pipeline: 完整数据增强管线
# 用法:
#   bash scripts/augmentation_pipeline.sh [OPTIONS]
#
# ====================== 分层采样策略概述 ======================
#
# 基于 pi05_step62k 模型的评测结果（10eps/件, seed=42），按单件衣物粒度
# 设计两阶段互补的数据增强策略:
#
# Phase 1a - Targeted Reject Sampling (针对性拒绝采样):
#   用当前策略对每件衣物做 rollout，只保留成功轨迹。
#   按成功率反比分配采样预算:
#     >=90%  → 3 eps   (已经很强，少量补充)
#     80-89% → 5 eps
#     50-79% → 10 eps
#     20-49% → 15 eps  (弱项，重点补充)
#     0%     → 跳过    (策略完全无法成功，交给 Phase 1b)
#
# Phase 1b - Cross-Garment Replay (跨衣物动作重放):
#   将同尺度组内高成功率衣物的成功轨迹，重放到弱项/Unseen 衣物上。
#   覆盖 Phase 1a 无法触及的 0% 衣物，以及 20-50% 区间的弱项。
#   严格遵守同尺度配对约束（衣物 scale 一致才能 replay）。
#
# Phase 2 - Merge (三级合并):
#   2a: 合并所有 reject sampling v2 子目录
#   2b: 合并所有 cross-garment replay 子目录
#   2c: 原始数据 + reject_v1 + reject_v2 + cross_garment → 最终训练集
#
# 设计原则:
#   - 采样预算向弱项品类倾斜 (Top_Short 50% / Pant_Long 57.5% 获得更多资源)
#   - 高成功率品类 (Pant_Short 84%) 削减预算，避免浪费
#   - 两阶段互补: reject sampling 产生策略分布内的轨迹，
#     cross-garment replay 通过迁移扩展覆盖范围
#
# ==============================================================
#
# 阶段控制 (环境变量):
#   RUN_REJECT_SAMPLING=true/false  (默认 true)
#   RUN_CROSS_GARMENT=true/false    (默认 true)
#   RUN_MERGE=true/false            (默认 true)
#
# 其他配置:
#   POLICY_PATH, DEVICE, HEADLESS 等透传给子脚本

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- 阶段控制 ----------
RUN_REJECT_SAMPLING="${RUN_REJECT_SAMPLING:-true}"
RUN_CROSS_GARMENT="${RUN_CROSS_GARMENT:-true}"
RUN_MERGE="${RUN_MERGE:-false}"

# ---------- 路径配置 ----------
ORIGINAL_DATASET="Datasets/example/four_types_merged"
REJECT_V1="Datasets/reject_sampling"
REJECT_V2="Datasets/reject_sampling_v2"
CROSS_GARMENT="Datasets/cross_garment_replay"
FINAL_OUTPUT="${FINAL_OUTPUT:-Datasets/augmented_training_v2}"
FINAL_REPO_ID="${FINAL_REPO_ID:-augmented_training_v2}"

# ---------- 日志 ----------
LOG_DIR="logs/augmentation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/pipeline.log"
}

# 透传给子脚本的额外参数
EXTRA_ARGS=("$@")

# ============================================================
echo "============================================================"
echo "Augmentation Pipeline"
echo "  原始数据:       ${ORIGINAL_DATASET}"
echo "  Reject v1:      ${REJECT_V1}"
echo "  Reject v2:      ${REJECT_V2}"
echo "  Cross-Garment:  ${CROSS_GARMENT}"
echo "  最终输出:       ${FINAL_OUTPUT}"
echo ""
echo "  阶段:"
echo "    Reject Sampling: ${RUN_REJECT_SAMPLING}"
echo "    Cross-Garment:   ${RUN_CROSS_GARMENT}"
echo "    Merge:           ${RUN_MERGE}"
echo "  日志目录:       ${LOG_DIR}"
echo "============================================================"
echo ""

PHASE_FAILED=()

# ============================================================
# Phase 1a: Targeted Rejection Sampling v2
# ============================================================
if [ "$RUN_REJECT_SAMPLING" = "true" ]; then
    log "Phase 1a: 开始 Targeted Rejection Sampling v2"
    if bash scripts/targeted_reject_sampling.sh "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        2>&1 | tee "${LOG_DIR}/reject_sampling_v2.log"; then
        log "Phase 1a: Reject Sampling v2 完成"
    else
        log "Phase 1a: Reject Sampling v2 部分失败 (继续执行)"
        PHASE_FAILED+=("reject_sampling_v2")
    fi
    echo ""
else
    log "Phase 1a: 跳过 Reject Sampling"
fi

# ============================================================
# Phase 1b: Cross-Garment Replay
# ============================================================
if [ "$RUN_CROSS_GARMENT" = "true" ]; then
    log "Phase 1b: 开始 Cross-Garment Replay"
    if bash scripts/cross_garment_replay.sh "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        2>&1 | tee "${LOG_DIR}/cross_garment_replay.log"; then
        log "Phase 1b: Cross-Garment Replay 完成"
    else
        log "Phase 1b: Cross-Garment Replay 部分失败 (继续执行)"
        PHASE_FAILED+=("cross_garment_replay")
    fi
    echo ""
else
    log "Phase 1b: 跳过 Cross-Garment Replay"
fi

# ============================================================
# Phase 2: Merge
# ============================================================
if [ "$RUN_MERGE" = "true" ]; then
    log "Phase 2: 开始合并数据"

    # --- 2a: 合并 Reject Sampling v2 (单件粒度，嵌套目录结构) ---
    if [ -d "$REJECT_V2" ]; then
        log "Phase 2a: 收集并合并 Reject Sampling v2 输出"

        REJECT_V2_SOURCES=()
        while IFS= read -r info_file; do
            eps=$(python3 -c "import json; print(json.load(open('${info_file}')).get('total_episodes', 0))" 2>/dev/null || echo "0")
            if [ "$eps" -gt 0 ]; then
                ds_dir=$(dirname "$(dirname "$info_file")")
                REJECT_V2_SOURCES+=("${ds_dir}")
                log "  发现: ${ds_dir} (${eps} episodes)"
            fi
        done < <(find "$REJECT_V2" -name "info.json" -path "*/meta/*" 2>/dev/null | sort)

        if [ ${#REJECT_V2_SOURCES[@]} -gt 0 ]; then
            SOURCE_LIST="["
            for i in "${!REJECT_V2_SOURCES[@]}"; do
                if [ "$i" -gt 0 ]; then
                    SOURCE_LIST+=", "
                fi
                SOURCE_LIST+="'${REJECT_V2_SOURCES[$i]}'"
            done
            SOURCE_LIST+="]"

            log "  合并 ${#REJECT_V2_SOURCES[@]} 个 reject v2 数据集"
            python scripts/dataset.py merge \
                --source_roots "${SOURCE_LIST}" \
                --output_root "${REJECT_V2}/all_merged" \
                --output_repo_id reject_v2_merged \
                2>&1 | tee "${LOG_DIR}/merge_reject_v2.log" || true
        else
            log "  无有效 reject v2 输出，跳过合并"
        fi
    else
        log "Phase 2a: ${REJECT_V2} 不存在，跳过"
    fi

    # --- 2b: 合并 Cross-Garment Replay 结果 ---
    if [ -d "$CROSS_GARMENT" ]; then
        log "Phase 2b: 收集并合并 Cross-Garment Replay 输出"

        # 收集所有有效的 replay 输出目录 (包含 meta/info.json 且 total_episodes > 0)
        REPLAY_SOURCES=()
        while IFS= read -r info_file; do
            eps=$(python3 -c "import json; print(json.load(open('${info_file}')).get('total_episodes', 0))" 2>/dev/null || echo "0")
            if [ "$eps" -gt 0 ]; then
                ds_dir=$(dirname "$(dirname "$info_file")")
                REPLAY_SOURCES+=("${ds_dir}")
                log "  发现: ${ds_dir} (${eps} episodes)"
            fi
        done < <(find "$CROSS_GARMENT" -name "info.json" -path "*/meta/*" 2>/dev/null | sort)

        if [ ${#REPLAY_SOURCES[@]} -gt 0 ]; then
            # 构建 Python list 字符串
            SOURCE_LIST="["
            for i in "${!REPLAY_SOURCES[@]}"; do
                if [ "$i" -gt 0 ]; then
                    SOURCE_LIST+=", "
                fi
                SOURCE_LIST+="'${REPLAY_SOURCES[$i]}'"
            done
            SOURCE_LIST+="]"

            log "  合并 ${#REPLAY_SOURCES[@]} 个 replay 数据集"
            python scripts/dataset.py merge \
                --source_roots "${SOURCE_LIST}" \
                --output_root "${CROSS_GARMENT}/all_merged" \
                --output_repo_id cross_garment_merged \
                2>&1 | tee "${LOG_DIR}/merge_cross_garment.log" || true
        else
            log "  无有效 replay 输出，跳过合并"
        fi
    else
        log "Phase 2b: ${CROSS_GARMENT} 不存在，跳过"
    fi

    # --- 2c: 最终合并 ---
    log "Phase 2c: 最终合并所有数据源"
    MERGE_SOURCES=("'${ORIGINAL_DATASET}'")

    # 添加 reject v1 merged
    for type_merged in "${REJECT_V1}"/*_merged; do
        if [ -d "$type_merged" ]; then
            MERGE_SOURCES+=("'${type_merged}'")
        fi
    done

    # 添加 reject v2 merged
    if [ -d "${REJECT_V2}/all_merged" ]; then
        MERGE_SOURCES+=("'${REJECT_V2}/all_merged'")
    fi

    # 添加 cross-garment merged
    if [ -d "${CROSS_GARMENT}/all_merged" ]; then
        MERGE_SOURCES+=("'${CROSS_GARMENT}/all_merged'")
    fi

    FINAL_SOURCE_LIST="[$(IFS=', '; echo "${MERGE_SOURCES[*]}")]"
    log "  数据源列表: ${FINAL_SOURCE_LIST}"

    python scripts/dataset.py merge \
        --source_roots "${FINAL_SOURCE_LIST}" \
        --output_root "${FINAL_OUTPUT}" \
        --output_repo_id "${FINAL_REPO_ID}" \
        2>&1 | tee "${LOG_DIR}/merge_final.log"

    log "Phase 2c: 最终合并完成 -> ${FINAL_OUTPUT}"
else
    log "Phase 2: 跳过合并"
fi

# ============================================================
# 总结
# ============================================================
echo ""
echo "============================================================"
log "Augmentation Pipeline 全部执行完毕"
if [ ${#PHASE_FAILED[@]} -eq 0 ]; then
    log "所有阶段均成功完成。"
else
    log "部分失败阶段: ${PHASE_FAILED[*]}"
fi
echo ""
echo "后续步骤:"
echo "  1. 检查最终数据集: ${FINAL_OUTPUT}"
echo "  2. 启动微调训练:"
echo "     lerobot-train --config_path=configs/train_pi05_finetune.yaml"
echo "  3. 训练完成后评估:"
echo "     POLICY_PATH=outputs/train/pi05_finetune_augmented/checkpoints/BEST/pretrained_model \\"
echo "     bash scripts/batch_eval.sh"
echo "============================================================"
