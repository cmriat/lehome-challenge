#!/usr/bin/env bash
# 合并两个 LeRobot 数据集 (训练用, 不合并 garment_info)
# 用法:
#   bash scripts_mz/merge_datasets.sh [source1] [source2] [output_root]
#
# 示例:
#   bash scripts_mz/merge_datasets.sh \
#     /path/to/others_combined \
#     /path/to/four_types_merged_great_v2 \
#     /path/to/merged_output

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC1="${1:-/home/jovyan/code/vla/temp_datas_v1/pi05_mz/datas/LeHome/eval_2k/others_combined}"
SRC2="${2:-/home/jovyan/code/vla/temp_datas_v1/pi05_mz/datas/LeHome/raw_clean/four_types_merged_great_v2}"
OUTPUT_ROOT="${3:-/home/jovyan/code/vla/temp_datas_v1/pi05_mz/datas/LeHome/final_3k}"

echo "=========================================="
echo "  数据集合并"
echo "=========================================="
echo "  Source 1: $SRC1"
echo "  Source 2: $SRC2"
echo "  Output:   $OUTPUT_ROOT"
echo "=========================================="

cd "$PROJECT_DIR"
python -m scripts.dataset merge \
    --source_roots "[\"$SRC1\", \"$SRC2\"]" \
    --output_root "$OUTPUT_ROOT"

echo ""
echo "Done. Merged dataset at: $OUTPUT_ROOT"
