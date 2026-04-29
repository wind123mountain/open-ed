#!/bin/bash
# Tokenize MINION data (per language) for Qwen3 student.
# Usage:  bash scripts/qwen/tools/process_data_minion.sh <lang>
# Example: bash scripts/qwen/tools/process_data_minion.sh spanish

set -e

LANG="${1:?usage: $0 <lang>}"
DATA_DIR="./data/minion_${LANG}"
OUT_DIR="./processed_data/minion_${LANG}"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: $DATA_DIR not found. Run gen_data_minion.py --lang $LANG first."
    exit 1
fi

export TF_CPP_MIN_LOG_LEVEL=3
PY="${PY:-python3}"

PYTHONPATH=. "$PY" ./tools/process_data.py \
    --data-dir "$DATA_DIR" \
    --processed-data-dir "$OUT_DIR" \
    --model-path Qwen/Qwen3-0.6B \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --dev-num 1000 \
    --model-type qwen
