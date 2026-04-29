#!/bin/bash
# Usage: bash scripts/qwen/tools/process_data_v6.sh <pair_mode>
# Example: bash scripts/qwen/tools/process_data_v6.sh same_prefix

PAIR_MODE=${1:?Usage: $0 <pair_mode> (same_prefix|cross_prefix|mixed|random)}

export TF_CPP_MIN_LOG_LEVEL=3

for ds in ace geneva maven rams; do
    for p in {0..4}; do
        for i in {0..4}; do
            DATA_DIR="./data/${ds}_v6_${PAIR_MODE}_${p}/${i}/"
            PROC_DIR="./processed_data/${ds}_v6_${PAIR_MODE}_${p}/${i}"
            if [ ! -d "${DATA_DIR}" ]; then
                continue
            fi
            PYTHONPATH=. python3 ./tools/process_data.py \
                --data-dir "${DATA_DIR}" \
                --processed-data-dir "${PROC_DIR}" \
                --model-path Qwen/Qwen3-0.6B \
                --data-process-workers 4 \
                --max-prompt-length 460 \
                --t-max-prompt-length 640 \
                --dev-num 1000 \
                --model-type qwen
        done
    done
done
