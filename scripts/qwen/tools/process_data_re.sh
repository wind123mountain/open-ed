#!/bin/bash
# Usage: bash scripts/qwen/tools/process_data_re.sh <data_prefix>
# Example: bash scripts/qwen/tools/process_data_re.sh fewrel_wm_re_random
#          bash scripts/qwen/tools/process_data_re.sh fewrel_wm_re_random_up3
#          bash scripts/qwen/tools/process_data_re.sh tacred_wm_re_same_prefix

DATA_PREFIX=${1:?Usage: $0 <data_prefix> (e.g. fewrel_wm_re_random)}

export TF_CPP_MIN_LOG_LEVEL=3

for p in {0..4}; do
    for i in {0..9}; do
        DATA_DIR="./data/${DATA_PREFIX}_${p}/${i}/"
        PROC_DIR="./processed_data/${DATA_PREFIX}_${p}/${i}"
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
