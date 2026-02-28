
export TF_CPP_MIN_LOG_LEVEL=3


# prompt and response for baselines
PYTHONPATH=. python3 ./tools/process_data.py \
    --data-dir ./data/geneva_teacher/ \
    --processed-data-dir ./processed_data/geneva_teacher \
    --model-path Qwen/Qwen3-0.6B \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --t-max-prompt-length 640 \
    --dev-num 1000 \
    --model-type qwen