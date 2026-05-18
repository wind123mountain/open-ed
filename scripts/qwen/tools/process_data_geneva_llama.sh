
export TF_CPP_MIN_LOG_LEVEL=3


# prompt and response for baselines
PYTHONPATH=. python3 ./tools/process_data.py \
    --data-dir ./data/geneva/ \
    --processed-data-dir ./processed_data/geneva \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --data-process-workers 4 \
    --max-prompt-length 460 \
    --dev-num 1000 \
    --model-type llama
