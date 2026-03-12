
export TF_CPP_MIN_LOG_LEVEL=3


# prompt and response for baselines
# PYTHONPATH=. python3 ./tools/process_data.py \
#     --data-dir ./data/ace/ \
#     --processed-data-dir ./processed_data/ace \
#     --model-path Qwen/Qwen3-0.6B \
#     --data-process-workers 4 \
#     --max-prompt-length 460 \
#     --dev-num 1000 \
#     --model-type qwen

for i in {0..4}
do
    PYTHONPATH=. python3 ./tools/process_data.py \
        --data-dir ./data/ace/${i}/ \
        --processed-data-dir ./processed_data/ace/${i} \
        --model-path Qwen/Qwen3-0.6B \
        --data-process-workers 4 \
        --max-prompt-length 460 \
        --dev-num 1000 \
        --model-type qwen
done