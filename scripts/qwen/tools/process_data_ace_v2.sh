
export TF_CPP_MIN_LOG_LEVEL=3


# # prompt and response for baselines
# PYTHONPATH=. python3 ./tools/process_data.py \
#     --data-dir ./data/ace_v2/ \
#     --processed-data-dir ./processed_data/ace_v2 \
#     --model-path Qwen/Qwen3-0.6B \
#     --data-process-workers 4 \
#     --max-prompt-length 460 \
#     --t-max-prompt-length 640 \
#     --dev-num 1000 \
#     --model-type qwen

for i in {0..4}
do
    PYTHONPATH=. python3 ./tools/process_data.py \
        --data-dir ./data/ace_v2/${i}/ \
        --processed-data-dir ./processed_data/ace_v2/${i} \
        --model-path Qwen/Qwen3-0.6B \
        --data-process-workers 4 \
        --max-prompt-length 460 \
        --t-max-prompt-length 640 \
        --dev-num 1000 \
        --model-type qwen
done