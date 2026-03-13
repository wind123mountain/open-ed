
export TF_CPP_MIN_LOG_LEVEL=3


for i in {0..4}
do
    PYTHONPATH=. python3 ./tools/process_data.py \
        --data-dir ./data/geneva_v2/${i}/ \
        --processed-data-dir ./processed_data/geneva_v2/${i} \
        --model-path Qwen/Qwen3-0.6B \
        --data-process-workers 4 \
        --max-prompt-length 460 \
        --t-max-prompt-length 640 \
        --dev-num 1000 \
        --model-type qwen
done