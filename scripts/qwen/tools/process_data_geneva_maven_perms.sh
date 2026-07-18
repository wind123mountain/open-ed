export TF_CPP_MIN_LOG_LEVEL=3
cd /home/hungpv/projects/OpenED

for ds in geneva maven
do
    for p in 0 1 2 3 4
    do
        for i in 0 1 2 3 4
        do
            PYTHONPATH=. ~/miniconda3/envs/mta/bin/python ./tools/process_data.py \
                --data-dir ./data/${ds}_perm${p}/${i}/ \
                --processed-data-dir ./processed_data/${ds}_perm${p}/${i} \
                --model-path Qwen/Qwen3-0.6B \
                --data-process-workers 4 \
                --max-prompt-length 460 \
                --t-max-prompt-length 640 \
                --dev-num 1000 \
                --model-type qwen || echo "FAILED ${ds} perm${p} task${i}"
        done
    done
done
echo "ALL DONE"
