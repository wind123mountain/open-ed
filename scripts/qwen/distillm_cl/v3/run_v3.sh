
for p in {0..4}
do
    bash scripts/qwen/distillm_cl/v3/cl_qwen3_0.6B_v3.sh ace ${p} 0.5
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/v3/cl_qwen3_0.6B_v3.sh geneva ${p} 1.0
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/v3/cl_qwen3_0.6B_v3.sh maven ${p} 1.0
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/v3/cl_qwen3_0.6B_v3.sh rams ${p} 0.7
done
