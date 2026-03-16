
for p in {0..4}
do
    bash scripts/qwen/distillm_cl/span/ace_cl_qwen3_0.6B_v3.sh ace ${p}
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/span/cl_qwen3_0.6B_v3.sh geneva ${p}
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/span/cl_qwen3_0.6B_v3.sh maven ${p}
done

for p in {0..4}
do
    bash scripts/qwen/distillm_cl/span/cl_qwen3_0.6B_v3.sh rams ${p}
done
