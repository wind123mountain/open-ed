# bash scripts/qwen/ablation/ace/train_0.6B_4B_1l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_2l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_4l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_0.5.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_1.0.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_3.sh

bash scripts/qwen/tools/process_data_ace_llama.sh
bash scripts/qwen/sft/sft_llama_3B_ace.sh
bash scripts/qwen/span_distillm/ace/train_1B_3B_llama.sh
bash scripts/qwen/tools/process_data_ace_gemma.sh
bash scripts/qwen/sft/sft_gemma_4B_ace.sh
bash scripts/qwen/span_distillm/ace/train_1B_4B_gemma.sh
bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.1.sh
bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.3.sh
bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.5.sh
bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.7.sh
bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_1.0.sh
