bash scripts/qwen3/distillm/train_0.6B_4B.sh 

bash scripts/eval/eval_qwen3_4B.sh
bash scripts/eval/eval_qwen3_0.6B_distillm.sh

bash scripts/qwen3/sft/sft_qwen3_0.6B.sh
bash scripts/eval/eval_qwen3_0.6B.sh