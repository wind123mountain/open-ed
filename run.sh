# bash scripts/qwen/ablation/ace/train_0.6B_4B_1l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_2l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_4l.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_0.5.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_1.0.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_3.sh

# bash scripts/qwen/tools/process_data_ace_llama.sh
bash scripts/qwen/sft/sft_llama_1B_ace.sh
# bash scripts/qwen/sft/sft_llama_3B_ace.sh
bash scripts/qwen/span_distillm/ace/train_1B_3B_llama.sh
bash scripts/qwen/distillm/ace/train_1B_3B_llama.sh
bash scripts/qwen/distillm/ace/train_1B_3B_llama_csd.sh
# bash scripts/qwen/tools/process_data_ace_gemma.sh
# bash scripts/qwen/sft/sft_gemma_1B_ace.sh
# bash scripts/qwen/sft/sft_gemma_4B_ace.sh
# bash scripts/qwen/span_distillm/ace/train_1B_4B_gemma.sh
# bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.1.sh
# bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.3.sh
# bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.5.sh
# bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_0.7.sh
# bash scripts/qwen/ablation/ace/kd_ratio/train_0.6B_4B_1.0.sh

# bash scripts/qwen/sft/sft_llama_3B_ace.sh
# bash scripts/qwen/distillm/ace/train_1B_3B_llama.sh
# bash scripts/qwen/distillm/ace/train_1B_3B_llama_csd.sh
# bash scripts/qwen/span_distillm/ace/train_1B_3B_llama.sh

# # Sweep 5 ratio
# for r in 0.1 0.3 0.5 0.7 1.0; do
#     bash scripts/qwen/ablation/geneva/kd_ratio/train_0.6B_4B_${r}.sh
# done


# (a) Convert MINION raw BIO → EventKD JSON format
# python3 gen_data_minion.py --lang all

# (b) Tokenize per language với Qwen3 tokenizer
# bash scripts/qwen/tools/process_data_minion.sh spanish
# bash scripts/qwen/tools/process_data_minion.sh portuguese

# (c) SFT teacher (4B) + student baseline (0.6B) per language
# bash scripts/qwen/sft/minion/sft_qwen3_4B_minion.sh spanish
# bash scripts/qwen/sft/minion/sft_qwen3_0.6B_minion.sh spanish
# bash scripts/qwen/sft/minion/sft_qwen3_4B_minion.sh portuguese
# bash scripts/qwen/sft/minion/sft_qwen3_0.6B_minion.sh portuguese

# (d) EventKD distill per language
#     LƯU Ý: 2 script này hardcode step number của teacher PEFT (246 / 114).
#     Sau khi (c) chạy xong, verify: ls results/qwen3/sft_4B_minion_<lang>/e3-*/
#     Nếu step thực ≠ 246 (spanish) hoặc 114 (portuguese) thì sửa giá trị hardcode trong script.
# bash scripts/qwen/span_distillm/minion/train_0.6B_4B_minion_spanish.sh
# bash scripts/qwen/span_distillm/minion/train_0.6B_4B_minion_portuguese.sh

# (e) DistiLLM SFKL baseline (Spanish) — for cross-method comparison vs EventKD
# bash scripts/qwen/distillm/minion/train_0.6B_4B_minion_spanish.sh
# bash scripts/qwen/ablation/ace/train_0.6B_4B_relation.sh
