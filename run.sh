# === Llama 1B/8B pipeline trên MAVEN ===
bash scripts/qwen/tools/process_data_maven_llama.sh
bash scripts/qwen/sft/sft_llama_1B_maven.sh
bash scripts/qwen/sft/sft_llama_8B_maven.sh
bash scripts/qwen/span_distillm/maven/train_1B_8B_llama.sh
bash scripts/qwen/distillm/maven/train_1B_8B_llama.sh
bash scripts/qwen/distillm/maven/train_1B_8B_llama_csd.sh
bash scripts/qwen/distillm/maven/train_1B_8B_llama_amid.sh

# === Llama 1B/8B pipeline trên GENEVA ===
bash scripts/qwen/tools/process_data_geneva_llama.sh
bash scripts/qwen/sft/sft_llama_1B_geneva.sh
bash scripts/qwen/sft/sft_llama_8B_geneva.sh
bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh
bash scripts/qwen/distillm/geneva/train_1B_8B_llama.sh
bash scripts/qwen/distillm/geneva/train_1B_8B_llama_csd.sh
bash scripts/qwen/distillm/geneva/train_1B_8B_llama_amid.sh

# === AMiD baseline (Qwen 0.6B/4B) trên 3 dataset — dùng teacher đã SFT sẵn ===
bash scripts/qwen/distillm/ace/train_0.6B_4B_amid.sh
bash scripts/qwen/distillm/maven/train_0.6B_4B_amid.sh
bash scripts/qwen/distillm/geneva/train_0.6B_4B_amid.sh

# === AMiD baseline (Llama 1B/8B) trên ACE — teacher đã SFT từ rebuttal ===
bash scripts/qwen/distillm/ace/train_1B_8B_llama_amid.sh
