# === Rerun SFT student 1B với Llama-3.2-1B-Instruct ===
bash scripts/qwen/sft/sft_llama_1B_ace.sh
bash scripts/qwen/sft/sft_llama_1B_maven.sh
bash scripts/qwen/sft/sft_llama_1B_geneva.sh

# === AMiD baseline (Llama 1B/8B) trên ACE — chưa chạy ===
bash scripts/qwen/distillm/ace/train_1B_8B_llama_amid.sh

# === AMiD baseline (Qwen 0.6B/4B) trên 3 dataset — chưa chạy ===
bash scripts/qwen/distillm/ace/train_0.6B_4B_amid.sh
bash scripts/qwen/distillm/maven/train_0.6B_4B_amid.sh
bash scripts/qwen/distillm/geneva/train_0.6B_4B_amid.sh
