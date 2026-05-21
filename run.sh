# === AMiD baseline (Llama 1B/8B) trên ACE — chưa chạy ===
bash scripts/qwen/distillm/ace/train_1B_8B_llama_amid.sh

# === AMiD baseline (Qwen 0.6B/4B) — ACE và GENEVA stuck từ run trước, cần retry ===
bash scripts/qwen/distillm/ace/train_0.6B_4B_amid.sh
bash scripts/qwen/distillm/geneva/train_0.6B_4B_amid.sh

# === EventKD GENEVA (Llama) với kd-ratio=0.7 — tune lại để vượt DistiLLM SFKL ===
bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh
