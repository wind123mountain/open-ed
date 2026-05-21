# === Download teacher PEFT từ HuggingFace nếu chưa có (cho Qwen GENEVA AMiD) ===
download_teacher() {
    local DEST="$1"
    local URL="$2"
    if [ ! -f "${DEST}/adapter_config.json" ]; then
        echo "Downloading teacher PEFT to ${DEST}..."
        mkdir -p "${DEST}"
        for f in adapter_config.json adapter_model.bin added_tokens.json chat_template.jinja merges.txt special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json README.md; do
            curl -L --fail -o "${DEST}/${f}" "${URL}/${f}" || { echo "FAILED to download ${f}"; exit 1; }
        done
        echo "Teacher PEFT downloaded to ${DEST}."
    fi
}

download_teacher \
    "results/qwen3/sft_4B_geneva/e5-bs2-lr0.0003-G8-N2-NN1-lora-64-128-0.05/305" \
    "https://huggingface.co/VoCuc/open-ed-ckpt/resolve/main/qwen3/sft_4B_geneva/e5-bs2-lr0.0003-G8-N2-NN1-lora-64-128-0.05/305"

# === AMiD baseline (Llama 1B/8B) trên ACE — chưa chạy ===
bash scripts/qwen/distillm/ace/train_1B_8B_llama_amid.sh

# === AMiD baseline (Qwen 0.6B/4B) GENEVA — retry sau khi đã có teacher PEFT ===
bash scripts/qwen/distillm/geneva/train_0.6B_4B_amid.sh

# === EventKD GENEVA (Llama) với kd-ratio=0.7 — tune lại để vượt DistiLLM SFKL ===
bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh
