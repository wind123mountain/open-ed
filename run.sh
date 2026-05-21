# === Download teacher PEFT từ HuggingFace nếu chưa có ===
download_teacher() {
    local DEST="$1"
    local URL="$2"
    if [ ! -f "${DEST}/adapter_config.json" ]; then
        echo "Downloading teacher PEFT to ${DEST}..."
        mkdir -p "${DEST}"
        # Essential
        for f in adapter_config.json adapter_model.bin; do
            curl -L --fail -o "${DEST}/${f}" "${URL}/${f}" || { echo "FAILED essential ${f}"; exit 1; }
        done
        # Optional (Llama không có merges.txt/vocab.json/added_tokens.json)
        for f in added_tokens.json chat_template.jinja merges.txt special_tokens_map.json tokenizer_config.json tokenizer.json vocab.json README.md; do
            curl -L --fail -s -o "${DEST}/${f}" "${URL}/${f}" 2>/dev/null || rm -f "${DEST}/${f}"
        done
        echo "Teacher PEFT downloaded to ${DEST}."
    fi
}

# Llama 8B GENEVA teacher — đầy đủ steps để auto-resolve pick last (2460) như run gốc đã work tốt
download_teacher \
    "results/llama/sft_8B_geneva/e5-bs2-lr0.0001-G1-N2-NN1-lora-32-64-0.1/2460" \
    "https://huggingface.co/VoCuc/open-ed-ckpt-v2/resolve/main/llama/sft_8B_geneva/e5-bs2-lr0.0001-G1-N2-NN1-lora-32-64-0.1/2460"

# === EventKD GENEVA (Llama) rerun với LR=1e-4 (giảm từ 2e-4) ===
# kd-ratio=0.9, teacher auto-resolve (step 2460), layer/lora/epochs giữ nguyên
bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh
