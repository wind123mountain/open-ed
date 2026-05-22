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

# === Sweep 6 combo (kd-ratio × w-span) trên GENEVA Llama ===
# Baseline (Run #1 lr=2e-4, kd=0.9, w=2.0) → test trigger 66.30 / arg 42.74 (thua DistiLLM SFKL 67.49/43.02)
# Mục tiêu: vượt DistiLLM. Tất cả run dưới dùng LR=1e-4 hiện tại của script.

# K1: lower β (more CE anchor), nhắm boost trigger
KD_RATIO=0.7 W_SPAN=2.0 TAG=K1-kd0.7-w2.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh

# K2: even lower β, test xem GENEVA có cần CE mạnh hơn không
KD_RATIO=0.5 W_SPAN=2.0 TAG=K2-kd0.5-w2.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh

# K3: lighter span loss, giữ β cao
KD_RATIO=0.9 W_SPAN=1.0 TAG=K3-kd0.9-w1.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh

# K4: stronger span loss, nhắm boost argument F1
KD_RATIO=0.9 W_SPAN=3.0 TAG=K4-kd0.9-w3.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh

# K5: lower β + lighter span (combined balance)
KD_RATIO=0.7 W_SPAN=1.0 TAG=K5-kd0.7-w1.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh

# K6: lower β + stronger span (cover opposite của K2)
KD_RATIO=0.7 W_SPAN=3.0 TAG=K6-kd0.7-w3.0 bash scripts/qwen/span_distillm/geneva/train_1B_8B_llama.sh
