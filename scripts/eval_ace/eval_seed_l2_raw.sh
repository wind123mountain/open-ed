#! /bin/bash
# Eval checkpoint span_distillm 0.6B_4B_ace_srkl_2_l2_raw voi 2 seed: 10 va 50
BASE=Qwen/Qwen3-0.6B
DEVICE=cuda:0

METHODS=(eventkd_l2_raw)
declare -A CKPT=(
  [eventkd_l2_raw]="results/qwen3/span_distillm/0.6B_4B_ace_srkl_2_l2_raw/490"
)

for SEED in 10 50; do
  for M in "${METHODS[@]}"; do
    LORA="${CKPT[$M]}"
    if [ ! -f "$LORA/adapter_model.bin" ]; then
      echo "SKIP $M (seed $SEED): $LORA khong ton tai"; continue
    fi
    OUT="eval_outputs/ace/${M}/seed${SEED}"
    mkdir -p "$OUT"
    echo "==== $M | seed $SEED -> $OUT ===="
    python run_eval.py \
      --model_path "$BASE" \
      --lora_path "$LORA" \
      --tokenizer Qwen/Qwen3-0.6B \
      --model_type qwen \
      --data_dir processed_data/ace/qwen/ \
      --dataset_name ace \
      --val_batch_size 16 --bf16 \
      --student_device "$DEVICE" \
      --seed "$SEED" \
      --output_dir "$OUT"
  done
done
