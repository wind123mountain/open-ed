#! /bin/bash
# Eval toan bo baseline Geneva cho 2 seed: 10 va 50
# Sau khi sua run_eval.py (seeds=[args.seed]), --seed moi tac dung len generation.
BASE=Qwen/Qwen3-0.6B
DEVICE=cuda:0
DATASET_NAME=geneva
DATA_DIR=processed_data/geneva/qwen/

# thu tu theo bang. path = lora adapter (step /305)
METHODS=(sft kd rkl sfkl fdd distillm csd amid eventkd)
declare -A CKPT=(
#   [sft]="results/qwen3/sft_0.6B_geneva_lora/e5-bs8-lr5e-05-G2-N2-NN1-lora-8-64-0.1/305"
#   [kd]="results/qwen3/distillm_0.6B_4B_geneva_kd/305"
#   [rkl]="results/qwen3/distillm_0.6B_4B_geneva_rkl/305"
#   [sfkl]="results/qwen3/distillm_0.6B_4B_geneva_srkl/305"
#   [fdd]="results/qwen3/fdd/0.6B_4B_geneva/305"
#   [distillm]="results/qwen3/distillm_0.6B_4B_geneva_srkl_2/305"
#   [csd]="results/qwen3/distillm_0.6B_4B_geneva_csd/305"
#   [amid]="results/qwen3/distillm_0.6B_4B_geneva_amid/305"
  [eventkd]="results/qwen3/span_distillm/0.6B_4B_geneva_srkl_2/244"
)

for SEED in 10 50; do
  for M in "${METHODS[@]}"; do
    LORA="${CKPT[$M]}"
    if [ -z "$LORA" ]; then
      echo "SKIP $M (seed $SEED): khong co ckpt duoc cau hinh"; continue
    fi
    if [ ! -f "$LORA/adapter_model.bin" ]; then
      echo "SKIP $M (seed $SEED): $LORA khong ton tai"; continue
    fi
    OUT="eval_outputs/${DATASET_NAME}/${M}/seed${SEED}"
    mkdir -p "$OUT"
    echo "==== $M | seed $SEED -> $OUT ===="
    python run_eval.py \
      --model_path "$BASE" \
      --lora_path "$LORA" \
      --tokenizer Qwen/Qwen3-0.6B \
      --model_type qwen \
      --data_dir "$DATA_DIR" \
      --dataset_name "$DATASET_NAME" \
      --val_batch_size 64 --bf16 \
      --student_device "$DEVICE" \
      --seed "$SEED" \
      --output_dir "$OUT"
  done
done
