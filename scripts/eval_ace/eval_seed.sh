#! /bin/bash
# Eval toan bo baseline ACE cho 2 seed: 10 va 50
# Sau khi sua run_eval.py (seeds=[args.seed]), --seed moi tac dung len generation.
BASE=Qwen/Qwen3-0.6B
DEVICE=cuda:0

# thu tu theo bang. path = lora adapter (step /490)
METHODS=(sft kd rkl sfkl fdd distillm csd amid eventkd eventkd_l2 eventkd_dot eventkd_te2)
declare -A CKPT=(
  # [sft]="results/qwen3/sft_0.6B_ace_lora/e5-bs8-lr5e-05-G2-N2-NN1-lora-8-64-0.1/490"
  # [kd]="results/qwen3/distillm_0.6B_4B_ace_kd/490"
  # [rkl]="results/qwen3/distillm_0.6B_4B_ace_rkl/490"
  # [sfkl]="results/qwen3/distillm_0.6B_4B_ace_srkl/490"
#   [fdd]="results/qwen3/fdd/0.6B_4B_ace/490"
#   [distillm]="results/qwen3/distillm_0.6B_4B_ace_srkl_2/490"
#   [csd]="results/qwen3/distillm_0.6B_4B_ace_csd/490"
#   [amid]="results/qwen3/distillm_0.6B_4B_ace_amid/490"
  # [eventkd]="results/qwen3/span_distillm/0.6B_4B_ace_srkl_2/490"
  # [eventkd_l2]="results/qwen3/span_distillm/0.6B_4B_ace_srkl_2_l2/490"
  [eventkd_te2]="results/qwen3/span_distillm/0.6B_4B_ace_srkl_2_te2/490"
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
