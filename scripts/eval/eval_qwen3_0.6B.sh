#! /bin/bash

SEED=42

# ==== Định nghĩa các biến ====
BASE_PATH=.
MODEL_PATH="results/qwen3/sft_0.6B/e10-bs8-lr5e-05-G2-N2-NN1/980"
OUTPUT_DIR="${BASE_PATH}/eval_outputs/${MODEL_PATH}"


mkdir -p ${OUTPUT_DIR}

OPTS=""

# training
OPTS+=" --val_batch_size 64"

# devices
OPTS+=" --student_device cuda:0"

# models
OPTS+=" --model_type qwen"
OPTS+=" --output_dir ${OUTPUT_DIR}"

OPTS+=" --bf16"
OPTS+=" --seed ${SEED}"
OPTS+=" --model_path ${MODEL_PATH}"
# OPTS+=" --lora_path "
OPTS+=" --tokenizer Qwen/Qwen3-0.6B"

python run_eval.py ${OPTS}
# python run_eval.py ${OPTS} >> ${OUTPUT_DIR}/eval.log
